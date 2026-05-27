import torch.optim as optim
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp
import random
import warnings
warnings.filterwarnings('ignore')

from utils.util import models_save, get_deltas, AvgMeter, eval_supervised
from utils.transducers import define_transducer


def _snapshot_rng():
    """Save global Python/NumPy/Torch RNG state so periodic eval can't perturb training."""
    state = {
        'numpy': np.random.get_state(),
        'random': random.getstate(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state()
    return state


def _restore_rng(state):
    np.random.set_state(state['numpy'])
    random.setstate(state['random'])
    torch.set_rng_state(state['torch'])
    if 'cuda' in state:
        torch.cuda.set_rng_state(state['cuda'])


def _periodic_eval(model, model_type, logdir, samples, train_deltas_acc, similarity_type,
                   skew, n_approx_train_deltas, sample_deltas, sample_train, type_idxs,
                   use_dom_know_eval):
    """Snapshot-transducer eval on id (eval_*) and ood (ood_*) splits — quiet, no disk writes."""
    train_deltas_np = np.array(train_deltas_acc).reshape(-1, samples['train_X'].shape[-1]) if train_deltas_acc else []
    transducer = define_transducer(samples, train_deltas_np, skew, n_approx_train_deltas,
                                   sample_deltas=sample_deltas, sample_train=sample_train,
                                   type_idxs=type_idxs, similarity_type=similarity_type, verbose=False)
    model.eval()
    with torch.no_grad():
        id_preds = eval_supervised(
            model_type, model, logdir,
            {'test_X': samples['eval_X'], 'test_Y': samples['eval_Y'], 'test_formula': samples['eval_formula']},
            similarity_type, transducer=transducer, use_dom_know_eval=use_dom_know_eval,
            eval_type='val', verbose=False, write_results=False,
        )
        ood_preds = eval_supervised(
            model_type, model, logdir,
            {'test_X': samples['ood_X'], 'test_Y': samples['ood_Y'], 'test_formula': samples['ood_formula']},
            similarity_type, transducer=transducer, use_dom_know_eval=use_dom_know_eval,
            eval_type='ood', verbose=False, write_results=False,
        )
    model.train()
    return id_preds, ood_preds


def train_supervised(model_type, dataset, model, logdir, obs_idxs, skew, \
                     num_epochs=500, batch_size=32, checkpoint_path=None, store_train_deltas=True,
                     similarity_type=None, wandb_run=None, eval_every=0,
                     periodic_eval_kwargs=None):
    """train model. If wandb_run is set, logs train/loss per epoch. If eval_every>0 and
    periodic_eval_kwargs is provided, runs id+ood eval every N epochs (snapshot transducer,
    RNG-isolated)."""

    X, Y = dataset['train_X'], dataset['train_Y']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = optim.Adam(list(model.parameters()))
    epoch_losses = []
    eval_history = {'epoch': [], 'id_mae': [], 'id_sem': [], 'ood_mae': [], 'ood_sem': []}
    idxs = np.array(range(len(X)))
    num_batches = len(idxs) // batch_size
    print('num_epochs', num_epochs, 'num_batches', num_batches)
    train_deltas = []

    # Train the model with regular SGD
    for epoch in range(num_epochs):
        loss_meter = AvgMeter()
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()

            t1_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of A
            t2_idx = np.random.randint(len(X), size=(batch_size,)) # Indices of sample B
            # priviledged training, knowledge of the Y distribution's skewness (transformation type/object class type)
            if skew=='right': #t2 have higher ys than t1
                swap_idxs = (Y[t1_idx] > Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]
            elif skew=='left':
                swap_idxs = (Y[t1_idx] < Y[t2_idx]).flatten()
                t1_idx[swap_idxs], t2_idx[swap_idxs] = t2_idx[swap_idxs], t1_idx[swap_idxs]

            t1_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t1_idx])).float().to(device)
            t1_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t1_idx])).float().to(device)
            t2_X = torch.Tensor(np.concatenate([X[c_idx][obs_idxs][None] for c_idx in t2_idx])).float().to(device)
            t2_Y = torch.Tensor(np.concatenate([Y[c_idx][None] for c_idx in t2_idx])).float().to(device)

            if model_type == 'mlp':
                y1_pred = model(t1_X)
                loss = torch.mean(torch.linalg.norm(y1_pred - t1_Y, dim=-1))
            elif 'bilinear' in model_type:
                deltas = get_deltas(t1_X, t2_X, similarity_type)
                if store_train_deltas:
                    delta_idx = np.random.randint(len(deltas), size=(1,))[0] # store 1 delta every batch
                    train_deltas.append(deltas[delta_idx].cpu().detach().numpy())
                y2_pred = model(t1_X, deltas)
                loss = torch.mean(torch.linalg.norm(y2_pred - t2_Y, dim=-1)) # MAE
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loss_meter.update(loss.item(), batch_size)
            if (i+1) % 5 == 0:
                print('[%d, %5d] loss: %.8f' %
                    (epoch+1, i+1, running_loss/(i+1)))

        epoch_loss = running_loss/num_batches
        epoch_losses.append(epoch_loss)
        if (epoch+1) % 2000 == 0 and checkpoint_path:
            models_save(model, logpath=osp.join(checkpoint_path, str(epoch)+'.pt'))

        log = {'misc/epoch': epoch+1, 'train/mae': epoch_loss}

        # Periodic id/ood eval (RNG-isolated so the no-wandb path remains byte-identical)
        if eval_every and periodic_eval_kwargs and ((epoch+1) % eval_every == 0 or (epoch+1) == num_epochs):
            rng_state = _snapshot_rng()
            try:
                id_preds, ood_preds = _periodic_eval(
                    model, model_type, logdir, train_deltas_acc=train_deltas,
                    **periodic_eval_kwargs,
                )
            finally:
                _restore_rng(rng_state)
            eval_history['epoch'].append(epoch+1)
            eval_history['id_mae'].append(id_preds['mae'])
            eval_history['id_sem'].append(id_preds['sem'])
            eval_history['ood_mae'].append(ood_preds['mae'])
            eval_history['ood_sem'].append(ood_preds['sem'])
            log.update({
                'val/id_mae': id_preds['mae'],
                'val/id_sem': id_preds['sem'],
                'val/ood_mae': ood_preds['mae'],
                'val/ood_sem': ood_preds['sem'],
            })
            print(f'[periodic-eval epoch {epoch+1}] id MAE {id_preds["mae"]:.4f} ± {id_preds["sem"]:.4f}  ood MAE {ood_preds["mae"]:.4f} ± {ood_preds["sem"]:.4f}')

        if wandb_run is not None:
            wandb_run.log(log, step=epoch+1)

    plt.figure()
    plt.xlabel('epochs')
    plt.ylabel('MAE')
    plt.plot(epoch_losses)
    losses_png = os.path.join(logdir, model_type+'_losses.png')
    plt.savefig(losses_png)
    plt.close()
    print('Finished Training')

    # id/ood evolution plot
    if eval_history['epoch']:
        plt.figure()
        plt.errorbar(eval_history['epoch'], eval_history['id_mae'], yerr=eval_history['id_sem'], label='id (eval)', marker='o')
        plt.errorbar(eval_history['epoch'], eval_history['ood_mae'], yerr=eval_history['ood_sem'], label='ood', marker='s')
        plt.xlabel('epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.title('id/ood MAE over training')
        evo_png = os.path.join(logdir, model_type+'_eval_evolution.png')
        plt.savefig(evo_png)
        plt.close()
        if wandb_run is not None:
            import wandb
            wandb_run.log({'plots/final/eval_evolution': wandb.Image(evo_png)})

    return model, np.array(train_deltas).reshape(-1, len(obs_idxs))
