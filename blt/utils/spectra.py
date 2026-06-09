"""Realized-rank diagnostics for the bilinear trunks, evaluated at the current weights.

The bilinear predictor scores ``diag(obs_trunk(x) @ delta_trunk(delta))`` — a sum of ``feature_dim``
rank-1 products, so ``feature_dim`` is the *nominal* rank cap. This probe measures the *realized*
soft rank of the two factor matrices ``F = obs_trunk(anchor)``, ``G = delta_trunk(delta)`` over the
training distribution, via scale-free spectral summaries (stable rank, participation ratio,
effective rank) plus the theory's sigma_p non-degeneracy proxy. Called every N epochs from the
trainer so the rank's *evolution during training* is visible.
"""

import numpy as np
import torch

from utils.util import get_deltas


def _spectrum_stats(S):
    """Scale-free soft-rank descriptors from singular values ``S`` (descending). lam = S**2."""
    lam = S.double().pow(2)
    p = lam / lam.sum()
    return {
        'n_sv': int(S.numel()),
        'stable_rank': float(lam.sum() / lam[0]),                  # ||S||_F^2 / ||S||_2^2
        'participation_ratio': float(lam.sum().pow(2) / lam.pow(2).sum()),
        'effective_rank': float(torch.exp(-(p * (p + 1e-30).log()).sum())),  # exp(spectral entropy)
        'top_eig_share': float(lam[0] / lam.sum()),
        'sv_top': float(S[0]),
        'sv_min': float(S[-1]),
    }


def _sym_sqrt(C):
    """PSD symmetric square root via eigendecomposition (double precision)."""
    w, V = torch.linalg.eigh(C.double())
    return V @ torch.diag(w.clamp_min(0).sqrt()) @ V.T


def _draw_pairs(n, size, train_Y, skew, rng):
    """Random directed train pairs (t1, t2) with the same skew reorder the trainer uses, so G is
    built from the *training* delta distribution."""
    t1 = rng.randint(n, size=(size,))
    t2 = rng.randint(n, size=(size,))
    if skew == 'right':  # t2 higher than t1
        swap = (train_Y[t1] > train_Y[t2]).flatten()
        t1[swap], t2[swap] = t2[swap], t1[swap]
    elif skew == 'left':
        swap = (train_Y[t1] < train_Y[t2]).flatten()
        t1[swap], t2[swap] = t2[swap], t1[swap]
    return t1, t2


def _rows(train_X, idx, obs_idxs, device):
    return torch.Tensor(np.stack([train_X[c][obs_idxs] for c in idx])).float().to(device)


def factor_spectra(model, train_X, train_Y, obs_idxs, similarity_type, skew, sample, device,
                   seed=0):
    """Soft rank of ``F=obs_trunk(anchor)`` and ``G=delta_trunk(delta)`` at the current weights.

    Uses a fixed local RNG seed so the *same* anchors/deltas are scored every call — the rank
    evolution then reflects weight change alone, not resampling noise. Returns per-factor spectrum
    stats + the sigma_p proxy (sigma_min of ``E[ffT]^.5 E[ggT]^.5``)."""
    rng = np.random.RandomState(seed)
    n = len(train_X)
    ns = min(sample, n)
    obs_i = rng.choice(n, size=ns, replace=False)
    t1, t2 = _draw_pairs(n, ns, train_Y, skew, rng)

    anchors = _rows(train_X, obs_i, obs_idxs, device)
    deltas = get_deltas(_rows(train_X, t1, obs_idxs, device),
                        _rows(train_X, t2, obs_idxs, device), similarity_type)

    was_training = model.training
    model.eval()
    with torch.no_grad():
        F = model.obs_trunk(anchors).reshape(ns, -1)    # [ns, feature_dim] (output_size=1)
        G = model.delta_trunk(deltas).reshape(ns, -1)
    if was_training:
        model.train()

    # Centered spectra describe each factor's spread (collapse => low participation ratio).
    sf = torch.linalg.svdvals((F - F.mean(0)).float())
    sg = torch.linalg.svdvals((G - G.mean(0)).float())
    # sigma_p proxy uses raw second moments E[ffT], E[ggT] (the theory's non-centered form).
    sp = torch.linalg.svdvals(_sym_sqrt(F.T @ F / ns) @ _sym_sqrt(G.T @ G / ns))

    return {
        'sample': ns,
        'F_obs_trunk': _spectrum_stats(sf),
        'G_delta_trunk': _spectrum_stats(sg),
        'sigp_proxy': {
            'sigma_min': float(sp[-1]),
            'sigma_max': float(sp[0]),
            'participation_ratio': float(sp.pow(2).sum().pow(2) / sp.pow(2).pow(2).sum()),
        },
    }
