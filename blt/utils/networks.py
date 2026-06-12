import torch
from torch import nn


# Utilities for defining neural nets
def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    """mlp with hidden_depth layers of size hidden_dim
       input to network is of size input_dim and output of size output_dim"""
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


# Define the forward model
class MlpPredictor(nn.Module):
    """mlp predictor"""
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(input_dim, hidden_dim, output_dim, hidden_depth)
        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs):
        next_pred = self.trunk(obs)
        return next_pred


class BilinearPredictor(nn.Module):
    """Bilinear Transduction
       dot product of non linear embeddings of data point and data point difference (delta)
       data and delta have same dim (e.g. difference calculated by subtraction)"""
    def __init__(self, input_dim, output_dim, hidden_dim, feature_dim, hidden_depth):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.obs_trunk = mlp(input_dim, hidden_dim, feature_dim*output_dim, hidden_depth)
        self.delta_trunk = mlp(input_dim, hidden_dim, feature_dim*output_dim, hidden_depth)

    def forward(self, obs, deltas):
        ob_embedding = self.obs_trunk(obs)
        ob_embedding = torch.reshape(ob_embedding, (-1, self.output_dim, self.feature_dim)) #output_dim x feature_dim
        delta_embedding = self.delta_trunk(deltas)
        delta_embedding = torch.reshape(delta_embedding, (-1, self.feature_dim, self.output_dim)) #feature_dim x output_dim
        pred = torch.diagonal(torch.matmul(ob_embedding, delta_embedding), dim1=1, dim2=2)
        return pred


class TransductionPredictor(nn.Module):
    """Plain-MLP transduction: MLP([obs ‖ delta]).
       BLT ablation — same (anchor, delta) reparametrization and transducer-based eval as
       BilinearPredictor, but an unstructured feed-forward readout instead of the bilinear
       low-rank form, so an MLP / Transduction / BLT comparison isolates the bilinear structure
       as the lone variable. delta has the same dim as obs (subtraction), so the trunk input is
       2*input_dim."""
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_depth):
        super().__init__()
        self.trunk = mlp(2 * input_dim, hidden_dim, output_dim, hidden_depth)
        self.apply(weight_init)

    def forward(self, obs, deltas):
        return self.trunk(torch.cat([obs, deltas], dim=-1))


class BilinearPredictorScalarDelta(nn.Module):
    """Bilinear Transduction, scalar delta
       dot product of non linear embeddings of data point and data point difference (delta)
       delta has dim 1 (e.g. difference calculated by cosine similarity)"""
    def __init__(self, input_dim, output_dim, hidden_dim, feature_dim, hidden_depth):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.obs_trunk = mlp(input_dim, hidden_dim, feature_dim*output_dim, hidden_depth)

        mods = [nn.Linear(1, hidden_dim//4), nn.ReLU(inplace=True)]
        mods += [nn.Linear(hidden_dim//4, hidden_dim//2), nn.ReLU(inplace=True)]
        mods += [nn.Linear(hidden_dim//2, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, feature_dim*output_dim))
        self.delta_trunk = nn.Sequential(*mods)

    def forward(self, obs, deltas):
        ob_embedding = self.obs_trunk(obs)
        ob_embedding = torch.reshape(ob_embedding, (-1, self.output_dim, self.feature_dim)) #output_dim x feature_dim
        delta_embedding = self.delta_trunk(deltas)
        delta_embedding = torch.reshape(delta_embedding, (-1, self.feature_dim, self.output_dim)) #feature_dim x output_dim
        pred = torch.diagonal(torch.matmul(ob_embedding, delta_embedding), dim1=1, dim2=2)
        return pred