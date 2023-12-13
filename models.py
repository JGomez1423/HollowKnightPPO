import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.ops.misc import SqueezeExcitation as SE
import torch.optim as optim
import os
from torch.distributions.categorical import Categorical

def param_init(m):  # code adapted from torchvision VGG class
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)


class NoisyLinear(nn.Module):
    """
    NoisyLinear code adapted from
    https://github.com/toshikwa/fqf-iqn-qrdqn.pytorch/blob/master/fqf_iqn_qrdqn/network.py
    """

    def __init__(self, in_features, out_features, sigma=0.5):
        super(NoisyLinear, self).__init__()

        # Learnable parameters.
        self.mu_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.sigma_W = nn.Parameter(
            torch.FloatTensor(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.FloatTensor(out_features))
        self.sigma_bias = nn.Parameter(torch.FloatTensor(out_features))

        # Factorized noise parameters.
        self.register_buffer('eps_p', torch.FloatTensor(in_features))
        self.register_buffer('eps_q', torch.FloatTensor(out_features))

        self.in_features = in_features
        self.out_features = out_features
        self.sigma = sigma

        self._noise_mode = True

        self.reset_param()
        self.reset_noise()

    @staticmethod
    def _f(x):
        return x.normal_().sign().mul(x.abs().sqrt())

    def reset_param(self):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_W.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_W.data.fill_(self.sigma / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(self.sigma / np.sqrt(self.out_features))

    def reset_noise(self):
        self.eps_p.copy_(self._f(self.eps_p))
        self.eps_q.copy_(self._f(self.eps_q))

    def noise_mode(self, mode):
        self._noise_mode = mode

    def forward(self, x):
        if self._noise_mode:
            weight = self.mu_W + self.sigma_W * self.eps_q.ger(self.eps_p)
            bias = self.mu_bias + self.sigma_bias * self.eps_q.clone()
        else:
            weight = self.mu_W
            bias = self.mu_bias

        return F.linear(x, weight, bias)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(BasicBlock, self).__init__()
        self.act = nn.ReLU(inplace=True)
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            self.act,
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(stride, stride),
            nn.Conv2d(in_channels, out_channels, 1)
        ) if stride > 1 or in_channels != out_channels else nn.Identity()

    def forward(self, x):
        feat_map = self.convs(x)
        shortcut = self.shortcut(x)
        x = feat_map + shortcut
        x = self.act(x)
        return x


class AbstractExtractor(nn.Module):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(AbstractExtractor, self).__init__()

    def forward(self, x):
        raise NotImplementedError


class ResidualExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(ResidualExtractor, self).__init__(obs_shape, n_frames)
        act = nn.ReLU(inplace=True)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 48, 4, 4),
            act,
            BasicBlock(48, 48),
            BasicBlock(48, 96, 2),
            BasicBlock(96, 160, 2),
            BasicBlock(160, 160),
            BasicBlock(160, 256, 2),
            BasicBlock(256, 256),
            nn.Conv2d(256, 1024, 1),
            act,
            nn.AvgPool2d(tuple(out_shape))
        )
        self.units = 1024

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class SimpleExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(SimpleExtractor, self).__init__(obs_shape, n_frames)
        act = nn.ReLU(inplace=True)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(160, 320, kernel_size=3, stride=2, padding=1),
            act,
            nn.Flatten(),
        )
        self.units = 320 * np.prod(out_shape)

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class AttentionExtractor(AbstractExtractor):
    def __init__(self, obs_shape: tuple, n_frames: int):
        super(AttentionExtractor, self).__init__(obs_shape, n_frames)
        act = nn.ReLU(inplace=True)
        out_shape = np.array(obs_shape, dtype=int)
        out_shape //= 32
        self.convs = nn.Sequential(
            nn.Conv2d(n_frames, 32, kernel_size=5, stride=2, padding=2),
            act,
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            act,
            SE(96, 12, activation=lambda: act),
            nn.Conv2d(96, 160, kernel_size=3, stride=2, padding=1),
            act,
            SE(160, 16, activation=lambda: act),
            nn.Conv2d(160, 256, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(256, 960, kernel_size=1),
            act,
            nn.AvgPool2d(tuple(out_shape)),
        )
        self.units = 960

        for m in self.modules():
            param_init(m)

    def forward(self, x):
        x = self.convs(x)
        return x


class AbstractFullyConnected(nn.Module):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(AbstractFullyConnected, self).__init__()
        self.noisy = nn.ModuleList()
        self.linear_cls = NoisyLinear if noisy else nn.Linear
        self.extractor = extractor
        self.act = nn.ReLU(inplace=True)

    def reset_noise(self):
        for layer in self.noisy:
            layer.reset_noise()

    def noise_mode(self, mode):
        for layer in self.noisy:
            layer.noise_mode(mode)

    def forward(self, x, **kwargs):
        raise NotImplementedError


class SinglePathMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(SinglePathMLP, self).__init__(extractor, n_out, noisy)
        self.linear = self.linear_cls(extractor.units, 512)
        self.out = self.linear_cls(512, n_out)
        if noisy:
            self.noisy.append(self.linear)
            self.noisy.append(self.out)

        param_init(self.linear)
        param_init(self.out)

    def forward(self, x, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        x = self.act(x)
        x = self.out(x)
        return x


class DuelingMLP(AbstractFullyConnected):
    def __init__(self, extractor: nn.Module, n_out: int, noisy=False):
        super(DuelingMLP, self).__init__(extractor, n_out, noisy)
        self.linear_val = self.linear_cls(extractor.units, 320)
        self.adv = self.linear_cls(320, n_out)

        if noisy:
            self.noisy.append(self.linear_adv)
            self.noisy.append(self.adv)

        param_init(self.linear_adv)
        param_init(self.adv)

    def forward(self, x, adv_only=False, **kwargs):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        adv = self.linear_adv(x)
        adv = self.act(adv)
        adv = self.adv(adv)
        return adv



class ActorNetwork(nn.Module):
    def __init__(self,extractor: nn.Module, n_actions, input_dims, alpha,
            fc1_dims=256, fc2_dims=256, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()
        self.extractor = extractor
        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo.pth')
        self.actor = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, n_actions),
                nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.extractor(state)
        state = torch.flatten(state,1)
        dist = self.actor(state)
        dist = Categorical(dist)
        
        return dist

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

class CriticNetwork(nn.Module):
    def __init__(self, extractor: nn.Module, input_dims, alpha, fc1_dims=256, fc2_dims=256,
            chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()
        self.extractor = extractor
        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo.pth')
        self.critic = nn.Sequential(
                nn.Linear(input_dims, fc1_dims),
                nn.ReLU(),
                nn.Linear(fc1_dims, fc2_dims),
                nn.ReLU(),
                nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        state = self.extractor(state)
        state = torch.flatten(state,1)
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file))
