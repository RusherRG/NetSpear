import argparse
import os
from distutils.util import strtobool

from gym_minigrid.wrappers import *
import torch
import gymnasium as gym
import copy
import nasim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=18,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with tensorboard")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="nasim:c-lr-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=7500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=10,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=1,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--hidden-size", type=int, default=256,
        help="hidden layer size of the neural networks")
    parser.add_argument("--coverage", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use coverage mechanism as per the paper.")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args


def make_env_list_random(env_id, seed, num_envs):
    if env_id[0:7] == "nasim:c":
        _fix_env = nasim.generate(
            num_hosts=100,
            num_os=3,
            num_services=10,
            num_exploits=30,
            num_processes=3,
            restrictiveness=5,
            step_limit=30000,
            yz_gen=False,
            save_fig=True,
        )
    else:
        _fix_env = gym.make(env_id)
    _fix_env = gym.wrappers.RecordEpisodeStatistics(_fix_env)
    _fix_env.action_space.seed(seed)
    _fix_env.observation_space.seed(seed)

    def _env_make(fix_env, idx):
        if idx == 0:
            return lambda: fix_env
        else:
            env = copy.deepcopy(fix_env)
            env.action_space.seed(seed + idx)
            env.observation_space.seed(seed + idx)
            return lambda: env

    return [_env_make(_fix_env, i) for i in range(num_envs)]


def make_env(env_id, seed, idx):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    def env_thunk():
        # print(env_id[0:7])
        if env_id[0:8] == "MiniGrid":
            print("==" * 10 + "MiniGrid" + "==" * 10)
            env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)

            # env = Monitor(env)
            env = OneHotPartialObsWrapper(env)
            # env = RGBImgPartialObsWrapper(env)
            env = FlatObsWrapper(env)
            return env

        if env_id[0:7] == "nasim:c":
            # env = nasim.generate(num_hosts=50, num_services=5, num_os=3, num_processes=2, \
            #         num_exploits=None, num_privescs=None, r_sensitive=10, r_user=10, \
            #         exploit_cost=1, exploit_probs="mixed", privesc_cost=1, privesc_probs=0.9, \
            #         service_scan_cost=1, os_scan_cost=1, subnet_scan_cost=1, process_scan_cost=1,\
            #         uniform=False, alpha_H=2.0, alpha_V=2.0, lambda_V=1.0, restrictiveness=3, \
            #         random_goal=False, base_host_value=1, host_discovery_value=1, \
            #         seed=None, name=None, step_limit=50000, address_space_bounds=None, yz_gen=True, save_fig=True)
            env = nasim.generate(
                num_hosts=95,
                num_os=3,
                num_services=10,
                num_exploits=30,
                num_processes=3,
                restrictiveness=5,
                step_limit=30000,
                yz_gen=False,
                save_fig=True,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        if env_id[0:7] == "nasim:d":
            env = nasim.generate(
                num_hosts=200,
                num_services=5,
                num_os=3,
                num_processes=2,
                num_exploits=None,
                num_privescs=None,
                r_sensitive=10,
                r_user=10,
                exploit_cost=1,
                exploit_probs="mixed",
                privesc_cost=1,
                privesc_probs=1.0,
                service_scan_cost=1,
                os_scan_cost=1,
                subnet_scan_cost=1,
                process_scan_cost=1,
                uniform=False,
                alpha_H=2.0,
                alpha_V=2.0,
                lambda_V=1.0,
                restrictiveness=3,
                random_goal=True,
                base_host_value=1,
                host_discovery_value=1,
                seed=None,
                name=None,
                step_limit=10000,
                address_space_bounds=None,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        else:
            env = gym.make(env_id)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            return env

    return env_thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def matrix_norm(v, axis=1):
    if np.all(v == 0):
        return v
    norm = np.divide(v, np.tile(np.sum(v, axis), (v.shape[axis], axis)).transpose())

    return norm


class FastGLU(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.linear = layer_init(nn.Linear(in_size, in_size * 2))

    def forward(self, x):
        x = self.linear(x)
        out = x[:, self.in_size :] * x[:, : self.in_size].sigmoid()
        return out


class Transformer(nn.Module):
    """AlphaStar transformer composed with only three encoder layers"""

    # refactored by reference to https://github.com/metataro/sc2_imitation_learning

    # default parameter from AlphaStar
    def __init__(
        self,
        d_model=256,
        d_inner=1024,
        n_layers=3,
        n_head=2,
        d_k=128,
        d_v=128,
        dropout=0.1,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        enc_output, *_ = self.encoder(x, mask=mask)

        return enc_output


class Encoder(nn.Module):
    """A alphastar encoder model with self attention mechanism."""

    # default parameter from AlphaStar
    def __init__(
        self,
        n_layers=3,
        n_head=2,
        d_k=128,
        d_v=128,
        d_model=256,
        d_inner=1024,
        dropout=0.1,
    ):
        super().__init__()

        self.layer_stack = nn.ModuleList(
            [
                EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
                for _ in range(n_layers)
            ]
        )

        # note "unbiased=False" will affect the results
        # layer_norm is b = (a - torch.mean(a))/(torch.var(a, unbiased=False)**0.5) * 1.0 + 0.0

    def forward(self, x, mask=None):
        # -- Forward

        enc_output = x
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, mask=mask)

        del enc_slf_attn

        return (enc_output,)


class EncoderLayer(nn.Module):
    """ """

    # default parameter from AlphaStar
    def __init__(
        self, d_model=256, d_inner=1024, n_head=2, d_k=128, d_v=128, dropout=0.1
    ):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

        self.ln1 = nn.LayerNorm(d_model, eps=1e-6)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask=None):
        att_out, enc_slf_attn = self.slf_attn(x, x, x, mask=mask)

        att_out = self.drop1(att_out)
        out_1 = self.ln1(x + att_out)

        ffn_out = self.pos_ffn(out_1)

        ffn_out = self.drop2(ffn_out)
        out = self.ln2(out_1 + ffn_out)

        del att_out, out_1, ffn_out

        return out, enc_slf_attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1, bias_value=-1e9):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.biasval = bias_value

    def forward(self, q, k, v, mask=None):
        # q: (b, n, lq, dk)
        # k: (b, n, lk, dk)
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        # atten: (b, n, lq, lk),
        if mask is not None:
            attn = attn.masked_fill(mask == 0, self.biasval)
            del mask

        attn = self.dropout(F.softmax(attn, dim=-1))

        # v: (b, n, lv, dv)
        # r: (b, n, lq, dv)
        r = torch.matmul(attn, v)

        return r, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        # pre-attention projection
        self.w_qs = layer_init(nn.Linear(d_model, n_head * d_k, bias=True))
        self.w_ks = layer_init(nn.Linear(d_model, n_head * d_k, bias=True))
        self.w_vs = layer_init(nn.Linear(d_model, n_head * d_v, bias=True))

        # after-attention projection
        self.fc = layer_init(nn.Linear(n_head * d_v, d_model, bias=True))

        # attention
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

    def forward(self, q, k, v, mask=None):
        # q: (b, lq, dm)
        # k: (b, lk, dm)
        # v: (b, lv, dm)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        size_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        # pass through the pre-attention projection
        # separate different heads

        # after that q: (b, lq, n, dk)
        q = self.w_qs(q).view(size_b, len_q, n_head, d_k)

        k = self.w_ks(k).view(size_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(size_b, len_v, n_head, d_v)

        # transpose for attention dot product: (b, n, lq, dk)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # q: (b, n, lq, dk), k: (b, n, lk, dk), atten = q \matmul k^t = (b, n, lq, lk),
        # v: (b, n, lv, dv), assert lk = lv
        # atten \matmul v = (b, n, lq, dv)

        # transpose to move the head dimension back: (b, lq, n, dv)
        # combine the last two dimensions to concatenate all the heads together: (b, lq, (n*dv))
        q = q.transpose(1, 2).contiguous().view(size_b, len_q, -1)

        # q: (b, lq, (n*dv)) \matmul ((n*dv), dm) = (b, lq, dm)
        # note, q has the same shape as when it enter in
        q = self.fc(q)

        del (
            mask,
            k,
            v,
        )

        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = layer_init(nn.Linear(d_in, d_hid))  # position-wise
        self.w_2 = layer_init(nn.Linear(d_hid, d_in))  # position-wise

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)

        return x


class CategoricalMasked(Categorical):
    def __init__(
        self, probs=None, logits=None, validate_args=None, masks=[], device="cpu"
    ):
        self.masks = masks
        self.device = device
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(self.device)
            logits = torch.where(
                self.masks, logits, torch.tensor(-1e17).to(self.device)
            )
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)
