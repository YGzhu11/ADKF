import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.kanlayers import TaylorKANLayer, WaveKANLayer


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == 'denorm':
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hist_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.var_num = configs.var_num
        self.num_experts = 4
        self.drop = configs.dropout
        self.revin_affine = True

        self.gate = nn.Linear(self.hist_len, self.num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.experts = nn.ModuleList([
            TaylorKANLayer(self.hist_len, self.pred_len, order=3, addbias=True),
            TaylorKANLayer(self.hist_len, self.pred_len, order=3, addbias=True),
            WaveKANLayer(self.hist_len, self.pred_len, wavelet_type="mexican_hat", device="cuda"),
            WaveKANLayer(self.hist_len, self.pred_len, wavelet_type="mexican_hat", device="cuda"),
        ])

        self.dropout = nn.Dropout(self.drop)
        self.rev = RevIN(self.var_num, affine=self.revin_affine)

    def forward(self, var_x, marker_x):
        B, L, N = var_x.shape
        var_x = self.rev(var_x, 'norm') if self.rev else var_x
        var_x = self.dropout(var_x).transpose(1, 2).reshape(B * N, L)

        score = F.softmax(self.gate(var_x), dim=-1)  # (BxN, E)

        expert_outputs = torch.stack([self.experts[i](var_x) for i in range(self.num_experts)], dim=-1)  # (BxN, Lo, E)

        prediction = torch.einsum("BLE,BE->BL", expert_outputs, score).reshape(B, N, -1).permute(0, 2, 1)
        prediction = self.rev(prediction, 'denorm')
        return prediction
