import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from .TimesNet import DataEmbedding, TimesBlock
except Exception:
    from TimesNet import DataEmbedding, TimesBlock


class _BlockConfig:
    def __init__(self, seq_len: int, pred_len: int, d_model: int, d_ff: int, num_kernels: int, top_k: int = 2):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k


class TimesNetPointCloud(nn.Module):
    """TimesNet reconstruction with exposed encode/project methods for point-cloud mixing."""
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.pred_len = getattr(configs, 'pred_len', 0)
        self.top_k = configs.top_k
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.num_kernels = configs.num_kernels
        self.e_layers = configs.e_layers
        self.dropout = configs.dropout
        self.c_out = configs.c_out

        self.enc_embedding = DataEmbedding(configs.enc_in, self.d_model, configs.embed, configs.freq,
                                           configs.dropout, configs.seq_len)
        self.model = nn.ModuleList([
            TimesBlock(_BlockConfig(self.seq_len, 0, self.d_model, self.d_ff, self.num_kernels, self.top_k))
            for _ in range(self.e_layers)
        ])
        self.layer = self.e_layers
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def encode_features_for_reconstruction(self, x_enc: torch.Tensor):
        means = x_enc.mean(1, keepdim=True).detach()
        x_norm = x_enc - means
        stdev = torch.sqrt(torch.var(x_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = x_norm / stdev
        enc_out = self.enc_embedding(x_norm, None)
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        return enc_out, means, stdev

    def project_features_for_reconstruction(self, enc_out: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor):
        dec_out = self.projection(enc_out)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc: torch.Tensor):
        enc_out, means, stdev = self.encode_features_for_reconstruction(x_enc)
        return self.project_features_for_reconstruction(enc_out, means, stdev)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """Forward pass compatible with anomaly_detection task."""
        return self.anomaly_detection(x_enc)


