import torch
import torch.nn as nn
import torch.nn.functional as F

# Package-agnostic import of station-conditioned TimesNet blocks
try:
    from .TimesNet_StationCond import DataEmbedding, TimesBlock
except Exception:
    from TimesNet_StationCond import DataEmbedding, TimesBlock


class _BlockConfigCond:
    def __init__(self, seq_len: int, d_model: int, d_ff: int, num_kernels: int, top_k: int, num_stations: int):
        self.seq_len = seq_len
        self.pred_len = 0
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.top_k = top_k
        self.num_stations = num_stations


class TimesEncoderCond(nn.Module):
    def __init__(self, c_in: int, d_model: int, e_layers: int, seq_len: int,
                 d_ff: int, num_kernels: int, top_k: int, dropout: float,
                 num_stations: int):
        super().__init__()
        self.seq_len = seq_len
        self.embedding = DataEmbedding(c_in, d_model, dropout=dropout, seq_len=seq_len)
        self.blocks = nn.ModuleList([
            TimesBlock(_BlockConfigCond(seq_len, d_model, d_ff or d_model, num_kernels, top_k, num_stations))
            for _ in range(e_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        enc = self.embedding(x, None)
        for blk in self.blocks:
            enc = self.layer_norm(blk(enc, station_ids))
        # Global average pool over time
        h = enc.mean(dim=1)
        return h


class TimesDecoderCond(nn.Module):
    def __init__(self, latent_dim: int, d_model: int, d_layers: int, seq_len: int, c_out: int,
                 d_ff: int, num_kernels: int, top_k: int, dropout: float,
                 num_stations: int):
        super().__init__()
        self.seq_len = seq_len
        self.project = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList([
            TimesBlock(_BlockConfigCond(seq_len, d_model, d_ff or d_model, num_kernels, top_k, num_stations))
            for _ in range(d_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, z: torch.Tensor, station_ids: torch.Tensor) -> torch.Tensor:
        dec = self.project(z).unsqueeze(1).repeat(1, self.seq_len, 1)
        for blk in self.blocks:
            dec = self.layer_norm(blk(dec, station_ids))
        y = self.projection(dec)
        return y


class TimesVAECond(nn.Module):
    def __init__(self, c_in: int, c_out: int, seq_len: int,
                 d_model: int = 64, e_layers: int = 2, d_layers: int = 2,
                 latent_dim: int = 64, d_ff: int = None, num_kernels: int = 6,
                 top_k: int = 2, dropout: float = 0.1, num_stations: int = 0,
                 station_emb: nn.Embedding = None):
        super().__init__()
        self.seq_len = seq_len
        self.station_emb = station_emb
        self.encoder = TimesEncoderCond(c_in, d_model, e_layers, seq_len, d_ff or d_model,
                                        num_kernels, top_k, dropout, num_stations)
        self.mu_head = nn.Linear(d_model, latent_dim)
        self.logvar_head = nn.Linear(d_model, latent_dim)
        # concat optional station embedding to latent
        dec_in = latent_dim + (self.station_emb.embedding_dim if self.station_emb is not None else 0)
        self.decoder = TimesDecoderCond(dec_in, d_model, d_layers, seq_len, c_out,
                                        d_ff or d_model, num_kernels, top_k, dropout, num_stations)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, station_ids: torch.Tensor):
        # simple normalization
        means = x.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - means) / stdev

        h = self.encoder(x_norm, station_ids)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        z = self.reparameterize(mu, logvar)
        if self.station_emb is not None:
            s = self.station_emb(station_ids)
            z = torch.cat([z, s], dim=-1)
        x_hat = self.decoder(z, station_ids)
        x_hat = x_hat * stdev + means
        return x_hat, mu, logvar


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor, free_bits: float = 1e-3) -> torch.Tensor:
    kl = 0.5 * (torch.exp(logvar) + mu.pow(2) - 1.0 - logvar)
    kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=1).mean()


def reconstruction_loss(x_hat: torch.Tensor, x: torch.Tensor, loss_type: str = 'l1') -> torch.Tensor:
    return F.l1_loss(x_hat, x) if loss_type == 'l1' else F.mse_loss(x_hat, x)


