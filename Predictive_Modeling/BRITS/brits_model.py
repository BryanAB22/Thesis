import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mask_from_nan(x: torch.Tensor) -> torch.Tensor:
    return (~torch.isnan(x)).float()


def fill_nan_with_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0)


def reverse_time(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[1])


def build_deltas(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [B, T, F], 1=observed, 0=missing
    delta[:, t, f] = number of time steps since feature f was last observed
    """
    B, T, F = mask.shape
    delta = torch.zeros_like(mask)

    for t in range(1, T):
        delta[:, t, :] = 1.0 + (1.0 - mask[:, t - 1, :]) * delta[:, t - 1, :]

    return delta


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (torch.abs(pred - target) * mask).sum() / denom


def masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum().clamp_min(1.0)
    return (((pred - target) ** 2) * mask).sum() / denom


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(masked_mse(pred, target, mask) + 1e-12)


class FeatureRegression(nn.Module):
    """
    Predict each feature from the others at the same time step.
    Diagonal weights are masked to zero.
    """
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        self.W = nn.Parameter(torch.empty(n_features, n_features))
        self.b = nn.Parameter(torch.zeros(n_features))
        self.register_buffer("diag_mask", 1.0 - torch.eye(n_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.W * self.diag_mask
        return F.linear(x, W, self.b)


class TemporalDecay(nn.Module):
    """
    gamma = exp(-relu(W delta + b))
    """
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, delta: torch.Tensor) -> torch.Tensor:
        return torch.exp(-F.relu(self.linear(delta)))


class RITSBlock(nn.Module):
    def __init__(self, n_features: int, hidden_size: int):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size

        self.temp_decay_h = TemporalDecay(n_features, hidden_size)
        self.temp_decay_x = TemporalDecay(n_features, n_features)

        self.hist_reg = nn.Linear(hidden_size, n_features)
        self.feat_reg = FeatureRegression(n_features)

        self.combine = nn.Linear(2 * n_features, n_features)
        self.rnn_cell = nn.GRUCell(input_size=2 * n_features, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, m: torch.Tensor, d: torch.Tensor):
        B, T, F = x.shape
        device = x.device

        h = torch.zeros(B, self.hidden_size, device=device)

        imputations = []
        total_loss = x.new_tensor(0.0)

        for t in range(T):
            x_t = x[:, t, :]
            m_t = m[:, t, :]
            d_t = d[:, t, :]

            gamma_h = self.temp_decay_h(d_t)
            gamma_x = self.temp_decay_x(d_t)

            h = h * gamma_h

            # history-based estimate
            x_hist = self.hist_reg(h)
            total_loss = total_loss + masked_mae(x_hist, x_t, m_t)

            # fill missing using history
            x_c = m_t * x_t + (1.0 - m_t) * x_hist

            # feature-based estimate
            z_feat = self.feat_reg(x_c)
            total_loss = total_loss + masked_mae(z_feat, x_t, m_t)

            # combine both
            alpha = torch.sigmoid(self.combine(torch.cat([gamma_x, m_t], dim=-1)))
            c_hat = alpha * z_feat + (1.0 - alpha) * x_hist
            total_loss = total_loss + masked_mae(c_hat, x_t, m_t)

            x_imp = m_t * x_t + (1.0 - m_t) * c_hat
            imputations.append(x_imp.unsqueeze(1))

            rnn_input = torch.cat([x_imp, m_t], dim=-1)
            h = self.rnn_cell(rnn_input, h)

        imputations = torch.cat(imputations, dim=1)
        total_loss = total_loss / T

        return {
            "imputed": imputations,
            "loss": total_loss,
        }


class BRITSImputer(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 64, consistency_weight: float = 0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.consistency_weight = consistency_weight

        self.rits_f = RITSBlock(n_features=n_features, hidden_size=hidden_size)
        self.rits_b = RITSBlock(n_features=n_features, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor):
        """
        x: [B, T, F] with NaNs marking missing entries
        """
        m = build_mask_from_nan(x)
        x_filled = fill_nan_with_zero(x)
        d = build_deltas(m)

        out_f = self.rits_f(x_filled, m, d)

        x_rev = reverse_time(x_filled)
        m_rev = reverse_time(m)
        d_rev = reverse_time(d)

        out_b_rev = self.rits_b(x_rev, m_rev, d_rev)
        imputed_b = reverse_time(out_b_rev["imputed"])
        imputed_f = out_f["imputed"]

        loss_consistency = torch.mean(torch.abs(imputed_f - imputed_b))
        imputed_avg = 0.5 * (imputed_f + imputed_b)
        imputed_final = m * x_filled + (1.0 - m) * imputed_avg

        total_loss = out_f["loss"] + out_b_rev["loss"] + self.consistency_weight * loss_consistency

        return {
            "imputed": imputed_final,
            "loss": total_loss,
            "loss_forward": out_f["loss"],
            "loss_backward": out_b_rev["loss"],
            "loss_consistency": loss_consistency,
            "mask": m,
            "delta": d,
        }


def make_artificial_mask(x: torch.Tensor, holdout_frac: float = 0.10):
    """
    Hide a fraction of observed values for self-supervised training.
    """
    obs_mask = build_mask_from_nan(x)
    rand = torch.rand_like(obs_mask)
    target_mask = ((rand < holdout_frac).float() * obs_mask).float()

    x_tilde = x.clone()
    x_tilde[target_mask.bool()] = float("nan")

    return x_tilde, obs_mask, target_mask


def training_step_loss(model: nn.Module, x: torch.Tensor, holdout_frac: float = 0.10):
    """
    Train on artificially hidden observed entries only.
    """
    x_tilde, obs_mask, target_mask = make_artificial_mask(x, holdout_frac=holdout_frac)
    out = model(x_tilde)

    x_true = fill_nan_with_zero(x)
    recon_loss = masked_mae(out["imputed"], x_true, target_mask)

    total_loss = recon_loss + out["loss"]

    metrics = {
        "total_loss": total_loss.detach(),
        "recon_mae": recon_loss.detach(),
        "internal_loss": out["loss"].detach(),
        "consistency_loss": out["loss_consistency"].detach(),
    }
    return total_loss, out, target_mask, metrics


@torch.no_grad()
def evaluate_on_holdout(model: nn.Module, x: torch.Tensor, holdout_frac: float = 0.10):
    x_tilde, obs_mask, target_mask = make_artificial_mask(x, holdout_frac=holdout_frac)
    out = model(x_tilde)

    x_true = fill_nan_with_zero(x)
    mae = masked_mae(out["imputed"], x_true, target_mask)
    rmse = masked_rmse(out["imputed"], x_true, target_mask)

    return {
        "mae": mae,
        "rmse": rmse,
        "target_mask": target_mask,
        "out": out,
    }