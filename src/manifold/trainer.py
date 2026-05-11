
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
import torch.optim as optim

from .dataset import ManifoldDataset
from .losses import (
    LossBreakdown,
    spine_loss,
    total_loss,
    vanguard_loss,
    inspire_loss,
)
from .model import (
    LinearProjection,
    ManifoldSpine,
    NaiveModel,
    SpatiotemporalMapper,
)


@dataclass
class TrainingConfig:

    # Architecture
    out_dim: int = 64
    mapper_hidden: List[int] = field(default_factory=lambda: [128, 128])
    spine_hidden: List[int] = field(default_factory=lambda: [128, 128])
    spine_e_dim: int = 32
    layer_norm: bool = True

    # Optimiser
    lr: float = 1e-3
    weight_decay: float = 1e-5

    # Training
    n_epochs: int = 300
    log_every: int = 10

    # Loss weights  (α, β, ν)
    alpha: float = 1.0   # spine
    beta: float = 0.5    # inspire
    nu: float = 0.2      # vanguard

    # Inspiration loss hyper-params
    inspire_margin: float = 0.5
    inspire_eta: float = 0.1

    # Vanguard loss hyper-params
    delta_vg: float = 0.5
    vanguard_eps: float = 1e-4


@dataclass
class TrainingHistory:
    epochs: List[int] = field(default_factory=list)
    breakdowns: List[LossBreakdown] = field(default_factory=list)


class ManifoldTrainer:

    def __init__(
        self,
        dataset: ManifoldDataset,
        config: Optional[TrainingConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.dataset = dataset
        self.config = config or TrainingConfig()
        self.device = torch.device(device)
        self.history = TrainingHistory()

        cfg = self.config
        d = dataset.embed_dim
        d_out = cfg.out_dim
        K = dataset.n_clusters

        self.mapper = SpatiotemporalMapper(
            in_dim=d,
            out_dim=d_out,
            hidden_dims=cfg.mapper_hidden,
            layer_norm=cfg.layer_norm,
        ).to(self.device)

        self.spine = ManifoldSpine(
            n_clusters=K,
            out_dim=d_out,
            e_dim=cfg.spine_e_dim,
            hidden_dims=cfg.spine_hidden,
            layer_norm=cfg.layer_norm,
        ).to(self.device)

        self.proj = LinearProjection(in_dim=d, out_dim=d_out).to(self.device)

        self.optimizer = optim.Adam(
            list(self.mapper.parameters())
            + list(self.spine.parameters())
            + list(self.proj.parameters()),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.x = dataset.embeddings.to(self.device)
        self.t = dataset.times.to(self.device)
        self.c = dataset.cluster_ids.to(self.device)
        self.edges = dataset.edge_pairs.to(self.device)
        self.weights = dataset.impact_weights.to(self.device)


    def train(self) -> torch.Tensor:
        cfg = self.config
        self.mapper.train()
        self.spine.train()
        self.proj.train()

        t0 = time.time()
        for epoch in range(1, cfg.n_epochs + 1):
            self.optimizer.zero_grad()

            z = self.mapper(self.x, self.t)           # (N, d')

            mu_ci_ti = self.spine(self.c, self.t)     # (N, d')

            l_sp = spine_loss(z, mu_ci_ti)

            l_ins = inspire_loss(
                z=z,
                spine=self.spine,
                c=self.c,
                t=self.t,
                edge_pairs=self.edges,
                margin=cfg.inspire_margin,
                eta=cfg.inspire_eta,
            )

            l_vgd = vanguard_loss(
                z=z,
                mu_at_t=mu_ci_ti,
                spine=self.spine,
                c=self.c,
                t=self.t,
                weights=self.weights,
                delta_vg=cfg.delta_vg,
                eps=cfg.vanguard_eps,
            )

            loss, breakdown = total_loss(
                l_spine=l_sp,
                l_inspire=l_ins,
                l_vanguard=l_vgd,
                alpha=cfg.alpha,
                beta=cfg.beta,
                nu=cfg.nu,
            )

            loss.backward()
            self.optimizer.step()

            self.history.epochs.append(epoch)
            self.history.breakdowns.append(breakdown)

            if epoch % cfg.log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(
                    f"[epoch {epoch:>4d}/{cfg.n_epochs}]  {breakdown}"
                    f"  ({elapsed:.1f}s elapsed)",
                    flush=True,
                )

        self.mapper.eval()
        self.spine.eval()
        with torch.no_grad():
            z_all = self.mapper(self.x, self.t).cpu()

        return z_all


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "mapper": self.mapper.state_dict(),
                "spine": self.spine.state_dict(),
                "proj": self.proj.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
                "n_clusters": self.dataset.n_clusters,
                "embed_dim": self.dataset.embed_dim,
            },
            str(path),
        )
        print(f"[trainer] checkpoint saved → {path}", flush=True)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.mapper.load_state_dict(ckpt["mapper"])
        self.spine.load_state_dict(ckpt["spine"])
        self.proj.load_state_dict(ckpt["proj"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        print(f"[trainer] checkpoint loaded ← {path}", flush=True)


    def embed_all(self) -> torch.Tensor:
        self.mapper.eval()
        with torch.no_grad():
            z = self.mapper(self.x, self.t).cpu()
        return z



@dataclass
class NaiveTrainingConfig:

    hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    layer_norm: bool = True

    lr: float = 1e-3
    weight_decay: float = 1e-5

    n_epochs: int = 300
    log_every: int = 10


@dataclass
class NaiveTrainingHistory:
    epochs: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)


class NaiveTrainer:
    def __init__(
        self,
        dataset: ManifoldDataset,
        config: Optional[NaiveTrainingConfig] = None,
        device: str = "cpu",
    ) -> None:
        self.dataset = dataset
        self.config = config or NaiveTrainingConfig()
        self.device = torch.device(device)
        self.history = NaiveTrainingHistory()

        cfg = self.config
        d = dataset.embed_dim

        self.model = NaiveModel(
            in_dim=d,
            hidden_dims=cfg.hidden_dims,
            layer_norm=cfg.layer_norm,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

        self.x = dataset.embeddings.to(self.device)
        self.weights = dataset.impact_weights.to(self.device)

    def train(self) -> torch.Tensor:
        cfg = self.config
        self.model.train()

        t0 = time.time()
        for epoch in range(1, cfg.n_epochs + 1):
            self.optimizer.zero_grad()

            pred = self.model(self.x)
            weights_norm = (self.weights - self.weights.mean()) / (self.weights.std() + 1e-8)
            loss = torch.nn.functional.mse_loss(pred, weights_norm)

            loss.backward()
            self.optimizer.step()

            self.history.epochs.append(epoch)
            self.history.losses.append(loss.item())

            if epoch % cfg.log_every == 0 or epoch == 1:
                elapsed = time.time() - t0
                print(
                    f"[naive epoch {epoch:>4d}/{cfg.n_epochs}]  loss={loss.item():.4f}"
                    f"  ({elapsed:.1f}s elapsed)",
                    flush=True,
                )

        self.model.eval()
        with torch.no_grad():
            pred_all = self.model(self.x).cpu()

        return pred_all

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self.config,
                "history": self.history,
                "embed_dim": self.dataset.embed_dim,
            },
            str(path),
        )
        print(f"[naive trainer] checkpoint saved → {path}", flush=True)

    def load(self, path: str | Path) -> None:
        path = Path(path)
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if "history" in ckpt:
            self.history = ckpt["history"]
        print(f"[naive trainer] checkpoint loaded ← {path}", flush=True)

    def predict_all(self) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            w = self.model(self.x).cpu()
        return w
