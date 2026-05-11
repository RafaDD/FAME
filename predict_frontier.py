from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from arxiv_agent.cache_store import load_topic_data, topic_data_dir
from manifold.dataset import _compute_impact_scores
from utils.stats import corr_metrics
from utils.frontier_channels import run_manifold_pred, run_naive_pred, run_llm_pred
from dotenv import load_dotenv



def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="predict_frontier",
        description=(
            "Train manifold on papers before a cutoff date, "
            "then predict frontier scores for papers in the following month."
        ),
    )
    p.add_argument(
        "--topic",
        required=True,
        default=None,
        help="Topic. Data read from data/<topic>, results written to frontier_result/<topic>_<YYYY_MM>.",
    )
    p.add_argument(
        "--cutoff",
        required=True,
        metavar="YYYY.MM",
        help="Cutoff date in YYYY.MM format. Train set: published before this month. "
             "Eval set: published in [cutoff, cutoff + 1 month).",
    )
    p.add_argument(
        "--llm-cutoff",
        default=None,
        metavar="YYYY.MM",
        help="LLM knowledge cutoff date in YYYY.MM format.",
    )
    p.add_argument(
        "--data-dir",
        default=None,
        help="Override data directory (default: data/<topic>).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: frontier_result/<topic>_<YYYY_MM>).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for training/inference (default: cuda).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs for manifold (default: 400).",
    )
    p.add_argument(
        "--cluster-count",
        type=int,
        default=12,
        metavar="K",
        help="Number of KMeans clusters (default: heuristic sqrt(N)).",
    )
    p.add_argument(
        "--out-dim",
        type=int,
        default=64,
        help="Manifold embedding dimension d' (default: 64).",
    )
    p.add_argument(
        "--spine-e-dim",
        type=int,
        default=32,
        help="Per-topic latent embedding size in spine (default: 32).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Adam learning rate (default: 1e-4).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Adam weight decay (default: 1e-5).",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="L_spine loss weight (default: 1.0).",
    )
    p.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="L_inspire loss weight (default: 0.1).",
    )
    p.add_argument(
        "--nu",
        type=float,
        default=0.3,
        help="L_vanguard loss weight (default: 0.1).",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=20,
        help="Log training loss every N epochs (default: 20).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for KMeans (default: 42).",
    )
    p.add_argument(
        "--method",
        nargs="+",
        default=["manifold", "llm"],
        metavar="M",
        help="Scoring channel(s) to run: manifold, naive, llm (default: manifold llm).",
    )
    p.add_argument(
        "--eval-model",
        default=None,
        metavar="MODEL",
        help="Override OPENAI_EVAL_MODEL for LLM-as-judge scoring (e.g. gpt-4o, gpt-4.1).",
    )
    p.add_argument(
        "--eval-months",
        type=int,
        default=5,
        help="Number of months to evaluate (default: 5).",
    )
    return p.parse_args(argv)


def _parse_cutoff(cutoff_str: str, eval_months: int = 5) -> Tuple[datetime, datetime]:
    """Parse YYYY.MM into (cutoff_start, cutoff_end) as UTC-aware datetimes.

    cutoff_start = first day of the given month
    cutoff_end   = first day of (cutoff month + eval_months) (exclusive upper bound)
    """
    try:
        parts = cutoff_str.strip().split(".")
        if len(parts) != 2:
            raise ValueError
        year = int(parts[0])
        month = int(parts[1])
        if not (1 <= month <= 12):
            raise ValueError
    except (ValueError, IndexError):
        print(f"error: --cutoff must be in YYYY.MM format, got {cutoff_str!r}", file=sys.stderr)
        sys.exit(1)

    cutoff_start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month >= 13 - eval_months:
        cutoff_end = datetime(year + 1, eval_months - (12 - month), 1, tzinfo=timezone.utc)
    else:
        cutoff_end = datetime(year, month + eval_months, 1, tzinfo=timezone.utc)

    return cutoff_start, cutoff_end

def _parse_month_start(ym_str: str) -> datetime:
    """Parse YYYY.MM to the first day of that month (UTC)."""
    parts = ym_str.strip().split(".")
    if len(parts) != 2:
        raise ValueError(f"expected YYYY.MM, got {ym_str!r}")
    year, month = int(parts[0]), int(parts[1])
    if not (1 <= month <= 12):
        raise ValueError(f"invalid month in {ym_str!r}")
    return datetime(year, month, 1, tzinfo=timezone.utc)


def _ensure_tz(dt: datetime) -> datetime:
    """Return a timezone-aware datetime; assume UTC if naive."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _set_seeds(seed: int) -> None:
    """Seed Python random, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _save_ranked_json(
    output_dir: Path,
    out_ranked: Path,
    topic_slug: str,
    args: argparse.Namespace,
    cutoff_start: datetime,
    cutoff_end: datetime,
    data_dir: Path,
    train_papers: list,
    eval_papers: list,
    correlation_updates: Dict[str, Any],
) -> None:
    """Save or update ranked.json. If file exists, merge new correlations into 'correlation'."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().isoformat()

    if out_ranked.exists():
        existing = json.loads(out_ranked.read_text(encoding="utf-8"))
        correlation = existing.get("correlation", {})
        correlation.update(correlation_updates)
        existing["correlation"] = correlation
        existing["generated_at"] = generated_at
        out_ranked.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        doc: Dict[str, Any] = {
            "topic": topic_slug,
            "cutoff": args.cutoff,
            "train_period": f"before {cutoff_start.date()}",
            "eval_period": f"{cutoff_start.date()} – {cutoff_end.date()}",
            "data_dir": str(data_dir),
            "n_train_papers": len(train_papers),
            "n_eval_papers": len(eval_papers),
            "generated_at": generated_at,
            "correlation": correlation_updates,
        }
        out_ranked.write_text(
            json.dumps(doc, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    print(f"[predict_frontier] ranked result → {out_ranked}", flush=True)


def main(argv: list[str] | None = None) -> None:
    load_dotenv()

    args = _parse_args(argv)

    if args.eval_model is not None:
        os.environ["OPENAI_EVAL_MODEL"] = args.eval_model
        from config import cfg
        cfg.llm.eval_model = args.eval_model
        print(f"[predict_frontier] eval model: {args.eval_model}", flush=True)

    if args.topic is None and args.data_dir is None:
        print("error: provide <topic> or --data-dir", file=sys.stderr)
        sys.exit(1)

    _set_seeds(args.seed)
    print(f"[predict_frontier] seed={args.seed}", flush=True)

    cutoff_start, cutoff_end = _parse_cutoff(args.cutoff, args.eval_months)
    cutoff_tag = args.cutoff.replace(".", "_")  # e.g. "2025_06"

    if args.data_dir is not None:
        data_dir = Path(args.data_dir)
        topic_slug = data_dir.name
    else:
        data_dir = topic_data_dir(args.topic)
        topic_slug = data_dir.name

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("frontier_result") / f"{topic_slug}"

    if not data_dir.exists():
        print(f"error: data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load all data ──────────────────────────────────────────────────────────
    print(f"[predict_frontier] loading data from {data_dir} …", flush=True)
    cache = load_topic_data(data_dir)
    if cache is None:
        print(f"error: could not load topic data from {data_dir}", file=sys.stderr)
        sys.exit(1)

    all_papers = cache.papers
    all_embeddings = cache.embeddings  # list[list[float]], aligned with all_papers
    all_edges = cache.inspiration_edges

    print(f"[predict_frontier] total papers loaded: {len(all_papers)}", flush=True)

    # ── Temporal split ─────────────────────────────────────────────────────────
    train_papers: list = []
    train_embeddings: list[list[float]] = []
    eval_papers: list = []
    eval_embeddings: list[list[float]] = []

    for paper, emb in zip(all_papers, all_embeddings):
        pub = _ensure_tz(paper.published)
        if pub < cutoff_start:
            train_papers.append(paper)
            train_embeddings.append(emb)
        elif cutoff_start <= pub < cutoff_end:
            eval_papers.append(paper)
            eval_embeddings.append(emb)

    print(
        f"[predict_frontier] split → train: {len(train_papers)}, eval: {len(eval_papers)}",
        flush=True,
    )

    if len(eval_papers) == 0:
        print(
            f"warning: no eval papers found in [{cutoff_start.date()}, {cutoff_end.date()}). "
            "The output files will be empty.",
            file=sys.stderr,
        )

    # ── Eval paper impact weights (needed for correlations) ─────────────────────
    eval_id_to_idx: dict[str, int] = {p.arxiv_id: i for i, p in enumerate(eval_papers)}
    eval_edge_pairs: list[tuple[int, int]] = []
    for edge in all_edges:
        if not edge.is_inspired:
            continue
        src_idx = eval_id_to_idx.get(edge.source_arxiv_id)
        tgt_idx = eval_id_to_idx.get(edge.target_arxiv_id)
        if tgt_idx is None:
            continue
        eval_edge_pairs.append((tgt_idx, src_idx))

    impact_path = data_dir / "impact.json"
    eval_impact_weights: list[float] = _compute_impact_scores(
        eval_papers,
        eval_edge_pairs,
        impact_path=impact_path,
    )
    eval_impact_dict: dict[str, float] = {
        p.arxiv_id: float(eval_impact_weights[i]) for i, p in enumerate(eval_papers)
    }

    # Historical hint-paper construction is currently disabled.
    llm_cutoff_str = args.llm_cutoff or args.cutoff
    llm_cutoff_start = _parse_month_start(llm_cutoff_str)
    hint_train_papers: list = [
        p for p in train_papers
        if llm_cutoff_start <= _ensure_tz(p.published) < cutoff_start
    ]

    hint_papers: list[tuple[str, float]] = []
    if hint_train_papers:
        hint_id_to_idx: dict[str, int] = {p.arxiv_id: i for i, p in enumerate(hint_train_papers)}
        hint_edge_pairs: list[tuple[int, int]] = []
        for edge in all_edges:
            if not edge.is_inspired:
                continue
            tgt_idx = hint_id_to_idx.get(edge.target_arxiv_id)
            if tgt_idx is None:
                continue
            hint_edge_pairs.append((tgt_idx, edge.source_arxiv_id))

        hint_impact_weights: list[float] = _compute_impact_scores(
            hint_train_papers,
            hint_edge_pairs,
            impact_path=impact_path,
        )
        hint_papers = [
            (p.title + '\n' + \
             'Abstract: ' + p.abstract + '\n' + \
             'Published: ' + p.published.strftime("%Y-%m-%d"), float(hint_impact_weights[i]))
            for i, p in enumerate(hint_train_papers)
        ]
    random.shuffle(hint_papers)
    hint_papers = hint_papers[:50]

    _aids = [p.arxiv_id for p in eval_papers]
    _iw = [eval_impact_dict[aid] for aid in _aids]

    # ── Channel: Manifold ───────────────────────────────────────────────────────
    manifold_result: Optional[Dict[str, Any]] = None
    if 'manifold' in args.method:
        manifold_result = run_manifold_pred(
            args, data_dir, train_papers, train_embeddings, eval_papers, eval_embeddings, all_edges
        )

    # ── Channel: Naive (embedding → impact weight predictor) ───────────────────
    naive_result: Optional[Dict[str, float]] = None
    if 'naive' in args.method:
        naive_result = run_naive_pred(
            args, data_dir, train_papers, train_embeddings,
            eval_papers, eval_embeddings, all_edges,
        )

    # ── Channel: LLM ───────────────────────────────────────────────────────────
    llm_potential_by_id: Dict[str, float] = {}
    llm_model_key: Optional[str] = None
    if "llm" in args.method:
        llm_potential_by_id, llm_model_key = run_llm_pred(
            eval_papers,
            topic_slug,
            hint_papers=hint_papers,
        )

    # ── Correlation analysis (from channels run) ────────────────────────────────
    correlation_updates: Dict[str, Any] = {}

    if manifold_result:
        fr_dict = manifold_result["fr_dict"]
        _fr_vals = [fr_dict[aid] for aid in _aids]
        correlation_updates["manifold"] = corr_metrics(_fr_vals, _iw)

    if naive_result:
        _naive_vals = [naive_result[aid] for aid in _aids]
        correlation_updates["naive"] = corr_metrics(_naive_vals, _iw)

    if llm_potential_by_id:
        _llm_scores = [llm_potential_by_id[aid] for aid in _aids]
        corr_llm = corr_metrics(_llm_scores, _iw) if _llm_scores else {"pearson_r": float("nan"), "spearman_r": float("nan")}
        correlation_updates[llm_model_key] = corr_llm

    # ── Save results ───────────────────────────────────────────────────────────
    out_ranked = output_dir / f"{cutoff_tag}_{args.seed}_ranked.json"
    _save_ranked_json(
        output_dir=output_dir,
        out_ranked=out_ranked,
        topic_slug=topic_slug,
        args=args,
        cutoff_start=cutoff_start,
        cutoff_end=cutoff_end,
        data_dir=data_dir,
        train_papers=train_papers,
        eval_papers=eval_papers,
        correlation_updates=correlation_updates,
    )

    print("[predict_frontier] done.", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
