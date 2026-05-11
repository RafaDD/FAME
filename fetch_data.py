from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from config import cfg, apply_args_to_config
from arxiv_agent.run import run_arxiv_agent
from impact_agent.run import run_impact_agent


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="fetch-data",
        description="Fetch arXiv papers for a topic and save to data/.",
    )
    p = cfg.pipeline

    parser.add_argument("--topic", required=True, help="Research topic to search on arXiv.")
    parser.add_argument(
        "--max-results",
        type=int,
        default=p.max_results,
        help=f"Max papers to retrieve (default: {p.max_results}).",
    )
    parser.add_argument(
        "--since-year",
        type=int,
        default=p.since_year,
        help="Filter papers by submission year (default: all).",
    )
    parser.add_argument(
        "--deterministic-arxiv",
        action="store_true",
        help="Disable random arXiv batch sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=p.seed,
        help=f"Random seed for reproducibility (default: {p.seed}).",
    )
    parser.add_argument(
        "--cluster-count",
        type=int,
        default=p.cluster_count,
        help=f"Topic cluster count (default: {p.cluster_count}).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Optional embedding model override.",
    )

    args = parser.parse_args(argv)
    if args.max_results < 1:
        parser.error("--max-results must be >= 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = apply_args_to_config(cfg, args)

    try:
        result = run_arxiv_agent(config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"[arxiv_agent] Saved {len(result.ranked_papers)} papers to {result.topic_dir}")
    
    result = run_impact_agent(config)

    print(f"[impact_agent] Computed impact metrics for {len(result.impacts)} papers in {result.topic_dir}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
