"""Command-line entry point for TraceMsg."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracemsg.config import PROJECTS, TraceMsgConfig
from tracemsg.pipeline import TraceMsgPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TraceMsg on curated RDS datasets.")
    parser.add_argument("--project", choices=PROJECTS, default="isis")
    parser.add_argument("--mode", choices=("all", "what", "why", "full"), default="all")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("retrieve", "rerank", "feedback", "final"),
        default=("retrieve", "rerank", "feedback", "final"),
    )
    parser.add_argument("--limit_issues", type=int, default=None)
    parser.add_argument("--data_root", type=Path, default=None)
    parser.add_argument("--evaluation_root", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TraceMsgConfig()
    if args.data_root:
        config.data_root = args.data_root
    if args.evaluation_root:
        config.evaluation_root = args.evaluation_root
    if args.output_root:
        config.output_root = args.output_root
    if args.max_workers:
        config.max_workers = args.max_workers

    pipeline = TraceMsgPipeline(config)
    summary = pipeline.run_project(
        project=args.project,
        stages=tuple(args.stages),
        mode=args.mode,
        limit_issues=args.limit_issues,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
