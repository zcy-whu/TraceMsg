"""Build TraceMsg knowledge_base.json from historical positive issue-commit links."""

from __future__ import annotations

import argparse
from pathlib import Path

from tracemsg.config import PROJECTS, TraceMsgConfig
from tracemsg.knowledge_builder import KnowledgeBaseBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TraceMsg maintenance patterns.")
    parser.add_argument("--project", choices=PROJECTS, default="ambari")
    parser.add_argument("--historical_csv", type=Path, default=None)
    parser.add_argument("--gold_links", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--chunksize", type=int, default=2000)
    parser.add_argument("--limit_pairs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TraceMsgConfig()
    builder = KnowledgeBaseBuilder(config)
    kb = builder.build_project(
        project=args.project,
        historical_csv=args.historical_csv,
        gold_links_path=args.gold_links,
        output_path=args.output,
        chunksize=args.chunksize,
        limit_pairs=args.limit_pairs,
    )
    print(f"Saved {len(kb)} patterns.")


if __name__ == "__main__":
    main()
