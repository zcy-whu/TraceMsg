"""Ranking metrics for issue-commit traceability."""

from __future__ import annotations

import pandas as pd


def hit_at_k(results: pd.DataFrame, gold_links: dict[str, list[str]], k: int) -> float:
    grouped = {str(key): group for key, group in results.groupby("s_id")}
    hits = 0
    total = 0
    for issue_id, gold_commits in gold_links.items():
        if not gold_commits:
            continue
        total += 1
        group = grouped.get(str(issue_id))
        if group is not None and any(str(row["t_id"]) in gold_commits for _, row in group.head(k).iterrows()):
            hits += 1
    return round(hits / total if total else 0.0, 4)


def recall_at_k(results: pd.DataFrame, gold_links: dict[str, list[str]], k: int) -> float:
    grouped = {str(key): group for key, group in results.groupby("s_id")}
    total = 0
    recall_sum = 0.0
    for issue_id, gold_commits in gold_links.items():
        if not gold_commits:
            continue
        total += 1
        group = grouped.get(str(issue_id))
        if group is None:
            continue
        hits = sum(1 for _, row in group.head(k).iterrows() if str(row["t_id"]) in gold_commits)
        recall_sum += hits / len(gold_commits)
    return round(recall_sum / total if total else 0.0, 4)


def mrr(results: pd.DataFrame, gold_links: dict[str, list[str]]) -> float:
    grouped = {str(key): group for key, group in results.groupby("s_id")}
    total = 0
    score = 0.0
    for issue_id, gold_commits in gold_links.items():
        if not gold_commits:
            continue
        total += 1
        group = grouped.get(str(issue_id))
        if group is None:
            continue
        for rank, (_, row) in enumerate(group.iterrows(), start=1):
            if str(row["t_id"]) in gold_commits:
                score += 1.0 / rank
                break
    return round(score / total if total else 0.0, 4)


def evaluate(results: pd.DataFrame, gold_links: dict[str, list[str]]) -> dict[str, float]:
    return {
        "Hit@1": hit_at_k(results, gold_links, 1),
        "Hit@5": hit_at_k(results, gold_links, 5),
        "Hit@10": hit_at_k(results, gold_links, 10),
        "Recall@1": recall_at_k(results, gold_links, 1),
        "Recall@5": recall_at_k(results, gold_links, 5),
        "Recall@10": recall_at_k(results, gold_links, 10),
        "MRR": mrr(results, gold_links),
    }
