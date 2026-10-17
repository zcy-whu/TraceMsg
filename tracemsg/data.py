"""Dataset and commit-message loading utilities."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import TraceMsgConfig


@dataclass
class CommitDoc:
    commit_id: str
    what: str
    why: str
    raw: dict[str, Any]

    @property
    def full_text(self) -> str:
        return f"{self.what}\n{self.why}".strip()

    def text_for_mode(self, mode: str) -> str:
        if mode == "what":
            return self.what
        if mode == "why":
            return self.why
        if mode in ("full", "all"):
            return self.full_text
        raise ValueError("mode must be one of: what, why, full, all")


def load_pickle(path: Path) -> Any:
    with path.open("rb") as file:
        return pickle.load(file)


def load_gold_links(path: Path) -> dict[str, list[str]]:
    raw_links = load_pickle(path)
    return {str(issue_id): [str(cid) for cid in commits] for issue_id, commits in raw_links.items()}


def load_evaluation_csv(path: Path) -> pd.DataFrame:
    """Load only the RDS fields used by TraceMsg, compacting repeated text.

    The source CSV repeats a potentially long issue description for every
    candidate commit. Reading every source column eagerly can require many
    times the on-disk size in memory, so build the four-column view in chunks
    and store repeated issue fields categorically.
    """
    columns = ["issue_id", "commit_id", "summary", "description"]
    pair_chunks: list[pd.DataFrame] = []
    issue_texts: dict[str, tuple[str, str]] = {}
    for chunk in pd.read_csv(path, usecols=columns, chunksize=100_000):
        chunk["issue_id"] = chunk["issue_id"].astype(str)
        chunk["commit_id"] = chunk["commit_id"].astype(str)
        first_rows = chunk.drop_duplicates("issue_id", keep="first")
        for row in first_rows.itertuples(index=False):
            issue_texts.setdefault(
                row.issue_id,
                (
                    "" if pd.isna(row.summary) else str(row.summary),
                    "" if pd.isna(row.description) else str(row.description),
                ),
            )
        pair_chunks.append(chunk[["issue_id", "commit_id"]])

    if not pair_chunks:
        return pd.DataFrame(columns=columns)
    df = pd.concat(pair_chunks, ignore_index=True)
    summaries = df["issue_id"].map(lambda value: issue_texts[value][0])
    descriptions = df["issue_id"].map(lambda value: issue_texts[value][1])
    df["issue_id"] = df["issue_id"].astype("category")
    df["commit_id"] = df["commit_id"].astype("category")
    df["summary"] = summaries.astype("category")
    df["description"] = descriptions.astype("category")
    return df


def load_commit_docs(path: Path, config: TraceMsgConfig) -> dict[str, CommitDoc]:
    commits_data = load_pickle(path)
    commit_docs: dict[str, CommitDoc] = {}
    for cid, info in commits_data.items():
        if config.require_valid_diff and not info.get("has_valid_diff", False):
            continue
        if config.exclude_multi_issue_commits and info.get("resolves_multiple_issues", True):
            continue

        what = info.get("change_summary_without_human_message", "")
        why = info.get("root_cause_without_human_message_qwen")
        if not why:
            why = info.get("root_cause_without_human_message", "")

        commit_docs[str(cid)] = CommitDoc(
            commit_id=str(cid),
            what=str(what or ""),
            why=str(why or ""),
            raw=info,
        )
    return commit_docs


def issue_text_from_group(issue_id: str, group: pd.DataFrame) -> tuple[str, str, str]:
    row = group.iloc[0]
    summary = row.get("summary", issue_id)
    description = row.get("description", "")
    title = str(summary) if pd.notna(summary) else issue_id
    desc = str(description) if pd.notna(description) else ""
    return title, desc, f"{title}\n{desc}".strip()
