"""Offline construction of TraceMsg software maintenance knowledge bases."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from .config import TraceMsgConfig
from .data import CommitDoc, load_commit_docs, load_gold_links
from .llm import LLMClient


HISTORICAL_COLUMNS = [
    "issue_id",
    "summary",
    "description",
    "commitid",
    "label",
]


class KnowledgeBaseBuilder:
    """Build pattern-centric maintenance knowledge from historical positive links.

    For a project such as Ambari, this reads historical pairs from
    data/ground_truth_dataset/zhang_dataset/ambari_link.csv, keeps only rows
    with label=1, excludes current evaluation issues, and abstracts the
    generated structured WHAT/WHY message rather than a human-written message.
    """

    def __init__(self, config: TraceMsgConfig, llm: Optional[LLMClient] = None):
        self.config = config
        self.llm = llm or LLMClient(config)

    def build_project(
        self,
        project: str,
        historical_csv: Optional[Path] = None,
        gold_links_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
        chunksize: int = 2000,
        limit_pairs: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        historical_csv = historical_csv or self.config.historical_link_csv_path(project)
        gold_links_path = gold_links_path or self.config.gold_links_path(project)
        output_path = output_path or self.config.knowledge_base_path(project)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        processed_path = self._processed_pairs_path(project)
        processed_path.parent.mkdir(parents=True, exist_ok=True)

        knowledge_base = self._load_json_list(output_path)
        processed_pairs = set(self._load_json_list(processed_path))
        processed_pairs.update(
            f"{entry['issue_id']}_{entry['commit_id']}"
            for entry in knowledge_base
            if "issue_id" in entry and "commit_id" in entry
        )
        excluded_issue_ids = set(load_gold_links(gold_links_path).keys())
        commit_docs = load_commit_docs(self.config.commits_path(project), self.config)
        next_id = len(knowledge_base)
        added = 0

        reader = pd.read_csv(
            historical_csv,
            usecols=lambda column: column in HISTORICAL_COLUMNS,
            chunksize=chunksize,
        )
        for chunk in tqdm(reader, desc=f"{project}: build knowledge base"):
            positives = chunk[chunk["label"].astype(str) == "1"]
            for _, row in positives.iterrows():
                issue_id = str(row.get("issue_id", "")).strip()
                commit_id = str(row.get("commitid", "")).strip()
                if not issue_id or not commit_id:
                    continue
                if issue_id in excluded_issue_ids:
                    continue

                pair_key = f"{issue_id}_{commit_id}"
                if pair_key in processed_pairs:
                    continue

                commit_doc = commit_docs.get(commit_id)
                if commit_doc is None:
                    continue
                abstractions = self._extract_pattern(row, commit_doc)
                if not abstractions:
                    processed_pairs.add(pair_key)
                    self._write_json_list(processed_path, sorted(processed_pairs))
                    continue

                knowledge_base.append(
                    {
                        "id": next_id,
                        "issue_id": issue_id,
                        "commit_id": commit_id,
                        **abstractions,
                    }
                )
                processed_pairs.add(pair_key)
                next_id += 1
                added += 1

                self._write_json_list(output_path, knowledge_base)
                self._write_json_list(processed_path, sorted(processed_pairs))

                if limit_pairs is not None and added >= limit_pairs:
                    return knowledge_base

        return knowledge_base

    def _extract_pattern(
        self, row: pd.Series, commit_doc: CommitDoc
    ) -> Optional[dict[str, dict[str, Any]]]:
        issue_text = self._issue_text(row)
        commit_text = f"WHAT: {commit_doc.what}\nWHY: {commit_doc.why}"

        issue_abs = self._abstract_issue(issue_text)
        if not issue_abs:
            return None
        commit_abs = self._abstract_commit(commit_text)
        if not commit_abs:
            return None
        pattern = self._induce_pattern(issue_abs, commit_abs)
        if not pattern:
            return None
        return {
            "issue_abstraction": issue_abs,
            "commit_abstraction": commit_abs,
            "pattern": pattern,
        }

    def _abstract_issue(self, issue_text: str) -> Optional[dict[str, Any]]:
        prompt = f"""Given an issue report, abstract it into generalized failure characteristics.

Issue:
{issue_text}

Output ONLY a JSON object with:
- symptom_type
- failure_behavior
- affected_entity
- execution_context
- semantic_clues (array of strings)
- possible_failure_mechanisms

Requirements:
1. Use generalized terminology.
2. Avoid project-specific details where possible.
3. Focus on transferable debugging patterns.
"""
        return self.llm.complete_json(
            prompt, system_prompt="You are an expert software maintenance analyst."
        )

    def _abstract_commit(self, commit_text: str) -> Optional[dict[str, Any]]:
        prompt = f"""Given a fixing commit, abstract the reusable repair knowledge.

Commit:
{commit_text}

Output ONLY a JSON object with:
- fix_operation
- code_target
- root_cause_type
- fix_strategy
- structural_change
- semantic_intent

Requirements:
1. Generalize implementation details.
2. Focus on reusable repair strategies.
3. Identify the actual repair mechanism.
"""
        return self.llm.complete_json(
            prompt, system_prompt="You are an expert in software evolution analysis."
        )

    def _induce_pattern(
        self, issue_abs: dict[str, Any], commit_abs: dict[str, Any]
    ) -> Optional[dict[str, Any]]:
        prompt = f"""Build a software maintenance pattern from an issue abstraction and a commit abstraction.

Issue Abstraction:
{json.dumps(issue_abs, indent=2)}

Commit Abstraction:
{json.dumps(commit_abs, indent=2)}

Generate ONLY a JSON object with:
- trigger_pattern (object with symptom_type, context, semantic_clues)
- cause_pattern (object with root_cause_type, failure_mechanism)
- fix_pattern (object with fix_strategy, fix_operation, structural_change)
- alignment_rationale (string explaining why the repair resolves the failure)
- transferable_rule (string: "When [symptom], developers often apply [fix strategy] to address [cause].")
- reliability (integer, default 1 for historical label=1 links)

Requirements:
1. Abstract away repository-specific details.
2. Produce transferable debugging knowledge.
3. Avoid copying original issue or commit text.
"""
        return self.llm.complete_json(
            prompt, system_prompt="You are building a software maintenance knowledge base."
        )

    @staticmethod
    def _issue_text(row: pd.Series) -> str:
        summary = row.get("summary", "")
        description = row.get("description", "")
        return f"{_safe_text(summary)}\n{_safe_text(description)}".strip()

    def _processed_pairs_path(self, project: str) -> Path:
        return self.config.cache_dir(project) / "knowledge_base_processed_pairs.json"

    @staticmethod
    def _load_json_list(path: Path) -> list:
        if not path.exists():
            return []
        with path.open("r") as file:
            return json.load(file)

    @staticmethod
    def _write_json_list(path: Path, data: list) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(data, file, indent=2)


def _safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value)
