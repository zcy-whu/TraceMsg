"""End-to-end TraceMsg workflow."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from tqdm import tqdm

from .config import TraceMsgConfig
from .data import (
    CommitDoc,
    issue_text_from_group,
    load_commit_docs,
    load_evaluation_csv,
    load_gold_links,
)
from .evaluation import evaluate
from .feedback_agent import FeedbackAgent
from .generation_agent import GenerationAgent
from .llm import LLMClient
from .traceability_agent import TraceabilityAgent


class TraceMsgPipeline:
    def __init__(self, config: TraceMsgConfig):
        self.config = config
        self.llm: Optional[LLMClient] = None
        self.generation_agent: Optional[GenerationAgent] = None
        self.traceability_agent = TraceabilityAgent(config, None)
        self.feedback_agent: Optional[FeedbackAgent] = None

    def run_project(
        self,
        project: str,
        stages: tuple[str, ...] = ("retrieve", "rerank", "feedback", "final"),
        mode: Optional[str] = None,
        limit_issues: Optional[int] = None,
    ) -> dict[str, Any]:
        mode = mode or self.config.retrieval_mode
        self._ensure_dirs(project, mode)

        commit_docs = load_commit_docs(self.config.commits_path(project), self.config)
        gold_links = load_gold_links(self.config.gold_links_path(project))
        eval_df = load_evaluation_csv(self.config.evaluation_csv_path(project))
        groups = list(eval_df.groupby("issue_id", observed=True))
        if limit_issues is not None:
            groups = groups[:limit_issues]
        evaluated_issue_ids = {str(issue_id) for issue_id, _ in groups}
        evaluation_gold = {
            issue_id: commits
            for issue_id, commits in gold_links.items()
            if issue_id in evaluated_issue_ids
        }

        outputs: dict[str, Any] = {
            "project": project,
            "mode": mode,
            "valid_commits": len(commit_docs),
            "issues": len(groups),
        }

        if "retrieve" in stages:
            stage0 = self.run_retrieval(project, groups, commit_docs, gold_links, mode)
            outputs["stage0_metrics"] = evaluate(stage0, evaluation_gold)

        if "rerank" in stages:
            self._ensure_llm_agents()
            rerank = self.run_initial_rerank(project, groups, commit_docs, gold_links, mode)
            outputs["stage2_metrics"] = evaluate(rerank, evaluation_gold)

        if "feedback" in stages:
            self._ensure_llm_agents()
            self.run_feedback_refinement(project, groups, commit_docs, mode)

        if "final" in stages:
            self._ensure_llm_agents()
            final = self.run_final_rerank(project, groups, commit_docs, gold_links, mode)
            outputs["stage3_metrics"] = evaluate(final, evaluation_gold)

        summary_path = self.config.result_dir(project) / f"summary_{mode}.json"
        with summary_path.open("w") as file:
            json.dump(outputs, file, indent=2)
        return outputs

    def run_retrieval(
        self,
        project: str,
        groups: list[tuple[str, pd.DataFrame]],
        commit_docs: dict[str, CommitDoc],
        gold_links: dict[str, list[str]],
        mode: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for issue_id, group in tqdm(groups, desc=f"{project}: retrieval"):
            if issue_id not in gold_links:
                continue
            _, _, issue_text = issue_text_from_group(issue_id, group)
            candidates = [str(cid) for cid in group["commit_id"].tolist()]
            top_k = max(
                self.config.initial_top_k,
                int(len(candidates) * self.config.retrieval_ratio + 0.9999),
            )
            retrieved = self.traceability_agent.retrieve(
                issue_text, candidates, commit_docs, mode=mode, top_k=top_k
            )
            self._write_json(self._retrieval_cache(project, mode, issue_id), retrieved)
            gold = gold_links.get(issue_id, [])
            for item in retrieved:
                rows.append(
                    {
                        "s_id": issue_id,
                        "t_id": item["commit_id"],
                        "label": 1 if item["commit_id"] in gold else 0,
                        "pred": item["retrieval_score"],
                    }
                )
        df = pd.DataFrame(rows).sort_values(["s_id", "pred"], ascending=[True, False])
        out_path = self.config.result_dir(project) / f"stage0_results_{mode}.csv"
        df.to_csv(out_path, index=False)
        return df

    def run_initial_rerank(
        self,
        project: str,
        groups: list[tuple[str, pd.DataFrame]],
        commit_docs: dict[str, CommitDoc],
        gold_links: dict[str, list[str]],
        mode: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for issue_id, group in tqdm(groups, desc=f"{project}: initial rerank"):
            if issue_id not in gold_links:
                continue
            cache_path = self._rerank_cache(project, mode, issue_id)
            if cache_path.exists():
                ranked = self._read_json(cache_path)
            else:
                _, _, issue_text = issue_text_from_group(issue_id, group)
                retrieved = self._read_json(self._retrieval_cache(project, mode, issue_id))
                ranked = self.traceability_agent.rerank(issue_text, retrieved, commit_docs, mode=mode)
                self._write_json(cache_path, ranked)

            gold = gold_links.get(issue_id, [])
            for index, item in enumerate(ranked, start=1):
                rows.append(
                    {
                        "s_id": issue_id,
                        "t_id": item["commit_id"],
                        "label": 1 if item["commit_id"] in gold else 0,
                        "rerank_pred": len(ranked) - index + 1,
                        "retrieval_score": item.get("retrieval_score", 0.0),
                    }
                )
        df = pd.DataFrame(rows).sort_values(["s_id", "rerank_pred"], ascending=[True, False])
        out_path = self.config.result_dir(project) / f"rerank_results_{mode}.csv"
        df.to_csv(out_path, index=False)
        return df

    def run_feedback_refinement(
        self,
        project: str,
        groups: list[tuple[str, pd.DataFrame]],
        commit_docs: dict[str, CommitDoc],
        mode: str,
    ) -> None:
        # Table 6 specifies that WHAT-only has no rationale to refine and its
        # S3 result is exactly its S2 result.
        if mode == "what":
            return
        knowledge_base = self.feedback_agent.load_knowledge_base(
            self.config.knowledge_base_path(project)
        )
        if not knowledge_base:
            raise FileNotFoundError(
                "The feedback stage requires a software maintenance knowledge base: "
                f"{self.config.knowledge_base_path(project)}"
            )

        for issue_id, group in tqdm(groups, desc=f"{project}: feedback/refine"):
            _, _, issue_text = issue_text_from_group(issue_id, group)
            ranked = self._read_json(self._rerank_cache(project, mode, issue_id))
            top_items = ranked[: self.config.initial_top_k]
            cache_path = self._feedback_cache(project, mode, issue_id)
            if cache_path.exists():
                feedback_rows = self._read_json(cache_path)
            else:
                feedback_rows = self._judge_issue(issue_text, top_items, commit_docs, knowledge_base)
                self._write_json(cache_path, feedback_rows)

            refined_path = self._refined_cache(project, mode, issue_id)
            if refined_path.exists():
                refined_rows = self._read_json(refined_path)
            else:
                refined_rows = self._refine_issue(issue_text, feedback_rows, commit_docs)
                self._write_json(refined_path, refined_rows)

    def run_final_rerank(
        self,
        project: str,
        groups: list[tuple[str, pd.DataFrame]],
        commit_docs: dict[str, CommitDoc],
        gold_links: dict[str, list[str]],
        mode: str,
    ) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for issue_id, group in tqdm(groups, desc=f"{project}: final rerank"):
            if issue_id not in gold_links:
                continue
            _, _, issue_text = issue_text_from_group(issue_id, group)
            ranked = self._read_json(self._rerank_cache(project, mode, issue_id))
            top_items = ranked[: self.config.initial_top_k]
            final_cache = self._final_cache(project, mode, issue_id)
            if final_cache.exists():
                final_ranked = self._read_json(final_cache)
            elif mode == "what":
                final_ranked = top_items
                self._write_json(final_cache, final_ranked)
            else:
                issue_commit_docs = self._issue_conditioned_docs(
                    project, mode, issue_id, commit_docs
                )
                final_ranked = self.traceability_agent.rerank(
                    issue_text, top_items, issue_commit_docs, mode=mode
                )
                self._write_json(final_cache, final_ranked)

            gold = gold_links.get(issue_id, [])
            for index, item in enumerate(final_ranked, start=1):
                rows.append(
                    {
                        "s_id": issue_id,
                        "t_id": item["commit_id"],
                        "label": 1 if item["commit_id"] in gold else 0,
                        "final_rank": index,
                        "final_pred": len(final_ranked) - index + 1,
                        "retrieval_score": item.get("retrieval_score", 0.0),
                    }
                )
        df = pd.DataFrame(rows).sort_values(["s_id", "final_rank"], ascending=[True, True])
        out_path = self.config.result_dir(project) / f"final_top10_rerank_results_{mode}.csv"
        df.to_csv(out_path, index=False)
        return df

    def _judge_issue(
        self,
        issue_text: str,
        top_items: list[dict[str, Any]],
        commit_docs: dict[str, CommitDoc],
        knowledge_base: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        def judge(item: dict[str, Any]) -> Optional[dict[str, Any]]:
            cid = item["commit_id"]
            doc = commit_docs.get(cid)
            if not doc:
                return None
            pair_abstraction = self.feedback_agent.abstract_pair(issue_text, doc)
            patterns = self.feedback_agent.retrieve_patterns(pair_abstraction, knowledge_base)
            detail = self.feedback_agent.judge_pair(pair_abstraction, patterns)
            decision = detail.get("linkage_decision", "Unsupported")
            confidence = float(detail.get("confidence", 0.0) or 0.0)
            score = confidence + (1.0 if decision == "Supported" else 0.0)
            if decision == "Unsupported":
                score *= 0.5
            return {"commit_id": cid, "judge_detail": detail, "judge_score": score}

        return self._parallel_map(judge, top_items, desc="judge")

    def _refine_issue(
        self,
        issue_text: str,
        feedback_rows: list[dict[str, Any]],
        commit_docs: dict[str, CommitDoc],
    ) -> list[dict[str, Any]]:
        def refine(item: dict[str, Any]) -> Optional[dict[str, Any]]:
            cid = item["commit_id"]
            doc = commit_docs.get(cid)
            if not doc:
                return None
            detail = item.get("judge_detail", {})
            refined = self.generation_agent.refine_why(
                issue_text=issue_text,
                original_why=doc.why,
                linkage_decision=detail.get("linkage_decision", "Unsupported"),
                guidance=detail.get(
                    "refinement_guidance_for_why", detail.get("explanation", "")
                ),
            )
            detail["refined_why"] = refined
            item["judge_detail"] = detail
            return item

        return self._parallel_map(refine, feedback_rows, desc="refine")

    def _parallel_map(self, fn, items: list[Any], desc: str) -> list[Any]:
        results: list[Any] = []
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(fn, item) for item in items]
            for future in as_completed(futures):
                value = future.result()
                if value is not None:
                    results.append(value)
        return results

    def _ensure_llm_agents(self) -> None:
        if self.llm is not None:
            return
        self.llm = LLMClient(self.config)
        self.generation_agent = GenerationAgent(self.llm)
        self.traceability_agent.llm = self.llm
        self.feedback_agent = FeedbackAgent(self.config, self.llm, self.traceability_agent)

    def _issue_conditioned_docs(
        self,
        project: str,
        mode: str,
        issue_id: str,
        commit_docs: dict[str, CommitDoc],
    ) -> dict[str, CommitDoc]:
        """Return a per-issue view without leaking refined WHY across issues."""
        issue_docs = dict(commit_docs)
        refined_path = self._refined_cache(project, mode, issue_id)
        if not refined_path.exists():
            raise FileNotFoundError(
                f"Missing refined WHY cache for issue {issue_id}: {refined_path}"
            )
        for item in self._read_json(refined_path):
            cid = item["commit_id"]
            refined_why = item.get("judge_detail", {}).get("refined_why")
            if refined_why and cid in commit_docs:
                issue_docs[cid] = replace(commit_docs[cid], why=refined_why)
        return issue_docs

    def _ensure_dirs(self, project: str, mode: str) -> None:
        self.config.result_dir(project).mkdir(parents=True, exist_ok=True)
        for path in [
            self.config.cache_dir(project) / "retrieval" / mode,
            self.config.cache_dir(project) / "rerank" / mode,
            self.config.cache_dir(project) / "feedback" / mode,
            self.config.cache_dir(project) / "refined" / mode,
            self.config.cache_dir(project) / "final" / mode,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def _retrieval_cache(self, project: str, mode: str, issue_id: str) -> Path:
        return self.config.cache_dir(project) / "retrieval" / mode / f"{issue_id}.json"

    def _rerank_cache(self, project: str, mode: str, issue_id: str) -> Path:
        return self.config.cache_dir(project) / "rerank" / mode / f"{issue_id}.json"

    def _feedback_cache(self, project: str, mode: str, issue_id: str) -> Path:
        return self.config.cache_dir(project) / "feedback" / mode / f"{issue_id}.json"

    def _refined_cache(self, project: str, mode: str, issue_id: str) -> Path:
        return self.config.cache_dir(project) / "refined" / mode / f"{issue_id}.json"

    def _final_cache(self, project: str, mode: str, issue_id: str) -> Path:
        return self.config.cache_dir(project) / "final" / mode / f"{issue_id}.json"

    @staticmethod
    def _read_json(path: Path) -> Any:
        with path.open("r") as file:
            return json.load(file)

    @staticmethod
    def _write_json(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(data, file, indent=2)
