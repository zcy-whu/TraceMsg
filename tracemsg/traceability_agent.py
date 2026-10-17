"""Traceability Agent: dense retrieval and RankGPT-style reranking."""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np

from .config import TraceMsgConfig
from .data import CommitDoc
from .llm import LLMClient


class TraceabilityAgent:
    def __init__(self, config: TraceMsgConfig, llm: Optional[LLMClient] = None):
        self.config = config
        self.llm = llm
        model_name = (
            str(config.local_embedding_model_path)
            if config.local_embedding_model_path.exists()
            else config.embedding_model
        )
        self.embeddings = SentenceTransformerEmbeddings(
            model_name=model_name,
            normalize_embeddings=config.normalize_embeddings,
        )

    def retrieve(
        self,
        issue_text: str,
        candidate_commit_ids: list[str],
        commit_docs: dict[str, CommitDoc],
        mode: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> list[dict]:
        mode = mode or self.config.retrieval_mode
        candidates = [cid for cid in candidate_commit_ids if cid in commit_docs]
        if not candidates:
            return []

        if top_k is None:
            top_k = max(
                self.config.initial_top_k,
                math.ceil(len(candidates) * self.config.retrieval_ratio),
            )
        top_k = min(top_k, len(candidates))

        issue_vec = np.asarray(self.embeddings.embed_documents([issue_text]), dtype="float32")

        # Eq. (4) embeds M(c) once.  For the complete variant, M(c) is the
        # concatenated structured message {WHAT, WHY}, not an ensemble of
        # three independently computed similarities.
        vecs = self._embed([commit_docs[cid].text_for_mode(mode) for cid in candidates])
        scores = np.dot(issue_vec, vecs.T)[0]

        ranked = [
            {"commit_id": cid, "retrieval_score": float(scores[index])}
            for index, cid in enumerate(candidates)
        ]
        return sorted(ranked, key=lambda item: item["retrieval_score"], reverse=True)[:top_k]

    def rerank(
        self,
        issue_text: str,
        candidate_commits: list[dict],
        commit_docs: dict[str, CommitDoc],
        mode: Optional[str] = None,
    ) -> list[dict]:
        if not self.llm:
            return candidate_commits
        mode = mode or self.config.retrieval_mode
        items = copy.deepcopy(candidate_commits)
        if len(items) <= self.config.rerank_window_size:
            return self._rerank_window(issue_text, items, commit_docs, mode)

        end_pos = len(items)
        start_pos = end_pos - self.config.rerank_window_size
        while start_pos >= 0:
            start_pos = max(start_pos, 0)
            window_items = items[start_pos:end_pos]
            items[start_pos:end_pos] = self._rerank_window(issue_text, window_items, commit_docs, mode)
            if start_pos == 0:
                break
            end_pos -= self.config.rerank_step_size
            start_pos -= self.config.rerank_step_size
        return items

    def _embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.embeddings.embed_documents(texts), dtype="float32")

    def _rerank_window(
        self,
        issue_text: str,
        items: list[dict],
        commit_docs: dict[str, CommitDoc],
        mode: str,
    ) -> list[dict]:
        if len(items) <= 1:
            return items

        prompt = f"""You are RankGPT, an intelligent assistant that ranks candidate commits by relevance to a bug issue.

Issue:
{issue_text}

Candidate commits:
"""
        for index, item in enumerate(items, start=1):
            cid = item["commit_id"]
            doc = commit_docs[cid]
            if mode == "what":
                body = f"WHAT: {doc.what}"
            elif mode == "why":
                body = f"WHY: {doc.why}"
            else:
                body = f"WHAT: {doc.what}\nWHY: {doc.why}"
            prompt += f"[{index}-{cid}] {body}\n\n"

        prompt += f"""Constraint:
Rank the {len(items)} candidate commits in descending relevance.
Only output a JSON object containing an array of strings named "ranking".
Example:
{{"ranking": ["2-abc1234", "1-def5678"]}}
"""
        result = self.llm.complete_json(
            prompt,
            system_prompt="You are an expert software maintenance analyst.",
            model=self.config.ranker_model,
        )
        ranking = (result or {}).get("ranking", [])
        order = self._parse_rank_indices(ranking, len(items))
        return [items[index - 1] for index in order]

    @staticmethod
    def _parse_rank_indices(ranking: list, n_items: int) -> list[int]:
        parsed: list[int] = []
        for value in ranking:
            if not isinstance(value, str) or "-" not in value:
                continue
            prefix = value.split("-", 1)[0]
            if prefix.isdigit():
                index = int(prefix)
                if 1 <= index <= n_items and index not in parsed:
                    parsed.append(index)
        for index in range(1, n_items + 1):
            if index not in parsed:
                parsed.append(index)
        return parsed


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str, normalize_embeddings: bool = True):
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TraceMsg retrieval requires sentence-transformers. "
                "Install dependencies with `pip install -r TraceMsg/requirements.txt` "
                "or create the repository conda environment from environment.yml."
            ) from exc
        self.model = SentenceTransformer(model_name)
        self.normalize_embeddings = normalize_embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=False,
        )
        return vectors.tolist()
