"""Feedback Agent: maintenance-pattern retrieval and reasoning feedback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import TraceMsgConfig
from .data import CommitDoc
from .llm import LLMClient
from .traceability_agent import TraceabilityAgent


class FeedbackAgent:
    def __init__(
        self,
        config: TraceMsgConfig,
        llm: LLMClient,
        traceability_agent: TraceabilityAgent,
    ):
        self.config = config
        self.llm = llm
        self.traceability_agent = traceability_agent

    def load_knowledge_base(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r") as file:
            return json.load(file)

    def retrieve_patterns(
        self,
        pair_abstraction: dict[str, Any],
        knowledge_base: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not knowledge_base:
            return []
        query = json.dumps(pair_abstraction, sort_keys=True)
        pattern_texts = [self._pattern_text(entry) for entry in knowledge_base]
        query_vec = np.asarray(
            self.traceability_agent.embeddings.embed_documents([query]), dtype="float32"
        )
        pattern_vecs = np.asarray(
            self.traceability_agent.embeddings.embed_documents(pattern_texts), dtype="float32"
        )
        scores = np.dot(query_vec, pattern_vecs.T)[0]
        order = np.argsort(-scores)[: self.config.feedback_pattern_top_k]
        return [knowledge_base[int(index)] for index in order]

    def abstract_pair(self, issue_text: str, commit_doc: CommitDoc) -> dict[str, Any]:
        """Construct the candidate pair abstractions used by Eq. (11)."""
        prompt = f"""Abstract an issue and a structured commit message into transferable maintenance characteristics.

Issue:
{issue_text}

Structured commit message:
WHAT: {commit_doc.what}
WHY: {commit_doc.why}

Output ONLY a JSON object with:
- issue_abstraction (object with symptom_type, failure_behavior, affected_entity, execution_context, semantic_clues, possible_failure_mechanisms)
- commit_abstraction (object with fix_operation, code_target, root_cause_type, fix_strategy, structural_change, semantic_intent)

Use generalized terminology and avoid project-specific wording where possible.
"""
        return self.llm.complete_json(
            prompt, system_prompt="You are an expert software maintenance analyst."
        ) or {"issue_abstraction": {}, "commit_abstraction": {}}

    def judge_pair(
        self,
        pair_abstraction: dict[str, Any],
        patterns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        pattern_text = "\n\n".join(self._pattern_text(pattern) for pattern in patterns) or "None"
        prompt = f"""Determine whether the candidate commit is plausibly linked to the issue.

Use retrieved software maintenance patterns as grounding evidence.

Candidate Pair Abstraction:
{json.dumps(pair_abstraction, indent=2)}

Retrieved Knowledge Patterns:
{pattern_text}

Reason through:
1. The engineering problem described by the issue.
2. Whether the commit repair strategy matches known repair patterns.
3. Failure-mechanism, repair-intention, and architecture consistency.
4. Supporting, missing, and contradictory evidence.

Output ONLY a JSON object with:
- linkage_decision ("Supported", "Weakly Supported", "Unsupported")
- confidence (float between 0.0 and 1.0)
- explanation (string)
- refinement_guidance_for_why (string)
"""
        result = self.llm.complete_json(
            prompt,
            system_prompt="You are an expert software evolution reviewer.",
            model=self.config.judge_model,
        )
        if result:
            return result
        return {
            "linkage_decision": "Unsupported",
            "confidence": 0.0,
            "explanation": "The LLM did not return valid JSON.",
            "refinement_guidance_for_why": "",
        }

    @staticmethod
    def _pattern_text(entry: dict[str, Any]) -> str:
        pattern = entry.get("pattern", entry)
        issue_abstraction = entry.get("issue_abstraction", "")
        commit_abstraction = entry.get("commit_abstraction", "")
        transferable = pattern.get("transferable_rule", "")
        rationale = pattern.get("alignment_rationale", "")
        cause = pattern.get("cause_pattern", "")
        fix = pattern.get("fix_pattern", "")
        return (
            f"Failure abstraction: {issue_abstraction}\n"
            f"Repair abstraction: {commit_abstraction}\n"
            f"Rule: {transferable}\n"
            f"Cause pattern: {cause}\n"
            f"Fix pattern: {fix}\n"
            f"Rationale: {rationale}"
        ).strip()
