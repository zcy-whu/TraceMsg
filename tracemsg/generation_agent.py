"""Generation Agent: What/Why generation and feedback-guided Why refinement."""

from __future__ import annotations

from .llm import LLMClient


class GenerationAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_what(self, commit_diff: str) -> str:
        prompt = f"""Given a commit diff, generate the WHAT component of a structured commit message.

Commit diff:
{commit_diff}

Requirements:
1. Summarize observable implementation changes.
2. Include file-level modifications and cross-file relationships when visible.
3. Stay concise and do not infer motivations.

Output ONLY a JSON object with:
- what (string)
"""
        result = self.llm.complete_json(
            prompt, system_prompt="You are an expert software evolution analyst."
        )
        return str((result or {}).get("what", "")).strip()

    def generate_why(self, commit_diff: str, what: str) -> str:
        prompt = f"""Infer the WHY component of a structured commit message from code changes.

Commit diff:
{commit_diff}

WHAT:
{what}

Requirements:
1. Infer the most plausible engineering intent from the diff and WHAT.
2. Treat the rationale as a hypothesis, not a confirmed issue fix.
3. Avoid restating implementation details already covered by WHAT.
4. Keep it concise and engineering-oriented.

Output ONLY a JSON object with:
- why (string)
"""
        result = self.llm.complete_json(
            prompt, system_prompt="You are an expert software evolution analyst."
        )
        return str((result or {}).get("why", "")).strip()

    def refine_why(
        self,
        issue_text: str,
        original_why: str,
        linkage_decision: str,
        guidance: str,
    ) -> str:
        if linkage_decision in {"Supported", "Weakly Supported"}:
            prompt = f"""Refine the WHY statement of a commit using traceability feedback.

The commit is plausibly linked to the issue.

Issue:
{issue_text}

Original WHY:
{original_why}

Feedback:
{guidance}

Requirements:
1. Preserve the valid repair intent.
2. Emphasize the actual engineering motivation.
3. Explain why the modification is necessary.
4. Avoid implementation-level details already covered by WHAT.
5. Avoid copying issue text directly.
6. Keep the WHY concise.

Output ONLY a JSON object with:
- refined_why (string)
"""
        else:
            prompt = f"""Regenerate the WHY statement so it better reflects the commit itself.

The previous WHY may overstate a connection to the issue.

Issue:
{issue_text}

Original WHY:
{original_why}

Feedback:
{guidance}

Requirements:
1. Remove unsupported repair intentions.
2. Avoid implying issue-resolution goals not supported by the code changes.
3. Focus on the localized engineering purpose.
4. Keep the WHY concise and technically grounded.
5. Do not state that the commit is unrelated.

Output ONLY a JSON object with:
- refined_why (string)
"""
        result = self.llm.complete_json(
            prompt, system_prompt="You are an expert software evolution analyst."
        )
        return str((result or {}).get("refined_why", original_why)).strip() or original_why
