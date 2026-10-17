"""LLM helpers used by the three TraceMsg agents."""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from .config import TraceMsgConfig


class LLMClient:
    def __init__(self, config: TraceMsgConfig):
        self.config = config
        api_key = os.getenv(config.qwen_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing {config.qwen_api_key_env}. Set it before running TraceMsg online stages."
            )
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key, base_url=config.qwen_base_url)

    def complete(self, prompt: str, system_prompt: str, model: Optional[str] = None) -> str:
        response = self.client.chat.completions.create(
            model=model or self.config.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.llm_temperature,
        )
        return response.choices[0].message.content or ""

    def complete_json(
        self, prompt: str, system_prompt: str, model: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        content = self.complete(prompt, system_prompt=system_prompt, model=model)
        return parse_json_object(content)


def parse_json_object(content: str) -> Optional[dict[str, Any]]:
    text = content.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None
