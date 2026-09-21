"""
Evidence-verification agent — Objective 2.

Given a question and the evidence retrieved so far, decides whether that
evidence is sufficient to answer reliably. If not, the QA graph (see
qa_pipeline.py) loops back for another retrieval pass with a widened
strategy, up to a bounded number of retries.
"""

from __future__ import annotations

import json
import logging

from .config import PrivGraphAIConfig
from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.verification_agent")

_VERIFY_PROMPT = """You are checking whether the EVIDENCE below is sufficient to answer the QUESTION.

Return ONLY a JSON object: {{"sufficient": true|false, "missing": "..."}}
"missing" should briefly say what's missing, or be an empty string if sufficient.

QUESTION: {question}

EVIDENCE:
{evidence}
"""


class VerificationAgent:
    def __init__(self, config: PrivGraphAIConfig, ollama_client: OllamaClient):
        self.config = config
        self.llm = ollama_client

    def verify(self, question: str, evidence_text: str) -> tuple[bool, str]:
        if not evidence_text.strip():
            return False, "no evidence retrieved"

        try:
            raw = self.llm.generate_text(
                _VERIFY_PROMPT.format(question=question, evidence=evidence_text[:3000]),
                self.config.text_model,
                temperature=0.0,
            )
        except Exception as exc:
            logger.warning("Verification LLM call failed, assuming insufficient: %s", exc)
            return False, "verification call failed"

        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1 or end < start:
            logger.warning("Verification returned unparseable output, assuming insufficient. Raw output: %.300s", raw)
            return False, "verification output unparseable"
        try:
            obj = json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            logger.warning("Verification JSON failed to parse, assuming insufficient. Raw output: %.300s", raw)
            return False, "verification output unparseable"

        sufficient = self._coerce_bool(obj.get("sufficient", False))
        missing = obj.get("missing", "")
        logger.info(
            "Verification result: sufficient=%s missing=%r (raw model output: %.300s)", sufficient, missing, raw
        )
        return sufficient, missing

    @staticmethod
    def _coerce_bool(value) -> bool:
        """Handle small models that sometimes return "true"/"false" as a
        JSON string instead of a real boolean — bool("false") is True in
        Python, so a naive bool(value) would silently invert that case."""
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)
