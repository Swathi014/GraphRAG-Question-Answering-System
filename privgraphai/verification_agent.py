"""Evidence-verification agent - Objective 2.

PROPOSED version. What changed and why:
  * The old verifier asked a 3B model to judge "is this sufficient?". In the pilot it said
    "insufficient" for almost every question, whether or not the answer was in the evidence,
    so its verdict carried no information.
  * The new verifier asks the model to COPY the sentences that help answer the question,
    then checks IN CODE that each copied quote really occurs in the evidence. Evidence is
    "sufficient" only if at least one quote is found there. A made-up quote cannot pass.
  * The old JSON format ({"sufficient": ..., "missing": ...}) is still understood, so existing
    tests and any scripted model outputs keep working.
  * The evidence is no longer cut at 3000 characters (the qa pipeline now budgets it).
"""

from __future__ import annotations

import json
import logging
import re

from .config import PrivGraphAIConfig
from .ollama_client import OllamaClient

logger = logging.getLogger("privgraphai.verification_agent")

_MAX_VERIFY_CHARS = 9000  # safety net only; the QA pipeline already budgets the evidence

_VERIFY_PROMPT = """Find the parts of the EVIDENCE that help answer the QUESTION.

Return ONLY a JSON object: {{"quotes": ["..."]}}

Rules:
- Each quote must be copied EXACTLY, word for word, from the EVIDENCE (a sentence, a table row, or a formula).
- Give at most 3 quotes, the most useful ones.
- If nothing in the EVIDENCE helps, return {{"quotes": []}}.

QUESTION: {question}

EVIDENCE:
{evidence}
"""


def _norm(text: str) -> str:
    text = text.lower().replace("|", " ").replace("*", " ").replace("`", " ")
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.\-+][a-z0-9]+)*", text.lower())


def _trigrams(tokens: list[str]) -> set[tuple]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


class VerificationAgent:
    def __init__(self, config: PrivGraphAIConfig, ollama_client: OllamaClient):
        self.config = config
        self.llm = ollama_client

    # ------------------------------------------------------------------ public
    def verify(self, question: str, evidence_text: str) -> tuple[bool, str]:
        if not evidence_text.strip():
            return False, "no evidence retrieved"

        try:
            raw = self.llm.generate_text(
                _VERIFY_PROMPT.format(question=question, evidence=evidence_text[:_MAX_VERIFY_CHARS]),
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

        if "quotes" in obj:  # new format: grounded check done in code
            quotes = obj.get("quotes") or []
            if isinstance(quotes, str):
                quotes = [quotes]
            found = [q for q in quotes if isinstance(q, str) and self._supported(q, evidence_text)]
            sufficient = bool(found)
            missing = "" if sufficient else f"no passage in the evidence answers: {question}"
            logger.info(
                "Verification result: sufficient=%s missing=%r (quotes given: %d, found in evidence: %d)",
                sufficient, missing, len(quotes), len(found),
            )
            return sufficient, missing

        # legacy format: {"sufficient": ..., "missing": ...}
        sufficient = self._coerce_bool(obj.get("sufficient", False))
        missing = obj.get("missing", "")
        logger.info(
            "Verification result: sufficient=%s missing=%r (raw model output: %.300s)", sufficient, missing, raw
        )
        return sufficient, missing

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _supported(quote: str, evidence: str) -> bool:
        """True if the quote really occurs in the evidence (exact, or >= 90% of word triples)."""
        quote = quote.strip().strip(".\u2026").strip()
        if len(quote) < 8:
            return False
        if _norm(quote) in _norm(evidence):
            return True
        q_tokens = _tokens(quote)
        if len(q_tokens) < 4:
            return False
        q_grams = _trigrams(q_tokens)
        if not q_grams:
            return False
        overlap = len(q_grams & _trigrams(_tokens(evidence))) / len(q_grams)
        return overlap >= 0.9

    @staticmethod
    def _coerce_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return bool(value)