"""Within-skill question deduplication for synthetic generation.

This module deliberately has no benchmark-data dependency.  Test-set
overlap checking runs in :mod:`msi.audit.decontaminate`
after generation, so benchmark instances cannot enter the generator's
address space or model prompts.
"""

from __future__ import annotations

import re
import threading
from difflib import SequenceMatcher

from collections.abc import Iterable

from msi.training.provenance import normalize_question


class QuestionDeduplicator:
    """Reject exact and near duplicates within one skill's synthetic pool."""

    def __init__(
        self, *, dedup_threshold: float = 0.88,
        existing_questions: Iterable[str] = (),
    ) -> None:
        if not 0 < dedup_threshold <= 1:
            raise ValueError("dedup_threshold must be in (0, 1]")
        self.dedup_threshold = dedup_threshold
        self.seen = {
            normalized: self._number_signature(normalized)
            for question in existing_questions
            if (normalized := normalize_question(question))
        }
        self._lock = threading.Lock()

    @staticmethod
    def _number_signature(normalized: str) -> tuple[str, ...]:
        return tuple(re.findall(r"\d+(?:\.\d+)?", normalized))

    def _near_seen(self, normalized: str) -> tuple[str, float] | None:
        if len(normalized.split()) < 12:
            return None
        signature = self._number_signature(normalized)
        for previous, previous_signature in self.seen.items():
            if len(previous.split()) < 12:
                continue
            # Preserve legitimate parameter variants.  A wording-only rewrite
            # of the same numeric problem has the same number signature.
            if signature != previous_signature:
                continue
            score = SequenceMatcher(None, normalized, previous, autojunk=False).ratio()
            if score >= self.dedup_threshold:
                return previous, score
        return None

    def admit(self, question: object) -> tuple[bool, str]:
        normalized = normalize_question(question)
        if not normalized:
            return False, "empty question"
        with self._lock:
            if normalized in self.seen:
                return False, "duplicate normalized question"
            duplicate = self._near_seen(normalized)
            if duplicate:
                return False, f"near within-skill duplicate ({duplicate[1]:.4f})"
            self.seen[normalized] = self._number_signature(normalized)
        return True, "accepted"


def validate_admitted_instance(instance: dict) -> tuple[bool, str]:
    validation = instance.get("validation", {})
    trajectory = instance.get("trajectory", {})
    if trajectory.get("truncated"):
        return False, "truncated trajectory"
    if not validation.get("acceptable"):
        return False, str(validation.get("reason", "evaluator rejected trajectory"))
    if not str(validation.get("extracted_answer", "")).strip():
        return False, "missing extracted answer"
    if not trajectory.get("training_samples"):
        return False, "missing training samples"
    return True, "accepted"
