# -*- coding: utf-8 -*-
"""
Error Mapping - Map provider-specific errors to unified exceptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Callable, List, Optional, Type

# Import unified exceptions from exceptions.py
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMError,
    LLMRateLimitError,
    ProviderContextWindowError,
)

try:
    import openai  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    openai = None  # type: ignore
    _HAS_OPENAI = False


logger = logging.getLogger(__name__)


ErrorClassifier = Callable[[Exception], bool]


@dataclass(frozen=True)
class MappingRule:
    classifier: ErrorClassifier
    factory: Callable[[Exception, Optional[str]], LLMError]


def _instance_of(*types: Type[BaseException]) -> ErrorClassifier:
    return lambda exc: isinstance(exc, types)


def _message_contains(*needles: str) -> ErrorClassifier:
    def _classifier(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(needle in msg for needle in needles)

    return _classifier


_GLOBAL_RULES: List[MappingRule] = [
    MappingRule(
        classifier=_message_contains("rate limit", "429", "quota"),
        factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
    ),
    MappingRule(
        classifier=_message_contains("context length", "maximum context"),
        factory=lambda exc, provider: ProviderContextWindowError(str(exc), provider=provider),
    ),
]

if _HAS_OPENAI:
    _GLOBAL_RULES[:0] = [
        MappingRule(
            classifier=_instance_of(openai.AuthenticationError),
            factory=lambda exc, provider: LLMAuthenticationError(str(exc), provider=provider),
        ),
        MappingRule(
            classifier=_instance_of(openai.RateLimitError),
            factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
        ),
    ]

# Attempt to load Anthropic and Google rules if SDKs are present
try:
    import anthropic

    _GLOBAL_RULES.append(
        MappingRule(
            classifier=_instance_of(anthropic.RateLimitError),
            factory=lambda exc, provider: LLMRateLimitError(str(exc), provider=provider),
        )
    )
except ImportError:
    pass


def map_error(exc: Exception, provider: Optional[str] = None) -> LLMError:
    """Map provider-specific errors to unified internal exceptions."""
    # Heuristic check for status codes before rules
    status_code = getattr(exc, "status_code", None)
    details = getattr(exc, "details", {})
    retry_after = details.get("retry_after")

    if status_code == 401:
        return LLMAuthenticationError(str(exc), provider=provider)
    if status_code == 429:
        # Try to parse retry_after if it's a string
        parsed_retry_after = None
        if retry_after:
            try:
                parsed_retry_after = float(retry_after)
            except (ValueError, TypeError):
                pass
        return LLMRateLimitError(str(exc), retry_after=parsed_retry_after, provider=provider)

    for rule in _GLOBAL_RULES:
        if rule.classifier(exc):
            mapped = rule.factory(exc, provider)
            # Ensure mapped error also gets retry_after if applicable
            if isinstance(mapped, LLMRateLimitError) and not mapped.retry_after:
                mapped.retry_after = parsed_retry_after if 'parsed_retry_after' in locals() else None
            return mapped

    return LLMAPIError(str(exc), status_code=status_code, provider=provider)
