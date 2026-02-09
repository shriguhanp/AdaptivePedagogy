# -*- coding: utf-8 -*-
"""
LLM Factory - Central Hub for LLM Calls
=======================================

This module serves as the central hub for all LLM calls in DeepTutor.
It provides a unified interface for agents to call LLMs, routing requests
to the appropriate provider (cloud or local) based on URL detection.

Architecture:
    Agents (ChatAgent, GuideAgent, etc.)
              ↓
         BaseAgent.call_llm() / stream_llm()
              ↓
         LLM Factory (this module)
              ↓
    ┌─────────┴─────────┐
    ↓                   ↓
CloudProvider      LocalProvider
(cloud_provider)   (local_provider)
              ↓                   ↓
OpenAI/DeepSeek/etc    LM Studio/Ollama/etc

Routing:
- Automatically routes to local_provider for local URLs (localhost, 127.0.0.1, etc.)
- Routes to cloud_provider for all other URLs

Retry Mechanism:
- Automatic retry with exponential backoff for transient errors
- Configurable max_retries, retry_delay, and exponential_backoff
- Only retries on retriable errors (timeout, rate limit, server errors)
"""

import asyncio
from typing import Any, AsyncGenerator, Dict, List, Optional

import tenacity

from src.logging.logger import get_logger

from . import cloud_provider, local_provider
from .config import get_llm_config
from .exceptions import (
    LLMAPIError,
    LLMAuthenticationError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from .utils import is_local_llm_server

# Initialize logger
logger = get_logger("LLMFactory")

# Default retry configuration
DEFAULT_MAX_RETRIES = 15  # Increased to allow waiting for quota resets
DEFAULT_RETRY_DELAY = 1.0  # seconds
DEFAULT_EXPONENTIAL_BACKOFF = True

# Fallback models (if primary hits rate limits)
FALLBACK_MODELS = {
    "llama-3.3-70b-versatile": "llama-3.1-8b-instant",
    "llama-3.3-70b-specdec": "llama-3.1-8b-instant",
    "llama3-70b-8192": "llama3-8b-8192",
    "deepseek-reasoner": "deepseek-chat",
}


def _is_retriable_error(error: Exception) -> bool:
    """
    Check if an error is retriable.

    Retriable errors:
    - Timeout errors
    - Rate limit errors (429)
    - Server errors (5xx)
    - Network/connection errors

    Non-retriable errors:
    - Authentication errors (401)
    - Bad request (400)
    - Not found (404)
    - Client errors (4xx except 429)
    """
    from aiohttp import ClientError
    from requests.exceptions import RequestException

    if isinstance(error, (asyncio.TimeoutError, ClientError, RequestException)):
        return True
    if isinstance(error, LLMTimeoutError):
        return True
    if isinstance(error, LLMRateLimitError):
        return True
    if isinstance(error, LLMAuthenticationError):
        return False  # Don't retry auth errors

    if isinstance(error, LLMAPIError):
        status_code = error.status_code
        if status_code:
            # Retry on server errors (5xx) and rate limits (429)
            if status_code >= 500 or status_code == 429:
                return True
            # Don't retry on client errors (4xx except 429)
            if 400 <= status_code < 500:
                return False
        return True  # Retry by default for unknown API errors

    # For other exceptions (network errors, etc.), retry
    return True


class WaitRetryAfter(tenacity.wait.wait_base):
    """
    Wait strategy that respects Retry-After headers in LLMRateLimitError.
    Falls back to another wait strategy if Retry-After is not present.
    """

    def __init__(self, fallback_wait: tenacity.wait.wait_base):
        self.fallback_wait = fallback_wait

    def __call__(self, retry_state: tenacity.RetryCallState) -> float:
        if retry_state.outcome.failed:
            exc = retry_state.outcome.exception()
            
            # Check if model has changed (fallback in progress)
            # We look at the actual model used in the last attempt vs current config
            # But easiest is to check if 'fallback' was logged or just return 0 for first few retries if 429
            
            # KEY CHANGE: If a fallback model is available for the current model,
            # we should NOT wait. We should return 0 so the retry loop can
            # immediately try the fallback model.
            # Since we don't have easy access to the 'model' variable here, 
            # we rely on the fact that the retry loop logic in _do_complete will 
            # switch the model if a fallback exists.
            
            # However, to be safe and avoid hammering the API if fallback fails too,
            # we'll use a small trick: 
            # If this is a rate limit error, the _do_complete wrapper will catch it
            # and try to switch model. If it switches, it calls itself recursively.
            # If it raises (because no fallback), THEN this wait strategy is called
            # by the outer tenacity retry.
            
            # So if we are here, it means either:
            # 1. No fallback was found
            # 2. The fallback also hit a rate limit
            
            # In either case, we SHOULD respect the retry_after.
            pass

            if isinstance(exc, LLMRateLimitError) and getattr(exc, "retry_after", None):
                wait_time = float(exc.retry_after) + 1.0
                logger.warning(f"Rate limit hit. Respecting Retry-After: waiting {wait_time}s")
                return wait_time

        return self.fallback_wait(retry_state)


def _should_use_local(base_url: Optional[str]) -> bool:
    """
    Determine if we should use the local provider based on URL.

    Args:
        base_url: The base URL to check

    Returns:
        True if local provider should be used (localhost, 127.0.0.1, etc.)
    """
    return is_local_llm_server(base_url) if base_url else False


async def complete(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs,
) -> str:
    """
    Unified LLM completion function with automatic retry.

    Routes to cloud_provider or local_provider based on configuration.
    Includes automatic retry with exponential backoff for transient errors.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name (optional, uses effective config if not provided)
        api_key: API key (optional)
        base_url: Base URL for the API (optional)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding type (optional)
        messages: Pre-built messages array (optional)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Returns:
        str: The LLM response
    """
    # Get config if parameters not provided
    if not model or not base_url:
        config = get_llm_config()
        model = model or config.model
        api_key = api_key if api_key is not None else config.api_key
        base_url = base_url or config.base_url
        api_version = api_version or config.api_version
        binding = binding or config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    # Define helper to determine if a generic LLMAPIError is retriable
    def _is_retriable_llm_api_error(exc: BaseException) -> bool:
        """
        Return True for LLMAPIError instances that represent retriable conditions.

        We only retry on:
          - HTTP 429 (rate limit), or
          - HTTP 5xx server errors.

        All other LLMAPIError instances (e.g., 4xx like 400, 401, 403, 404) are treated
        as non-retriable to avoid unnecessary retries.
        """
        if not isinstance(exc, LLMAPIError):
            return False

        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            # Do not retry when status code is unknown to avoid retrying non-transient errors
            return False

        if status_code == 429:
            return True

        if 500 <= status_code < 600:
            return True

        return False

    # Calculate total attempts for logging (1 initial + max_retries)
    total_attempts = max_retries + 1

    # Define the actual completion function with tenacity retry
    @tenacity.retry(
        retry=(
            tenacity.retry_if_exception_type(LLMRateLimitError)
            | tenacity.retry_if_exception_type(LLMTimeoutError)
            | tenacity.retry_if_exception(_is_retriable_llm_api_error)
        ),
        wait=WaitRetryAfter(
            tenacity.wait_exponential(multiplier=retry_delay, min=retry_delay, max=120)
        ),
        stop=tenacity.stop_after_attempt(total_attempts),
        before_sleep=lambda retry_state: logger.warning(
            f"LLM call failed (attempt {retry_state.attempt_number}/{total_attempts}), "
            f"retrying in {retry_state.upcoming_sleep:.1f}s... Error: {str(retry_state.outcome.exception())}"
        ),
    )
    async def _do_complete(**call_kwargs):
        nonlocal model
        try:
            if use_local:
                return await local_provider.complete(**call_kwargs)
            else:
                return await cloud_provider.complete(**call_kwargs)
        except Exception as e:
            # Map raw SDK exceptions to unified exceptions for retry logic
            from .error_mapping import map_error

            mapped_error = map_error(e, provider=call_kwargs.get("binding", "unknown"))

            # Check if we should try a fallback model (for Rate Limits)
            if isinstance(mapped_error, LLMRateLimitError):
                fallback = call_kwargs.get("fallback_model") or FALLBACK_MODELS.get(model)
                if fallback and fallback != model:
                    logger.warning(
                        f"Rate limit hit for {model}. Switching to fallback {fallback} immediately..."
                    )
                    model = fallback
                    call_kwargs["model"] = fallback
                    
                    # Update binding/base_url if needed for fallback model? 
                    # Assuming fallback models are on the same provider/binding for now, 
                    # or compatible (e.g. both OpenAI-compatible).
                    # If switching from Llama to something else, might need careful handling,
                    # but for now we assume fallback works with current config or default.
                    
                    # Call the decorated function recursively to start a fresh retry cycle with the new model
                    return await _do_complete(**call_kwargs)

            # Re-raise the mapped error
            raise mapped_error from e

    # Build call kwargs
    call_kwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "messages": messages,
        **kwargs,
    }

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    # Execute with retry (handled by tenacity decorator)
    return await _do_complete(**call_kwargs)


async def stream(
    prompt: str,
    system_prompt: str = "You are a helpful assistant.",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_version: Optional[str] = None,
    binding: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    max_retries: int = DEFAULT_MAX_RETRIES,
    retry_delay: float = DEFAULT_RETRY_DELAY,
    exponential_backoff: bool = DEFAULT_EXPONENTIAL_BACKOFF,
    **kwargs,
) -> AsyncGenerator[str, None]:
    """
    Unified LLM streaming function with automatic retry.

    Routes to cloud_provider or local_provider based on configuration.
    Includes automatic retry with exponential backoff for connection errors.

    Note: Retry only applies to initial connection errors. Once streaming
    starts, errors during streaming will not be automatically retried.

    Args:
        prompt: The user prompt
        system_prompt: System prompt for context
        model: Model name (optional, uses effective config if not provided)
        api_key: API key (optional)
        base_url: Base URL for the API (optional)
        api_version: API version for Azure OpenAI (optional)
        binding: Provider binding type (optional)
        messages: Pre-built messages array (optional)
        max_retries: Maximum number of retry attempts (default: 5)
        retry_delay: Initial delay between retries in seconds (default: 2.0)
        exponential_backoff: Whether to use exponential backoff (default: True)
        **kwargs: Additional parameters (temperature, max_tokens, etc.)

    Yields:
        str: Response chunks
    """
    # Get config if parameters not provided
    if not model or not base_url:
        config = get_llm_config()
        model = model or config.model
        api_key = api_key if api_key is not None else config.api_key
        base_url = base_url or config.base_url
        api_version = api_version or config.api_version
        binding = binding or config.binding or "openai"

    # Determine which provider to use
    use_local = _should_use_local(base_url)

    # Build call kwargs
    call_kwargs = {
        "prompt": prompt,
        "system_prompt": system_prompt,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
        "messages": messages,
        **kwargs,
    }

    # Add cloud-specific kwargs if not local
    if not use_local:
        call_kwargs["api_version"] = api_version
        call_kwargs["binding"] = binding or "openai"

    # Retry logic for streaming (retry on connection errors)
    # Total attempts = 1 initial + max_retries
    total_attempts = max_retries + 1
    last_exception = None
    delay = retry_delay
    max_delay = 120  # Cap maximum delay at 120 seconds (consistent with complete())

    for attempt in range(total_attempts):
        try:
            # Route to appropriate provider
            if use_local:
                async for chunk in local_provider.stream(**call_kwargs):
                    yield chunk
            else:
                async for chunk in cloud_provider.stream(**call_kwargs):
                    yield chunk
            # If we get here, streaming completed successfully
            return
        except Exception as e:
            last_exception = e
            
            # Check if we should try a fallback model on rate limit
            if isinstance(e, LLMRateLimitError):
                fallback = kwargs.get("fallback_model") or FALLBACK_MODELS.get(model)
                if fallback and fallback != model:
                    logger.warning(f"Rate limit hit for {model} during streaming. Switching to fallback {fallback} immediately...")
                    model = fallback
                    call_kwargs["model"] = fallback
                    # Continue loop to retry with new model immediately
                    continue

            # Check if we should retry
            if attempt >= max_retries or not _is_retriable_error(e):
                raise

            # Calculate delay for next attempt
            if exponential_backoff:
                current_delay = min(delay * (2**attempt), max_delay)
            else:
                current_delay = delay

            # Special handling for rate limit errors with retry_after
            if isinstance(e, LLMRateLimitError) and e.retry_after:
                current_delay = max(current_delay, e.retry_after)

            # Log retry attempt (consistent with complete() function)
            logger.warning(
                f"LLM streaming failed (attempt {attempt + 1}/{total_attempts}), "
                f"retrying in {current_delay:.1f}s... Error: {str(e)}"
            )

            # Wait before retrying
            await asyncio.sleep(current_delay)

    # Should not reach here, but just in case
    if last_exception:
        raise last_exception


async def fetch_models(
    binding: str,
    base_url: str,
    api_key: Optional[str] = None,
) -> List[str]:
    """
    Fetch available models from the provider.

    Routes to cloud_provider or local_provider based on URL.

    Args:
        binding: Provider type (openai, ollama, etc.)
        base_url: API endpoint URL
        api_key: API key (optional for local providers)

    Returns:
        List of available model names
    """
    if is_local_llm_server(base_url):
        return await local_provider.fetch_models(base_url, api_key)
    else:
        return await cloud_provider.fetch_models(base_url, api_key, binding)


# API Provider Presets
API_PROVIDER_PRESETS = {
    "openai": {
        "name": "OpenAI",
        "base_url": "https://api.openai.com/v1",
        "requires_key": True,
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
    },
    "anthropic": {
        "name": "Anthropic",
        "base_url": "https://api.anthropic.com/v1",
        "requires_key": True,
        "binding": "anthropic",
        "models": ["claude-3-5-sonnet-20241022", "claude-3-haiku-20240307"],
    },
    "gemini": {
        "name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "requires_key": True,
        "binding": "gemini",
        "models": ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
    },
    "deepseek": {
        "name": "DeepSeek",
        "base_url": "https://api.deepseek.com",
        "requires_key": True,
        "models": ["deepseek-chat", "deepseek-reasoner"],
    },
    "openrouter": {
        "name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "requires_key": True,
        "models": [],  # Dynamic
    },
}

# Local Provider Presets
LOCAL_PROVIDER_PRESETS = {
    "ollama": {
        "name": "Ollama",
        "base_url": "http://localhost:11434/v1",
        "requires_key": False,
        "default_key": "ollama",
    },
    "lm_studio": {
        "name": "LM Studio",
        "base_url": "http://localhost:1234/v1",
        "requires_key": False,
        "default_key": "lm-studio",
    },
    "vllm": {
        "name": "vLLM",
        "base_url": "http://localhost:8000/v1",
        "requires_key": False,
        "default_key": "vllm",
    },
    "llama_cpp": {
        "name": "llama.cpp",
        "base_url": "http://localhost:8080/v1",
        "requires_key": False,
        "default_key": "llama-cpp",
    },
}


def get_provider_presets() -> Dict[str, Any]:
    """
    Get all provider presets for frontend display.
    """
    return {
        "api": API_PROVIDER_PRESETS,
        "local": LOCAL_PROVIDER_PRESETS,
    }


__all__ = [
    "complete",
    "stream",
    "fetch_models",
    "get_provider_presets",
    "API_PROVIDER_PRESETS",
    "LOCAL_PROVIDER_PRESETS",
    # Retry configuration defaults
    "DEFAULT_MAX_RETRIES",
    "DEFAULT_RETRY_DELAY",
    "DEFAULT_EXPONENTIAL_BACKOFF",
]
