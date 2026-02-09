#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Error Handler - Error handling and retry mechanism
"""

from collections.abc import Callable
from typing import Any, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator
import tenacity

from src.config.constants import VALID_INVESTIGATE_TOOLS, VALID_SOLVE_TOOLS
from src.logging.logger import get_logger
from src.services.llm.exceptions import LLMParseError


def _format_validation_errors(e: ValidationError) -> str:
    """Format Pydantic validation errors into a readable string."""
    return "; ".join(
        [f"{'.'.join(str(x) for x in err['loc']) or 'root'}: {err['msg']}" for err in e.errors()]
    )


# Pydantic models for output validation
class ToolIntent(BaseModel):
    """Model for tool intent in investigate output"""

    tool_type: str = Field(..., description="Type of tool to use")
    query: str = Field("", description="Query for the tool")
    identifier: Optional[str] = Field(None, description="Optional identifier")

    @field_validator("tool_type")
    @classmethod
    def validate_tool_type(cls, v):
        if v.lower() not in VALID_INVESTIGATE_TOOLS:
            raise ValueError(f"tool_type must be one of {VALID_INVESTIGATE_TOOLS}, got: {v}")
        return v.lower()

    @field_validator("query")
    @classmethod
    def validate_query_required(cls, v, info):
        tool_type = info.data.get("tool_type", "").lower()
        if tool_type != "none" and not v:
            raise ValueError("query is required for non-none tools")
        return v


class InvestigateOutput(BaseModel):
    """Model for InvestigateAgent output"""

    reasoning: str = Field(..., description="Reasoning for the investigation")
    tools: list[ToolIntent] = Field(..., min_length=1, description="List of tool intents")

    @field_validator("tools")
    @classmethod
    def validate_tools_consistency(cls, v):
        # Check for 'none' tool exclusivity
        has_none = any(tool.tool_type == "none" for tool in v)
        if has_none and len(v) > 1:
            raise ValueError("When 'none' tool exists, no other tool intents should be provided")
        return v


class Citation(BaseModel):
    """Model for citation in note output"""

    reference_id: Optional[str] = None
    source: Optional[str] = None
    content: Optional[str] = None

    @model_validator(mode="after")
    def validate_citation_fields(self):
        """Validate that at least one of reference_id or source is provided."""
        if not self.reference_id and not self.source:
            raise ValueError("citation must contain reference_id or source")
        return self


class NoteOutput(BaseModel):
    """Model for NoteAgent output"""

    summary: str = Field(..., description="Summary of the notes")
    citations: list[Citation] = Field(default_factory=list, description="List of citations")


class ReflectOutput(BaseModel):
    """Model for InvestigateReflectAgent output"""

    should_stop: bool = Field(..., description="Whether to stop the investigation")
    reason: str = Field(..., description="Reason for the decision")
    remaining_questions: list[str] = Field(..., description="List of remaining questions")


class PlanStep(BaseModel):
    """Model for plan step"""

    step_id: str = Field(..., description="Step identifier")
    plan: str = Field(..., description="Plan description")


class PlanBlock(BaseModel):
    """Model for plan block"""

    block_id: str = Field(..., description="Block identifier")
    format: str = Field(..., description="Output format")
    steps: list[PlanStep] = Field(..., min_length=1, description="List of steps")


class PlanOutput(BaseModel):
    """Model for PlanAgent output"""

    answer_style: str = Field(..., description="Style of the answer")
    blocks: list[PlanBlock] = Field(..., min_length=1, description="List of plan blocks")


class SolveToolCall(BaseModel):
    """Model for tool call in solve output"""

    tool_type: str = Field(..., description="Type of tool to call")
    query: str = Field(..., description="Query for the tool call")

    @field_validator("tool_type")
    @classmethod
    def validate_tool_type(cls, v):
        if v.lower() not in VALID_SOLVE_TOOLS:
            raise ValueError(f"tool_type must be one of {VALID_SOLVE_TOOLS}, got: {v}")
        return v.lower()


class SolveOutput(BaseModel):
    """Model for SolveAgent output"""

    tool_calls: list[SolveToolCall] = Field(..., min_length=1, description="List of tool calls")


# Initialize module logger
logger = get_logger("ErrorHandler")


def retry_on_parse_error(
    max_retries: int = 2,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (LLMParseError,),
):
    """
    Parse error retry decorator using tenacity.

    Args:
        max_retries: Maximum retry count
        delay: Initial delay time (seconds)
        backoff: Delay multiplier factor
        exceptions: Tuple of exception types to retry on (default: LLMParseError only)

    Returns:
        Decorated function
    """

    def decorator(func: Callable):
        return tenacity.retry(
            retry=tenacity.retry_if_exception_type(*exceptions),
            wait=tenacity.wait_exponential(multiplier=backoff, min=delay, max=60),
            stop=tenacity.stop_after_attempt(max_retries + 1),
            before_sleep=lambda retry_state: logger.warning(
                f"Parse failed (attempt {retry_state.attempt_number}/{max_retries + 1}), "
                f"retrying in {retry_state.upcoming_sleep:.1f}s... Error: {str(retry_state.outcome.exception())}"
            ),
        )(func)

    return decorator


def validate_output(
    output: dict[str, Any], required_fields: list, field_types: dict[str, type] | None = None
) -> bool:
    """
    Validate output contains required fields and correct types

    Args:
        output: Output dictionary
        required_fields: List of required fields
        field_types: Field type dictionary (optional)

    Returns:
        bool: Whether valid

    Raises:
        LLMParseError: Raised when validation fails
    """
    # Check required fields
    missing_fields = [field for field in required_fields if field not in output]

    if missing_fields:
        raise LLMParseError(f"Missing required fields: {', '.join(missing_fields)}")

    # Check field types
    if field_types:
        for field, expected_type in field_types.items():
            if field in output and not isinstance(output[field], expected_type):
                actual_type = type(output[field]).__name__
                expected_type_name = expected_type.__name__
                raise LLMParseError(
                    f"Field '{field}' type error: expected {expected_type_name}, got {actual_type}"
                )

    return True


def safe_parse(
    text: str, parser_func: Callable, default: Any = None, raise_on_error: bool = False
) -> Any:
    """
    Safe parsing (catch exceptions and return default value)

    Args:
        parser_func: Parser function
        text: Text to parse
        default: Default value
        raise_on_error: Whether to raise exception on error

    Returns:
        Parsed result or default value
    """
    try:
        return parser_func(text)
    except Exception as e:
        if raise_on_error:
            raise LLMParseError(f"Parsing failed: {e!s}") from e

        logger.error(
            f"Parsing failed; falling back to default value {default!r}. This may affect behavior. Error: {e!s}",
            exc_info=True,
        )
        return default


def validate_investigate_output(
    output: dict[str, Any], valid_tools: list[str] = VALID_INVESTIGATE_TOOLS
) -> bool:
    """Validate InvestigateAgent output using Pydantic model"""
    # Check if custom tools are provided
    if valid_tools != VALID_INVESTIGATE_TOOLS:
        # For custom tools, do manual validation
        validate_output(output, ["reasoning"], {"reasoning": str})
        tools = output.get("tools", [])
        if not isinstance(tools, list) or len(tools) < 1:
            raise LLMParseError("tools must be a non-empty list")

        for i, tool in enumerate(tools):
            if not isinstance(tool, dict):
                raise LLMParseError(f"tool[{i}] must be a dictionary")
            tool_type = tool.get("tool_type", "").lower()
            if tool_type not in valid_tools:
                raise LLMParseError(
                    f"tool[{i}] tool_type must be one of {valid_tools}, got: {tool_type}"
                )
            if tool_type != "none" and not tool.get("query"):
                raise LLMParseError(f"tool[{i}] missing query")

        # Check none tool exclusivity
        has_none = any(t.get("tool_type", "").lower() == "none" for t in tools)
        if has_none and len(tools) > 1:
            raise LLMParseError("When 'none' tool exists, no other tool intents should be provided")
        return True

    # Use Pydantic for standard validation
    try:
        InvestigateOutput(**output)
        return True
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise LLMParseError(f"InvestigateAgent output validation failed: {error_details}") from e


def validate_note_output(output: dict[str, Any]) -> bool:
    """Validate NoteAgent output using Pydantic model"""
    try:
        NoteOutput(**output)
        return True
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise LLMParseError(f"NoteAgent output validation failed: {error_details}") from e


def validate_reflect_output(output: dict[str, Any]) -> bool:
    """Validate InvestigateReflectAgent output using Pydantic model"""
    try:
        ReflectOutput(**output)
        return True
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise LLMParseError(
            f"InvestigateReflectAgent output validation failed: {error_details}"
        ) from e


def validate_plan_output(output: dict[str, Any]) -> bool:
    """Validate PlanAgent output using Pydantic model"""
    try:
        PlanOutput(**output)
        return True
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise LLMParseError(f"PlanAgent output validation failed: {error_details}") from e


def validate_solve_output(
    output: dict[str, Any], valid_tool_types: list[str] = VALID_SOLVE_TOOLS
) -> bool:
    """Validate SolveAgent output using Pydantic model"""
    # Check if custom tools are provided
    if valid_tool_types != VALID_SOLVE_TOOLS:
        # For custom tools, do manual validation
        validate_output(output, ["tool_calls"], {"tool_calls": list})
        tool_calls = output.get("tool_calls", [])
        if not isinstance(tool_calls, list) or len(tool_calls) < 1:
            raise LLMParseError("tool_calls must be a non-empty list")

        for i, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                raise LLMParseError(f"tool_call[{i}] must be a dictionary")
            if "tool_type" not in tool_call or "query" not in tool_call:
                raise LLMParseError(f"tool_call[{i}] missing required fields: tool_type, query")
            tool_type = tool_call.get("tool_type", "").lower()
            if tool_type not in valid_tool_types:
                raise LLMParseError(
                    f"Invalid tool_type: {tool_type}, must be one of {valid_tool_types}"
                )
        return True

    # Use Pydantic for standard validation
    try:
        SolveOutput(**output)
        return True
    except ValidationError as e:
        error_details = _format_validation_errors(e)
        raise LLMParseError(f"SolveAgent output validation failed: {error_details}") from e


def validate_none_tool_constraint(
    tools: list[dict[str, Any]], tool_type_key: str = "tool_type"
) -> None:
    """
    Validate that 'none' tool does not coexist with other tools.

    Args:
        tools: List of tool dictionaries
        tool_type_key: Key to access tool type in each dict (default: "tool_type")

    Raises:
        LLMParseError: If none tool constraint is violated
    """
    has_none = any(
        isinstance(tool_type := tool.get(tool_type_key), str) and tool_type.lower() == "none"
        for tool in tools
    )

    if has_none and len(tools) > 1:
        raise LLMParseError(
            f"When 'none' tool exists, no other tool intents should be provided. "
            f"Found {len(tools)} tools with types: {[tool.get(tool_type_key) for tool in tools]}"
        )
