#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
JSON Utils - JSON parsing and validation utilities
- Robustly extract JSON from LLM text output
- Provide strict structure validation and error messages
"""

import json
import re
from typing import Any, Dict, Iterable, List, Union


def extract_json_from_text(text: str) -> Union[Dict[str, Any], List[Any], None]:
    """
    Extract JSON object or array from text.
    Allows the following formats:
    1) Code blocks wrapped in ```json ...``` or ``` ...```
    2) Any valid JSON object {...} or array [...] found in text
    """
    if not text:
        return None

    # 1) Try to find code blocks first (most reliable if present)
    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    for match in code_block_pattern.finditer(text):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue

    # 2) Iterative finding using JSONDecoder.raw_decode
    # This scans the text for any valid JSON structure
    decoder = json.JSONDecoder()
    pos = 0
    text_len = len(text)
    
    while pos < text_len:
        # Find next opening brace
        next_obj = text.find('{', pos)
        next_arr = text.find('[', pos)
        
        # Determine which comes first
        if next_obj == -1 and next_arr == -1:
            break
        
        if next_obj != -1 and (next_arr == -1 or next_obj < next_arr):
            start = next_obj
        else:
            start = next_arr
            
        try:
            result, index = decoder.raw_decode(text, start)
            return result
        except json.JSONDecodeError:
            # If parsing failed, move past this opening brace and try again
            pos = start + 1
            
    return None


# --------- Strict Validation Utilities ---------


def ensure_json_dict(data: Any, err: str = "Expected JSON object") -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(err)
    return data


def ensure_json_list(data: Any, err: str = "Expected JSON array") -> List[Any]:
    if not isinstance(data, list):
        raise ValueError(err)
    return data


def ensure_keys(data: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing required keys: {', '.join(missing)}")
    return data


def safe_json_loads(text: str, default: Any = None) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


def json_to_text(data: Any, indent: int = 2) -> str:
    return json.dumps(data, ensure_ascii=False, indent=indent)


__all__ = [
    "extract_json_from_text",
    "ensure_json_dict",
    "ensure_json_list",
    "ensure_keys",
    "safe_json_loads",
    "json_to_text",
]
