#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SolveMemory - Solve-chain based solving memory system
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


def _now() -> str:
    return datetime.utcnow().isoformat()


@dataclass
class ToolCallRecord:
    """Single tool call record"""

    tool_type: str
    query: str
    cite_id: Optional[str] = None
    raw_answer: Optional[str] = None
    summary: Optional[str] = None
    status: str = "pending"  # pending | running | success | failed | none | finish
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    call_id: str = field(default_factory=lambda: f"tc_{uuid.uuid4().hex[:8]}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCallRecord":
        data.setdefault("metadata", {})
        data.setdefault("status", "pending")
        data.setdefault("created_at", _now())
        data.setdefault("updated_at", data["created_at"])
        data.setdefault("call_id", f"tc_{uuid.uuid4().hex[:8]}")
        return cls(**data)

    def mark_running(self):
        self.status = "running"
        self.updated_at = _now()

    def mark_result(
        self,
        raw_answer: str,
        summary: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.raw_answer = raw_answer
        self.summary = summary
        self.status = status
        if metadata:
            self.metadata.update(metadata)
        self.updated_at = _now()


@dataclass
class SolveChainStep:
    """Single step structure in solve-chain"""

    step_id: str
    step_target: str
    available_cite: List[str] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    step_response: Optional[str] = None
    status: str = "undone"  # undone | in_progress | waiting_response | done | failed
    used_citations: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SolveChainStep":
        tool_calls = [ToolCallRecord.from_dict(tc) for tc in data.get("tool_calls", [])]
        data.setdefault("available_cite", [])
        data.setdefault("used_citations", [])
        data.setdefault("status", "undone")
        data.setdefault("step_response", None)
        data.setdefault("created_at", _now())
        data.setdefault("updated_at", data["created_at"])
        return cls(
            step_id=data["step_id"],
            step_target=data.get("step_target", data.get("plan", "")),
            available_cite=data["available_cite"],
            tool_calls=tool_calls,
            step_response=data.get("step_response", data.get("content")),
            status=data["status"],
            used_citations=data.get("used_citations", []),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )

    def append_tool_call(self, tool_call: ToolCallRecord):
        self.tool_calls.append(tool_call)
        self.updated_at = _now()
        if self.status == "undone":
            self.status = "in_progress"

    def update_response(self, response: str, used_citations: Optional[List[str]] = None):
        self.step_response = response
        self.status = "done"
        self.used_citations = used_citations or []
        self.updated_at = _now()

    def mark_waiting_response(self):
        self.status = "waiting_response"
        self.updated_at = _now()


class SolveMemory:
    """solve-chain data storage"""

    def __init__(
        self,
        task_id: Optional[str] = None,
        user_question: str = "",
        output_dir: Optional[str] = None,
    ):
        self.task_id = task_id or f"solve_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.user_question = user_question
        self.output_dir = output_dir

        self.version = "solve_chain_v1"
        self.created_at = _now()
        self.updated_at = _now()

        self.solve_chains: List[SolveChainStep] = []

        self.metadata: Dict[str, Any] = {
            "total_steps": 0,
            "completed_steps": 0,
            "total_tool_calls": 0,
        }

        self.file_path = Path(output_dir) / "solve_chain.json" if output_dir else None

    # ------------------------------------------------------------------ #
    # Load/Save
    # ------------------------------------------------------------------ #
    @classmethod
    def load_or_create(
        cls,
        output_dir: str,
        user_question: str = "",
        task_id: Optional[str] = None,
    ) -> "SolveMemory":
        file_path = Path(output_dir) / "solve_chain.json"
        legacy_path = Path(output_dir) / "solve_memory.json"
        if not file_path.exists() and legacy_path.exists():
            memory = cls(task_id=task_id, user_question=user_question, output_dir=output_dir)
            memory._load_from_legacy_file(legacy_path)
            memory.save()
            return memory
        if not file_path.exists():
            return cls(task_id=task_id, user_question=user_question, output_dir=output_dir)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls(
            task_id=data.get("task_id", task_id),
            user_question=data.get("user_question", user_question),
            output_dir=output_dir,
        )

        memory.version = data.get("version", "solve_chain_v1")
        memory.created_at = data.get("created_at", memory.created_at)
        memory.updated_at = data.get("updated_at", memory.updated_at)
        memory.metadata = data.get("metadata", memory.metadata)
        memory.solve_chains = [
            SolveChainStep.from_dict(step) for step in data.get("solve_chains", [])
        ]

        return memory

    def save(self):
        if not self.file_path:
            raise ValueError("output_dir not set, cannot save solve-chain")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        payload = self.to_dict()
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "task_id": self.task_id,
            "user_question": self.user_question,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "solve_chains": [step.to_dict() for step in self.solve_chains],
            "metadata": self.metadata,
        }

    # ------------------------------------------------------------------ #
    # Step Management
    # ------------------------------------------------------------------ #
    def create_chains(self, chains: List[SolveChainStep]):
        self.solve_chains = chains
        self.metadata["total_steps"] = len(chains)
        self.metadata["completed_steps"] = sum(1 for c in chains if c.status == "done")
        self.metadata["total_tool_calls"] = sum(len(c.tool_calls) for c in chains)
        self.updated_at = _now()

    def get_step(self, step_id: str) -> Optional[SolveChainStep]:
        return next((step for step in self.solve_chains if step.step_id == step_id), None)

    def get_current_step(self) -> Optional[SolveChainStep]:
        for step in self.solve_chains:
            if step.status in {"undone", "in_progress", "waiting_response"}:
                return step
        return None

    def append_tool_call(
        self,
        step_id: str,
        tool_type: str,
        query: str,
        cite_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolCallRecord:
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        record = ToolCallRecord(
            tool_type=tool_type,
            query=query,
            cite_id=cite_id,
            metadata=metadata or {},
        )
        step.append_tool_call(record)
        self.metadata["total_tool_calls"] += 1
        self.updated_at = _now()
        return record

    def update_tool_call_result(
        self,
        step_id: str,
        call_id: str,
        raw_answer: str,
        summary: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        record = next((tc for tc in step.tool_calls if tc.call_id == call_id), None)
        if not record:
            raise ValueError(f"Tool call {call_id} not found in step {step_id}")
        record.mark_result(raw_answer=raw_answer, summary=summary, status=status, metadata=metadata)
        self.updated_at = _now()

    def mark_step_waiting_response(self, step_id: str):
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        step.mark_waiting_response()
        self.updated_at = _now()

    def submit_step_response(
        self,
        step_id: str,
        response: str,
        used_citations: Optional[List[str]] = None,
    ):
        step = self.get_step(step_id)
        if not step:
            raise ValueError(f"Step {step_id} not found")
        step.update_response(response=response, used_citations=used_citations or [])
        self.metadata["completed_steps"] = sum(1 for c in self.solve_chains if c.status == "done")
        self.updated_at = _now()

    def get_summary(self) -> str:
        lines = [
            f"Task ID: {self.task_id}",
            f"Question: {self.user_question}",
            f"Total Steps: {self.metadata['total_steps']}",
            f"Completed: {self.metadata['completed_steps']}",
        ]
        for step in self.solve_chains:
            lines.append(f"- {step.step_id} | {step.status} | target: {step.step_target[:60]}...")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Legacy support
    # ------------------------------------------------------------------ #
    def _load_from_legacy_file(self, legacy_path: Path):
        try:
            with open(legacy_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        steps_data = data.get("steps", [])
        converted: List[SolveChainStep] = []
        for idx, item in enumerate(steps_data, start=1):
            tool_logs = item.get("tool_logs", [])
            records: List[ToolCallRecord] = []
            for log in tool_logs:
                records.append(
                    ToolCallRecord(
                        tool_type=log.get("tool", "unknown"),
                        query=log.get("input") or log.get("query") or "",
                        cite_id=log.get("cite_id"),
                        raw_answer=log.get("output"),
                        summary=log.get("output"),
                        status=log.get("status", "success"),
                    )
                )
            converted.append(
                SolveChainStep(
                    step_id=item.get("step_id", f"S{idx}"),
                    step_target=item.get("plan", ""),
                    available_cite=item.get("available_citations", []),
                    tool_calls=records,
                    step_response=item.get("content"),
                    status="done" if item.get("status") == "completed" else "undone",
                    used_citations=item.get("used_citations", []),
                )
            )

        self.solve_chains = converted
        self.metadata["total_steps"] = len(converted)
        self.metadata["completed_steps"] = sum(1 for step in converted if step.status == "done")
