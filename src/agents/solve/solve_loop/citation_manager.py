#!/usr/bin/env python
"""
CitationManager - Citation number manager
Maintains singleton throughout Solve process, ensures citation numbers are globally unique and continuous
"""

from typing import Any, Optional


class CitationManager:
    """Citation number manager"""

    _instance: Optional["CitationManager"] = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize citation manager"""
        if self._initialized:
            return

        self.citation_counter = 0
        self.citation_map: dict[str, dict[str, Any]] = {}  # {citation_id: citation_info}
        self._initialized = True

    def allocate_citation_id(
        self,
        knowledge_id: str,
        reference_id: str | None = None,
        source: str | None = None,
        content: str | None = None,
    ) -> str:
        """
        Allocate citation number

        Args:
            knowledge_id: Knowledge item ID
            reference_id: reference_id from KnowledgeItem.citations (e.g., "[1]")
            source: Citation source
            content: Citation content

        Returns:
            str: Citation number (e.g., "[1]")
        """
        self.citation_counter += 1
        citation_id = f"[{self.citation_counter}]"

        self.citation_map[citation_id] = {
            "knowledge_id": knowledge_id,
            "reference_id": reference_id,  # From KnowledgeItem.citations
            "source": source,
            "content": content,
            "citation_id": citation_id,
        }

        return citation_id

    def get_citation_info(self, citation_id: str) -> dict[str, Any] | None:
        """Get citation information"""
        return self.citation_map.get(citation_id)

    def get_all_citations(self) -> dict[str, dict[str, Any]]:
        """Get all citation mappings"""
        return self.citation_map.copy()

    def reset(self):
        """Reset citation manager (for new task)"""
        self.citation_counter = 0
        self.citation_map = {}
