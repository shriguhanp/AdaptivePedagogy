"""
Paper Search Tool - ArXiv paper search tool

Features:
1. Search ArXiv papers
2. Parse paper metadata
3. Format paper information

Author: DeepTutor Team
Version: v1.0
Based on: TODO.md specification
"""

import asyncio
from datetime import datetime
import re

import arxiv


class PaperSearchTool:
    """ArXiv paper search tool"""

    def __init__(self):
        """Initialize search tool"""
        self.client = arxiv.Client()

    async def search_papers(
        self,
        query: str,
        max_results: int = 3,
        years_limit: int | None = 3,
        sort_by: str = "relevance",
    ) -> list[dict]:
        """
        Search ArXiv papers

        Args:
            query: Search query keywords
            max_results: Number of papers to return
            years_limit: Paper year limit (last N years), None means no limit
            sort_by: Sort method - "relevance" or "date"

        Returns:
            List of papers, each paper contains:
                - title: Title
                - authors: Author list
                - year: Publication year
                - abstract: Abstract
                - url: Paper URL
                - arxiv_id: ArXiv ID
                - published: Publication date (ISO format)
        """
        # Determine sort method
        if sort_by == "date":
            sort_criterion = arxiv.SortCriterion.SubmittedDate
        else:
            sort_criterion = arxiv.SortCriterion.Relevance

        # Build search object
        search = arxiv.Search(
            query=query,
            max_results=max_results * 3,  # Search more for filtering
            sort_by=sort_criterion,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        current_year = datetime.now().year

        # Execute search asynchronously (arxiv library is synchronous, but we can run in executor)
        results = list(self.client.results(search))

        for result in results:
            # Extract year
            published_date = result.published
            paper_year = published_date.year

            # Year filtering
            if years_limit and (current_year - paper_year) > years_limit:
                continue

            # Extract ArXiv ID
            arxiv_id = result.entry_id.split("/")[-1]
            if "v" in arxiv_id:
                arxiv_id = arxiv_id.split("v")[0]  # Remove version number

            # Extract authors
            authors = [author.name for author in result.authors]

            # Build paper information
            paper_info = {
                "title": result.title,
                "authors": authors,
                "year": paper_year,
                "abstract": result.summary,
                "url": result.entry_id,
                "arxiv_id": arxiv_id,
                "published": published_date.isoformat(),
            }

            papers.append(paper_info)

            # If enough collected, stop
            if len(papers) >= max_results:
                break

        return papers

    def format_paper_citation(self, paper: dict) -> str:
        """
        Format paper citation

        Args:
            paper: Paper information dictionary

        Returns:
            Citation string: (FirstAuthor et al., Year)
        """
        if not paper["authors"]:
            return f"(Unknown, {paper['year']})"

        first_author = paper["authors"][0].split()[-1]  # Extract surname

        if len(paper["authors"]) > 1:
            return f"({first_author} et al., {paper['year']})"
        return f"({first_author}, {paper['year']})"

    def extract_arxiv_id_from_url(self, url: str) -> str | None:
        """
        Extract ArXiv ID from URL

        Args:
            url: ArXiv URL

        Returns:
            ArXiv ID or None
        """
        match = re.search(r"arxiv\.org/(?:abs|pdf)/(\d+\.\d+)", url)
        if match:
            return match.group(1)
        return None


# ========== Usage Example ==========


async def main():
    """Test function"""
    tool = PaperSearchTool()

    # Test search
    print("Search: transformer attention mechanism")
    papers = await tool.search_papers(
        query="transformer attention mechanism", max_results=3, years_limit=3, sort_by="relevance"
    )

    print(f"\nFound {len(papers)} papers:\n")

    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper['title']}")
        print(f"   Authors: {', '.join(paper['authors'][:3])}")
        print(f"   Year: {paper['year']}")
        print(f"   Citation: {tool.format_paper_citation(paper)}")
        print(f"   URL: {paper['url']}")
        print(f"   ArXiv ID: {paper['arxiv_id']}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
