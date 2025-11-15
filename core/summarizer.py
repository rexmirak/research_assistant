"""Topic-focused summarization using LLM."""

import logging
from typing import Dict, Optional

from utils.llm_provider import llm_generate

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate topic-focused summaries of papers."""

    def __init__(
        self, model: str = "deepseek-r1:8b", temperature: float = 0.3, max_summary_length: int = 800
    ):
        """
        Initialize summarizer.

        Args:
            model: Ollama model name
            temperature: Sampling temperature
            max_summary_length: Maximum summary length in words
        """
        self.model = model
        self.temperature = temperature
        self.max_summary_length = max_summary_length

    def summarize_paper(
        self,
        title: str,
        abstract: Optional[str],
        intro: Optional[str],
        topic: str,
        metadata: Dict,
        full_text: Optional[str] = None,
    ) -> str:
        """
        Generate topic-focused summary of paper.

        Args:
            title: Paper title
            abstract: Paper abstract
            intro: Paper introduction
            topic: Research topic
            metadata: Paper metadata (authors, year, venue)
            full_text: Full paper text (if available)

        Returns:
            Markdown-formatted summary
        """
        try:
            # Build prompt
            prompt = self._build_summary_prompt(title, abstract, intro, topic, metadata, full_text)

            # Call provider-agnostic LLM
            response = llm_generate(
                prompt=prompt,
                model=self.model,
                options={"temperature": self.temperature},
            )
            summary = response["response"].strip()
            # Format as markdown
            return self._format_summary(summary, metadata)
        except Exception as e:
            logger.error(f"Summarization failed for {title}: {e}")
            return self._fallback_summary(title, abstract, metadata)

    def _build_summary_prompt(
        self,
        title: str,
        abstract: Optional[str],
        intro: Optional[str],
        topic: str,
        metadata: Dict,
        full_text: Optional[str] = None,
    ) -> str:
        """Build summarization prompt using only abstract to focus on key relations to research topic."""
        # Use only abstract for focused, efficient summarization
        if abstract:
            content_str = f"Abstract: {abstract}"
        else:
            content_str = "Abstract not available"

        return f"""Analyze this research paper's abstract and identify its key relations to the following research topic.

Research Topic:
{topic}

Paper Title: {title}

{content_str}

Provide a concise analysis (max {self.max_summary_length} words) covering:
1. Main contributions and key findings from the abstract
2. Direct relevance to the research topic
3. Specific connections or insights that relate to the topic
4. What this paper could contribute to research on "{topic}"
5. Leveragable information: techniques, methodologies, datasets, or findings that could be applied to advance research on the topic
6. Potential gaps or limitations mentioned

Format as bullet points for easy scanning.
"""

    def _format_summary(self, summary: str, metadata: Dict) -> str:
        """Format summary as markdown."""
        authors = ", ".join(metadata.get("authors", ["Unknown"]))
        year = metadata.get("year", "n.d.")
        venue = metadata.get("venue", "")

        formatted = f"**{metadata.get('title', '')}**\n\n"
        formatted += f"*{authors} ({year})*"

        if venue:
            formatted += f" - *{venue}*"

        formatted += "\n\n"
        formatted += summary
        formatted += "\n\n"

        return formatted

    def _fallback_summary(self, title: str, abstract: Optional[str], metadata: Dict) -> str:
        """Generate fallback summary when LLM fails."""
        authors = ", ".join(metadata.get("authors", ["Unknown"]))
        year = metadata.get("year", "n.d.")

        summary = f"**{title}**\n\n"
        summary += f"*{authors} ({year})*\n\n"

        if abstract:
            summary += f"{abstract[:500]}...\n\n"
        else:
            summary += "*Summary generation failed; no abstract available.*\n\n"

        return summary
