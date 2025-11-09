"""Topic-focused summarization using LLM."""

import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class Summarizer:
    """Generate topic-focused summaries of papers."""

    def __init__(
        self, model: str = "llama3.1:8b", temperature: float = 0.3, max_summary_length: int = 800
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
        self, title: str, abstract: Optional[str], intro: Optional[str], topic: str, metadata: Dict
    ) -> str:
        """
        Generate topic-focused summary of paper.

        Args:
            title: Paper title
            abstract: Paper abstract
            intro: Paper introduction
            topic: Research topic
            metadata: Paper metadata (authors, year, venue)

        Returns:
            Markdown-formatted summary
        """
        try:
            import ollama

            # Build prompt
            prompt = self._build_summary_prompt(title, abstract, intro, topic, metadata)

            # Call LLM
            response = ollama.generate(
                model=self.model, prompt=prompt, options={"temperature": self.temperature}
            )

            summary = response["response"].strip()

            # Format as markdown
            return self._format_summary(summary, metadata)

        except Exception as e:
            logger.error(f"Summarization failed for {title}: {e}")
            return self._fallback_summary(title, abstract, metadata)

    def _build_summary_prompt(
        self, title: str, abstract: Optional[str], intro: Optional[str], topic: str, metadata: Dict
    ) -> str:
        """Build summarization prompt."""
        content = []

        if abstract:
            content.append(f"Abstract: {abstract}")

        if intro:
            content.append(f"Introduction: {intro[:2000]}")

        content_str = "\n\n".join(content) if content else "Content not available"

        return f"""Summarize this research paper with a focus on its relevance to the following research topic.

Research Topic:
{topic}

Paper Title: {title}

{content_str}

Provide a concise summary (max {self.max_summary_length} words) covering:
1. Main contributions and key findings
2. Methods or approach used
3. Specific points/insights relevant to the research topic
4. How this work could help or inform research on the topic
5. Any notable limitations

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
