"""Topic-focused summarization using LLM."""

import json
import logging
import re
from typing import Any, Dict, Optional, List
from pydantic import BaseModel

from utils.llm_provider import llm_generate

logger = logging.getLogger(__name__)


class PaperSummarySchema(BaseModel):
    """Schema for structured paper summary."""
    main_contributions: List[str]  # Key findings and contributions
    topic_relevance: str  # How the paper relates to the research topic
    key_techniques: List[str]  # Methods, datasets, or approaches used
    potential_applications: str  # What this could contribute to the research
    limitations: Optional[List[str]] = None  # Gaps or limitations mentioned


class Summarizer:
    """Generate topic-focused summaries of papers."""

    def __init__(
        self, model: Optional[str] = None, temperature: float = 0.3, max_summary_length: int = 800
    ):
        """
        Initialize summarizer.

        Args:
            model: LLM model name (None = use provider default)
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
        Generate topic-focused summary of paper using structured schema.

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
            # Build prompt with schema
            prompt = self._build_summary_prompt(title, abstract, intro, topic, metadata, full_text)

            # Call LLM with schema in options (provider-agnostic)
            options = {
                "temperature": self.temperature,
                "schema": PaperSummarySchema
            }
            
            response = llm_generate(
                prompt=prompt,
                model=self.model,
                options=options,
            )
            
            # Parse response - check if it's already structured JSON or needs parsing
            response_data = response["response"]
            structured_summary: dict[Any, Any] | None = None
            if isinstance(response_data, dict):
                # Provider returned structured data (e.g., Gemini with native schema support)
                structured_summary = response_data
            else:
                # Provider returned text - parse JSON (e.g., Ollama)
                structured_summary = self._parse_json_response(str(response_data))
            
            # Validate and format
            if structured_summary:
                logger.info(f"Generated structured summary for '{title}'")
                return self._format_structured_summary(structured_summary, metadata)
            else:
                logger.warning(f"Failed to parse structured summary for '{title}', using fallback")
                return self._fallback_summary(title, abstract, metadata)
                
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
        """Build summarization prompt requesting structured JSON output."""
        # Use only abstract for focused, efficient summarization
        if abstract:
            content_str = f"Abstract: {abstract}"
        else:
            content_str = "Abstract not available"

        return f"""Analyze this research paper's abstract in relation to the research topic: "{topic}"

Paper Title: {title}

{content_str}

Extract the following information and return ONLY a valid JSON object with these exact fields:

{{
  "main_contributions": ["list of 2-4 key contributions or findings"],
  "topic_relevance": "2-3 sentences explaining how this paper directly relates to '{topic}'",
  "key_techniques": ["list of 2-4 methods, datasets, frameworks, or approaches mentioned"],
  "potential_applications": "2-3 sentences describing what this paper could contribute to research on '{topic}' and any leverageable insights",
  "limitations": ["list of 1-3 limitations or gaps if mentioned, or empty array if none"]
}}

Be concise and specific. Focus on concrete information from the abstract.
Return ONLY the JSON object, no other text."""

    def _parse_json_response(self, response_text: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response text.
        
        Handles common issues like markdown code blocks, extra text, etc.
        """
        try:
            # Remove markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            # Find JSON object using regex
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
                data = json.loads(json_str)
                
                # Validate it's a dict and has required fields
                if not isinstance(data, dict):
                    logger.warning("Parsed JSON is not a dictionary")
                    return None
                    
                required_fields = ["main_contributions", "topic_relevance", "key_techniques", "potential_applications"]
                if all(field in data for field in required_fields):
                    return data
                else:
                    logger.warning(f"JSON missing required fields. Got: {list(data.keys())}")
                    return None
            else:
                logger.warning("No JSON object found in response")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing response: {e}")
            return None
    
    def _format_structured_summary(self, summary_data: Dict, metadata: Dict) -> str:
        """Format structured summary data as readable markdown."""
        authors = ", ".join(metadata.get("authors", ["Unknown"]))
        year = metadata.get("year", "n.d.")
        venue = metadata.get("venue", "")

        formatted = f"**{metadata.get('title', '')}**\n\n"
        formatted += f"*{authors} ({year})*"

        if venue:
            formatted += f" - *{venue}*"

        formatted += "\n\n"
        
        # Main contributions
        if summary_data.get("main_contributions"):
            formatted += "**Key Contributions:**\n"
            for contrib in summary_data["main_contributions"]:
                formatted += f"- {contrib}\n"
            formatted += "\n"
        
        # Topic relevance
        if summary_data.get("topic_relevance"):
            formatted += "**Relevance to Topic:**\n"
            formatted += f"{summary_data['topic_relevance']}\n\n"
        
        # Techniques and methods
        if summary_data.get("key_techniques"):
            formatted += "**Key Techniques/Methods:**\n"
            for technique in summary_data["key_techniques"]:
                formatted += f"- {technique}\n"
            formatted += "\n"
        
        # Potential applications
        if summary_data.get("potential_applications"):
            formatted += "**Potential Applications:**\n"
            formatted += f"{summary_data['potential_applications']}\n\n"
        
        # Limitations
        limitations = summary_data.get("limitations")
        if limitations and len(limitations) > 0:
            formatted += "**Limitations:**\n"
            for limitation in limitations:
                formatted += f"- {limitation}\n"
            formatted += "\n"

        return formatted
    
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
