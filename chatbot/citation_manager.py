import logging
from typing import List, Dict, Any, Optional
import re
import json
import webbrowser

logger = logging.getLogger(__name__)

class Citation:
    """Represents a citation source for information"""
    def __init__(self, source: str, url: Optional[str] = None, title: Optional[str] = None, 
                 authors: Optional[List[str]] = None, publication_date: Optional[str] = None,
                 accessed_date: Optional[str] = None):
        self.source = source
        self.url = url
        self.title = title
        self.authors = authors or []
        self.publication_date = publication_date
        self.accessed_date = accessed_date

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary for JSON serialization"""
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date,
            "accessed_date": self.accessed_date
        }

    def to_html(self) -> str:
        """Format citation as HTML"""
        html = f'<div class="citation">'

        if self.authors and len(self.authors) > 0:
            if len(self.authors) == 1:
                html += f"{self.authors[0]}. "
            elif len(self.authors) == 2:
                html += f"{self.authors[0]} & {self.authors[1]}. "
            else:
                html += f"{self.authors[0]} et al. "

        if self.publication_date:
            html += f"({self.publication_date}). "

        if self.title:
            if self.url:
                html += f'<a href="{self.url}" target="_blank" rel="noopener noreferrer">{self.title}</a>. '
            else:
                html += f"{self.title}. "

        html += f'<span class="citation-source">{self.source}</span>'

        if self.accessed_date:
            html += f'. Accessed on {self.accessed_date}'

        html += '</div>'
        return html

    def to_markdown(self) -> str:
        """Format citation as Markdown"""
        md = ""

        if self.authors and len(self.authors) > 0:
            if len(self.authors) == 1:
                md += f"{self.authors[0]}. "
            elif len(self.authors) == 2:
                md += f"{self.authors[0]} & {self.authors[1]}. "
            else:
                md += f"{self.authors[0]} et al. "

        if self.publication_date:
            md += f"({self.publication_date}). "

        if self.title:
            if self.url:
                md += f"[{self.title}]({self.url}). "
            else:
                md += f"{self.title}. "

        md += f"*{self.source}*"

        if self.accessed_date:
            md += f". Accessed on {self.accessed_date}"

        return md


class CitationManager:
    """Manages citations for chatbot responses"""

    # Standard citation sources - limited to only Abortion Policy API and Planned Parenthood links
    SOURCES = {
        "planned_parenthood": Citation(
            source="Planned Parenthood",
            url="https://www.plannedparenthood.org/",
            title="Planned Parenthood",
            authors=["Planned Parenthood Federation of America"],
        ),
        "abortion_policy_api": Citation(
            source="Abortion Policy API",
            url="https://www.abortionpolicyapi.com/",
            title="Abortion Policy API",
        )
    }

    def __init__(self):
        """Initialize the citation manager"""
        logger.info("Initializing Citation Manager")
        self.sources = self.SOURCES
        self.default_sources = ["planned_parenthood"]


    def extract_citations_from_text(self, text: str) -> List[Citation]:
        """
        Extract citations from text and return structured citation data.

        Args:
            text (str): Text with citation markers

        Returns:
            List[Citation]: List of extracted citations
        """
        citations = []

        # Skip citation extraction for short conversational responses
        if len(text.split()) < 15 or "take care" in text.lower():
            return []

        # Check for explicit [SOURCE:source_id] pattern (our primary citation method)
        source_pattern = r'\[SOURCE:([\w_]+)\]'
        source_matches = re.findall(source_pattern, text)

        for source_id in source_matches:
            if source_id in self.SOURCES:
                citations.append(self.SOURCES[source_id])

        # Also check for older [cite:] pattern for backwards compatibility
        if "[cite:" in text:
            pattern = r'\[cite:(.*?)\]'
            citation_matches = re.findall(pattern, text)

            for citation in citation_matches:
                citation_parts = citation.split('|')
                if len(citation_parts) >= 1:
                    source_id = citation_parts[0].strip()

                    if source_id in self.SOURCES:
                        citations.append(self.SOURCES[source_id])

        # Also check for API citations
        api_pattern = r'\[API:(.*?)\]'
        api_matches = re.findall(api_pattern, text)

        for api_citation in api_matches:
            if "abortion_policy_api" in self.SOURCES:
                citations.append(self.SOURCES["abortion_policy_api"])

        # If no explicit citations found but text contains certain keywords,
        # add appropriate citation sources
        if len(citations) == 0:
            text_lower = text.lower()

            # Don't add the abortion policy API citation if the text 
            # indicates we had trouble accessing the API data
            if "having trouble" in text_lower or "couldn't retrieve" in text_lower:
                if "planned_parenthood" in self.SOURCES:
                    citations.append(self.SOURCES["planned_parenthood"])
            # For abortion policy related information
            elif any(term in text_lower for term in ["abortion", "policy", "legal", "law", "state", "legislation"]):
                if "abortion_policy_api" in self.SOURCES:
                    citations.append(self.SOURCES["abortion_policy_api"])

            # For general reproductive health information
            if any(term in text_lower for term in ["health", "pregnancy", "birth control", "contraception", "menstrual"]):
                if "planned_parenthood" in self.SOURCES:
                    citations.append(self.SOURCES["planned_parenthood"])

        # If still no citations, add default sources as fallback
        if len(citations) == 0 and hasattr(self, 'default_sources'):
            for source_id in self.default_sources:
                if source_id in self.SOURCES:
                    citations.append(self.SOURCES[source_id])

        return citations

    def format_response_with_citations(self, text: str, format_type: str = "html") -> Dict[str, Any]:
        """
        Format a response with citations, removing citation markers

        Args:
            text (str): Text with citation markers
            format_type (str): Output format ('html' or 'markdown')

        Returns:
            Dict[str, Any]: Dictionary with text and formatted citations
        """
        # Extract citations
        citations = self.extract_citations_from_text(text)

        # Remove citation markers from text
        clean_text = re.sub(r'\[SOURCE:[\w_]+\]', '', text)
        clean_text = re.sub(r'\[cite:.*?\]', '', clean_text)
        clean_text = re.sub(r'\[API:.*?\]', '', clean_text)

        # Make sure we don't have duplicate sources
        unique_citations = []
        seen_sources = set()
        for citation in citations:
            if citation.source not in seen_sources:
                unique_citations.append(citation)
                seen_sources.add(citation.source)

        citations = unique_citations

        # Format citations based on format type - don't include direct source text in answer
        if format_type == "html":
            formatted_citations = [c.to_html() for c in citations]
        else:  # markdown
            formatted_citations = [c.to_markdown() for c in citations]

        logger.debug(f"Formatting response with {len(citations)} citations")

        return {
            "text": clean_text.strip(),
            "citations": formatted_citations,
            "citation_objects": [c.to_dict() for c in citations]
        }

    def add_citation_to_text(self, text: str, source_id: Optional[str] = None, include_citations: bool = True, link: Optional[str] = None) -> str:
        """
        Add citation to text if needed, but only for Planned Parenthood or Abortion Policy API.

        Args:
            text (str): Text to add citation to
            source_id (str, optional): Explicit source ID to use
            include_citations (bool): Whether to include citations in the output
            link (str, optional): Link to use for Planned Parenthood citation

        Returns:
            str: Text with or without citations
        """
        try:
            # If citations are disabled, just return the clean text
            if not include_citations:
                text = self._remove_citation_markers(text)
                return text
                
            # If text already has citation markers, don't add more
            if "[SOURCE:" in text or "[cite:" in text or "[API:" in text:
                logger.debug(f"Text already has citation markers, extracting citations")
                citations = self.extract_citations_from_text(text)

                # Remove any citation markers from the text for clean display
                text = self._remove_citation_markers(text)

                return self._format_text_with_citations(text, citations, include_citations)

            # If explicit source ID provided, only use it if it's one of our approved sources
            if source_id and source_id in self.SOURCES:
                if source_id in ["planned_parenthood", "abortion_policy_api"]:
                    logger.debug(f"Using explicit source ID: {source_id}")
                    # Update link if provided for planned parenthood
                    if source_id == "planned_parenthood" and link:
                        # Create a copy of the citation to avoid modifying the original
                        citation = Citation(
                            source=self.SOURCES[source_id].source,
                            url=link,
                            title=self.SOURCES[source_id].title,
                            authors=self.SOURCES[source_id].authors,
                            publication_date=self.SOURCES[source_id].publication_date,
                            accessed_date=self.SOURCES[source_id].accessed_date
                        )
                        return self._format_text_with_citations(text, [citation], include_citations)
                    else:
                        return self._format_text_with_citations(text, [self.SOURCES[source_id]], include_citations)
                else:
                    # Don't add citation for non-approved sources
                    return text

            # Otherwise extract citations from the text based on content
            # This will now only get explicit source mentions due to our changes
            citations = self.extract_citations_from_text(text)

            logger.debug(f"Formatting response with {len(citations)} citations")
            return self._format_text_with_citations(text, citations, include_citations)
        except Exception as e:
            logger.error(f"Error adding citation to text: {str(e)}", exc_info=True)
            return text

    def _remove_citation_markers(self, text: str) -> str:
        """Helper function to remove citation markers from text."""
        clean_text = re.sub(r'\[SOURCE:[\w_]+\]', '', text)
        clean_text = re.sub(r'\[cite:.*?\]', '', clean_text)
        clean_text = re.sub(r'\[API:.*?\]', '', clean_text)
        return clean_text

    def _format_text_with_citations(self, text: str, citations: List[Citation], include_citations: bool = True) -> str:
        """
        Helper function to format text with citations.
        
        Args:
            text (str): The text content
            citations (List[Citation]): List of citations
            include_citations (bool): Whether to include citations in the output
            
        Returns:
            str: Formatted text with or without citations
        """
        if not include_citations:
            return text
            
        # Make sure we don't have duplicate sources
        unique_citations = []
        seen_sources = set()
        for citation in citations:
            if citation.source not in seen_sources:
                unique_citations.append(citation)
                seen_sources.add(citation.source)
        
        citations = unique_citations
        
        # Only show sources section if we have citations
        if citations:
            formatted_citations = [c.to_html() for c in citations]
            return f"{text}<br><br><h4>Sources</h4>{''.join(formatted_citations)}"
        return text


def quick_exit():
    webbrowser.open("https://www.google.com")