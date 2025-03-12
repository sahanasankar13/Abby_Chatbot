    import logging
    from typing import List, Dict, Any, Optional
    import re

    logger = logging.getLogger(__name__)

    class Citation:
        """Represents a citation source for information."""
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
            return {
                "source": self.source,
                "url": self.url,
                "title": self.title,
                "authors": self.authors,
                "publication_date": self.publication_date,
                "accessed_date": self.accessed_date
            }

        def to_html(self) -> str:
            html = f'<div class="citation">'
            if self.authors:
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
            md = ""
            if self.authors:
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
        """
        Manages citations for chatbot responses.
        Sources will only be added if the response is based on actual data from an API or RAG system.
        """
        SOURCES = {
            "planned_parenthood": Citation(
                source="Planned Parenthood",
                url="https://www.plannedparenthood.org/",
                title="Planned Parenthood",
                authors=["Planned Parenthood Federation of America"]
            ),
            "abortion_policy_api": Citation(
                source="Abortion Policy API",
                url="https://www.abortionpolicyapi.com/",
                title="Abortion Policy API"
            )
        }

        def __init__(self):
            logger.info("Initializing Citation Manager")
            self.sources = self.SOURCES
            self.default_sources = ["planned_parenthood"]

        def add_citation_to_text(self, text: str, source_id: str, include_citation: bool = True) -> str:
            """
            Append an inline citation (e.g. "(Source: Planned Parenthood)") to the text
            if include_citation is True. Otherwise, return the text unchanged.
            """
            if include_citation and source_id in self.sources:
                citation_text = f"(Source: {self.sources[source_id].source})"
                if citation_text not in text:
                    return f"{text} {citation_text}"
            return text

        def extract_citations_from_text(self, text: str) -> List[Citation]:
            """
            Parse and return a list of Citation objects from inline citation markers in the text.
            """
            citations = []
            # Skip if text is too short or clearly conversational
            if len(text.split()) < 15 or "take care" in text.lower():
                return []
            source_pattern = r'\(Source: ([^)]+)\)'
            source_matches = re.findall(source_pattern, text)
            for source in source_matches:
                for key, citation in self.sources.items():
                    if citation.source == source:
                        citations.append(citation)
            return citations

        def format_response_with_citations(self, text: str, format_type: str = "html") -> Dict[str, Any]:
            """
            Remove inline citation markers and return a formatted response that includes
            both the clean text and a list of formatted citations.
            """
            citations = self.extract_citations_from_text(text)
            # Remove citation markers from text
            clean_text = re.sub(r'\(Source: [^)]+\)', '', text).strip()
            unique_citations = []
            seen_sources = set()
            for citation in citations:
                if citation.source not in seen_sources:
                    unique_citations.append(citation)
                    seen_sources.add(citation.source)
            if format_type == "html":
                formatted_citations = [c.to_html() for c in unique_citations]
            else:
                formatted_citations = [c.to_markdown() for c in unique_citations]
            return {
                "text": clean_text,
                "citations": formatted_citations,
                "citation_objects": [c.to_dict() for c in unique_citations]
            }
