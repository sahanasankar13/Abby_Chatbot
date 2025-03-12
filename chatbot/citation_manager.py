import logging
from typing import List, Dict, Any, Optional
import re
import json

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
    
    # Standard citation sources
    SOURCES = {
        "planned_parenthood": Citation(
            source="Planned Parenthood",
            url="https://www.plannedparenthood.org/",
            title="Planned Parenthood",
            authors=["Planned Parenthood Federation of America"],
        ),
        "acog": Citation(
            source="American College of Obstetricians and Gynecologists",
            url="https://www.acog.org/",
            title="ACOG",
            authors=["American College of Obstetricians and Gynecologists"],
        ),
        "cdc": Citation(
            source="Centers for Disease Control and Prevention",
            url="https://www.cdc.gov/",
            title="CDC",
            authors=["Centers for Disease Control and Prevention"],
        ),
        "who": Citation(
            source="World Health Organization",
            url="https://www.who.int/",
            title="WHO",
            authors=["World Health Organization"],
        ),
        "abortion_policy_api": Citation(
            source="Abortion Policy API",
            url="https://www.abortionpolicyapi.com/",
            title="Abortion Policy API",
        ),
        "guttmacher": Citation(
            source="Guttmacher Institute",
            url="https://www.guttmacher.org/",
            title="Guttmacher Institute",
        ),
        "ai_generated": Citation(
            source="AI-generated response",
            title="AI-generated content",
            publication_date="2025",
        )
    }
    
    def __init__(self):
        """Initialize the citation manager"""
        self.default_sources = ["planned_parenthood", "ai_generated"]
    
    def extract_citations_from_text(self, text: str) -> List[Citation]:
        """
        Extract citations from text containing citation markers
        
        Args:
            text (str): Text with citation markers like [SOURCE:planned_parenthood]
            
        Returns:
            List[Citation]: List of extracted citations
        """
        citations = []
        citation_pattern = r'\[SOURCE:([\w_]+)\]'
        
        # Find all citation markers
        matches = re.findall(citation_pattern, text)
        
        # Get unique source identifiers
        source_ids = set(matches)
        
        # Get citation objects for each source
        for source_id in source_ids:
            if source_id in self.SOURCES:
                citations.append(self.SOURCES[source_id])
            else:
                logger.warning(f"Unknown citation source: {source_id}")
                
        # If no citations found, use default sources
        if not citations:
            for source_id in self.default_sources:
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
        
        # Format citations based on format type
        if format_type == "html":
            formatted_citations = [c.to_html() for c in citations]
        else:  # markdown
            formatted_citations = [c.to_markdown() for c in citations]
            
        return {
            "text": clean_text.strip(),
            "citations": formatted_citations,
            "citation_objects": [c.to_dict() for c in citations]
        }
    
    def add_citation_to_text(self, text: str, source_id: str) -> str:
        """
        Add a citation marker to text
        
        Args:
            text (str): Original text
            source_id (str): Source identifier
            
        Returns:
            str: Text with citation marker added
        """
        if source_id in self.SOURCES:
            return f"{text} [SOURCE:{source_id}]"
        else:
            logger.warning(f"Unknown citation source: {source_id}")
            return text