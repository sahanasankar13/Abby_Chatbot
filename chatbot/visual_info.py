import logging
import os
import json
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

class VisualInfoGraphics:
    """
    Provides visual information graphics for reproductive health topics
    """
    def __init__(self):
        """Initialize the visual information graphics system"""
        logger.info("Initializing Visual Information Graphics")
        self.graphics_library = {
            # Menstrual cycle graphics
            "menstrual_cycle": {
                "title": "Menstrual Cycle Phases",
                "type": "svg",
                "description": "Visual representation of the menstrual cycle phases",
                "content": self._create_menstrual_cycle_svg()
            },
            # Contraception effectiveness graphics
            "contraception_effectiveness": {
                "title": "Contraception Method Effectiveness",
                "type": "svg",
                "description": "Comparison of effectiveness rates for different contraception methods",
                "content": self._create_contraception_effectiveness_svg()
            },
            # Pregnancy stages graphics
            "pregnancy_stages": {
                "title": "Pregnancy Stages by Trimester",
                "type": "svg",
                "description": "Visual timeline of pregnancy development by trimester",
                "content": self._create_pregnancy_stages_svg()
            },
            # Reproductive anatomy graphics
            "reproductive_anatomy_female": {
                "title": "Female Reproductive Anatomy",
                "type": "svg",
                "description": "Diagram of female reproductive anatomy",
                "content": self._create_female_anatomy_svg()
            },
            "reproductive_anatomy_male": {
                "title": "Male Reproductive Anatomy",
                "type": "svg",
                "description": "Diagram of male reproductive anatomy",
                "content": self._create_male_anatomy_svg()
            },
            # STI prevention graphics
            "sti_prevention": {
                "title": "STI Prevention Methods",
                "type": "svg",
                "description": "Visual guide to STI prevention methods",
                "content": self._create_sti_prevention_svg()
            }
        }
        logger.info("Visual Information Graphics initialized successfully")
    
    def get_graphic(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get visual graphic for a specific topic
        
        Args:
            topic (str): Topic to get graphic for
            
        Returns:
            Optional[Dict[str, Any]]: Graphic information or None if not found
        """
        return self.graphics_library.get(topic)
    
    def suggest_graphics(self, text: str) -> List[str]:
        """
        Suggest relevant graphics based on message content
        Focus on timeline-related content only
        
        Args:
            text (str): Message text to analyze
            
        Returns:
            List[str]: List of suggested graphic topics
        """
        suggestions = []
        text_lower = text.lower()
        
        # Only suggest timeline-related graphics
        
        # Menstrual cycle timeline
        if ('timeline' in text_lower or 'phases' in text_lower or 'stages' in text_lower or 'cycle' in text_lower) and \
           ('period' in text_lower or 'menstrual' in text_lower or 'menstruation' in text_lower):
            suggestions.append('menstrual_cycle')
            
        # Pregnancy timeline
        if ('timeline' in text_lower or 'development' in text_lower or 'stages' in text_lower or 'weeks' in text_lower or 'trimesters' in text_lower) and \
           ('pregnancy' in text_lower or 'fetal' in text_lower or 'fetus' in text_lower or 'baby' in text_lower):
            suggestions.append('pregnancy_stages')
            
        return suggestions
    
    def add_graphics_to_response(self, response: Dict[str, Any], message: str) -> Dict[str, Any]:
        """
        Add relevant timeline graphics to a response based on message content
        
        Args:
            response (Dict[str, Any]): Original response dictionary
            message (str): User's message
            
        Returns:
            Dict[str, Any]: Response with graphics added
        """
        # Initialize with empty graphics
        response['graphics'] = []
        
        # Check if user is explicitly asking for a timeline
        combined_text = message.lower() + ' ' + response['text'].lower()
        timeline_keywords = ['timeline', 'stages', 'phases', 'development', 'cycle', 'weeks', 
                          'trimesters', 'progression', 'process']
        
        # Only show graphics if explicitly talking about timelines
        has_timeline_keywords = any(keyword in combined_text for keyword in timeline_keywords)
        
        if not has_timeline_keywords:
            # No timeline-related keywords found, don't show graphics
            return response
            
        # Skip graphics for simple greetings or very short messages
        if len(message.strip().split()) <= 5:
            # Only show graphics for longer, specific questions about timelines
            return response
        
        # Get suggested timeline-related graphics
        suggested_topics = self.suggest_graphics(combined_text)
        
        if not suggested_topics:
            # No relevant timeline graphics found
            return response
            
        graphics = []
        
        # Only show timeline graphics, limit to 1 per response
        for topic in suggested_topics[:1]:
            graphic = self.get_graphic(topic)
            if graphic:
                graphics.append(graphic)
                logger.debug(f"Adding timeline graphic: {topic}")
        
        # Add graphics to response
        response['graphics'] = graphics
        return response
    
    def _create_menstrual_cycle_svg(self) -> str:
        """Create SVG for menstrual cycle phases"""
        svg = """<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
            <style>
                .phase { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }
                .day { font-family: Arial, sans-serif; font-size: 12px; }
                .hormone { font-family: Arial, sans-serif; font-size: 14px; }
                .label { font-family: Arial, sans-serif; font-size: 14px; }
                .axis { stroke: #333; stroke-width: 2; }
                .marker { stroke-width: 2; }
            </style>
            
            <!-- Timeline axis -->
            <line x1="50" y1="200" x2="750" y2="200" class="axis" />
            
            <!-- Phase markers -->
            <rect x="50" y="150" width="175" height="40" fill="#ff9999" opacity="0.7" rx="5" />
            <rect x="225" y="150" width="100" height="40" fill="#ffcc99" opacity="0.7" rx="5" />
            <rect x="325" y="150" width="175" height="40" fill="#99ccff" opacity="0.7" rx="5" />
            <rect x="500" y="150" width="250" height="40" fill="#cc99ff" opacity="0.7" rx="5" />
            
            <!-- Phase labels -->
            <text x="137" y="175" text-anchor="middle" class="phase">Menstruation</text>
            <text x="275" y="175" text-anchor="middle" class="phase">Follicular</text>
            <text x="412" y="175" text-anchor="middle" class="phase">Ovulation</text>
            <text x="625" y="175" text-anchor="middle" class="phase">Luteal</text>
            
            <!-- Day markers -->
            <text x="50" y="220" text-anchor="middle" class="day">Day 1</text>
            <text x="225" y="220" text-anchor="middle" class="day">Day 5</text>
            <text x="325" y="220" text-anchor="middle" class="day">Day 14</text>
            <text x="500" y="220" text-anchor="middle" class="day">Day 16</text>
            <text x="750" y="220" text-anchor="middle" class="day">Day 28</text>
            
            <!-- Hormone curves -->
            <path d="M50,300 Q125,290 225,260 T325,150 T500,260 T750,300" fill="none" stroke="#ff6666" stroke-width="3" />
            <path d="M50,280 Q125,290 225,270 T325,240 T500,150 T750,280" fill="none" stroke="#6666ff" stroke-width="3" />
            
            <!-- Hormone labels -->
            <text x="775" y="300" class="hormone" fill="#ff6666">Estrogen</text>
            <text x="775" y="280" class="hormone" fill="#6666ff">Progesterone</text>
            
            <!-- Event markers -->
            <circle cx="325" cy="200" r="10" fill="#ff3333" />
            <text x="325" y="240" text-anchor="middle" class="label">Ovulation</text>
        </svg>"""
        return svg
    
    def _create_contraception_effectiveness_svg(self) -> str:
        """Create SVG for contraception effectiveness comparison"""
        svg = """<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
                .method { font-family: Arial, sans-serif; font-size: 14px; }
                .percent { font-family: Arial, sans-serif; font-size: 14px; font-weight: bold; }
                .label { font-family: Arial, sans-serif; font-size: 12px; }
                .bar { stroke: #333; stroke-width: 1; }
                .axis { stroke: #333; stroke-width: 2; }
            </style>
            
            <!-- Title -->
            <text x="400" y="40" text-anchor="middle" class="title">Contraception Effectiveness (Perfect Use)</text>
            
            <!-- Axis -->
            <line x1="200" y1="80" x2="200" y2="450" class="axis" />
            <line x1="200" y1="450" x2="700" y2="450" class="axis" />
            
            <!-- Axis labels -->
            <text x="450" y="480" text-anchor="middle" class="label">Effectiveness (%)</text>
            
            <!-- Methods and bars -->
            <text x="180" y="100" text-anchor="end" class="method">Implant</text>
            <rect x="200" y="85" width="495" height="30" fill="#3498db" class="bar" />
            <text x="705" y="105" class="percent">99.95%</text>
            
            <text x="180" y="150" text-anchor="end" class="method">IUD (hormonal)</text>
            <rect x="200" y="135" width="495" height="30" fill="#3498db" class="bar" />
            <text x="705" y="155" class="percent">99.8%</text>
            
            <text x="180" y="200" text-anchor="end" class="method">Sterilization</text>
            <rect x="200" y="185" width="495" height="30" fill="#3498db" class="bar" />
            <text x="705" y="205" class="percent">99.5%</text>
            
            <text x="180" y="250" text-anchor="end" class="method">IUD (copper)</text>
            <rect x="200" y="235" width="495" height="30" fill="#3498db" class="bar" />
            <text x="705" y="255" class="percent">99.2%</text>
            
            <text x="180" y="300" text-anchor="end" class="method">Pill</text>
            <rect x="200" y="285" width="445" height="30" fill="#2ecc71" class="bar" />
            <text x="655" y="305" class="percent">99.7%</text>
            
            <text x="180" y="350" text-anchor="end" class="method">Condom</text>
            <rect x="200" y="335" width="400" height="30" fill="#f1c40f" class="bar" />
            <text x="610" y="355" class="percent">98%</text>
            
            <text x="180" y="400" text-anchor="end" class="method">Withdrawal</text>
            <rect x="200" y="385" width="200" height="30" fill="#e74c3c" class="bar" />
            <text x="410" y="405" class="percent">78%</text>
            
            <!-- Legend -->
            <rect x="250" y="470" width="20" height="10" fill="#3498db" />
            <text x="275" y="480" class="label">Highly effective (>99%)</text>
            
            <rect x="400" y="470" width="20" height="10" fill="#2ecc71" />
            <text x="425" y="480" class="label">Effective (>95%)</text>
            
            <rect x="520" y="470" width="20" height="10" fill="#f1c40f" />
            <text x="545" y="480" class="label">Less effective (>80%)</text>
            
            <rect x="650" y="470" width="20" height="10" fill="#e74c3c" />
            <text x="675" y="480" class="label">Least effective (<80%)</text>
        </svg>"""
        return svg
    
    def _create_pregnancy_stages_svg(self) -> str:
        """Create SVG for pregnancy stages timeline"""
        svg = """<svg viewBox="0 0 800 500" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
                .trimester { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }
                .milestone { font-family: Arial, sans-serif; font-size: 14px; }
                .week { font-family: Arial, sans-serif; font-size: 12px; }
                .note { font-family: Arial, sans-serif; font-size: 14px; font-style: italic; }
                .timeline { stroke: #333; stroke-width: 2; }
            </style>
            
            <!-- Title -->
            <text x="400" y="40" text-anchor="middle" class="title">Pregnancy Development Timeline</text>
            
            <!-- Timeline -->
            <line x1="100" y1="130" x2="700" y2="130" class="timeline" />
            
            <!-- Trimesters -->
            <rect x="100" y="80" width="200" height="100" fill="#ffcccc" opacity="0.5" rx="10" />
            <rect x="300" y="80" width="200" height="100" fill="#ccffcc" opacity="0.5" rx="10" />
            <rect x="500" y="80" width="200" height="100" fill="#ccccff" opacity="0.5" rx="10" />
            
            <text x="200" y="70" text-anchor="middle" class="trimester">First Trimester</text>
            <text x="400" y="70" text-anchor="middle" class="trimester">Second Trimester</text>
            <text x="600" y="70" text-anchor="middle" class="trimester">Third Trimester</text>
            
            <!-- Week markers -->
            <line x1="100" y1="125" x2="100" y2="135" stroke="#333" stroke-width="2" />
            <text x="100" y="150" text-anchor="middle" class="week">Week 1</text>
            
            <line x1="150" y1="125" x2="150" y2="135" stroke="#333" stroke-width="2" />
            <text x="150" y="150" text-anchor="middle" class="week">Week 4</text>
            
            <line x1="200" y1="125" x2="200" y2="135" stroke="#333" stroke-width="2" />
            <text x="200" y="150" text-anchor="middle" class="week">Week 8</text>
            
            <line x1="300" y1="125" x2="300" y2="135" stroke="#333" stroke-width="2" />
            <text x="300" y="150" text-anchor="middle" class="week">Week 13</text>
            
            <line x1="400" y1="125" x2="400" y2="135" stroke="#333" stroke-width="2" />
            <text x="400" y="150" text-anchor="middle" class="week">Week 20</text>
            
            <line x1="500" y1="125" x2="500" y2="135" stroke="#333" stroke-width="2" />
            <text x="500" y="150" text-anchor="middle" class="week">Week 27</text>
            
            <line x1="600" y1="125" x2="600" y2="135" stroke="#333" stroke-width="2" />
            <text x="600" y="150" text-anchor="middle" class="week">Week 34</text>
            
            <line x1="700" y1="125" x2="700" y2="135" stroke="#333" stroke-width="2" />
            <text x="700" y="150" text-anchor="middle" class="week">Week 40</text>
            
            <!-- Milestones -->
            <circle cx="150" cy="130" r="8" fill="#ff6666" />
            <line x1="150" y1="170" x2="150" y2="200" stroke="#666" stroke-dasharray="4" />
            <text x="150" y="215" text-anchor="middle" class="milestone">Heartbeat begins</text>
            
            <circle cx="200" cy="130" r="8" fill="#ff6666" />
            <line x1="200" y1="170" x2="200" y2="250" stroke="#666" stroke-dasharray="4" />
            <text x="200" y="265" text-anchor="middle" class="milestone">All major organs formed</text>
            
            <circle cx="350" cy="130" r="8" fill="#66cc66" />
            <line x1="350" y1="170" x2="350" y2="200" stroke="#666" stroke-dasharray="4" />
            <text x="350" y="215" text-anchor="middle" class="milestone">Movement felt</text>
            
            <circle cx="400" cy="130" r="8" fill="#66cc66" />
            <line x1="400" y1="170" x2="400" y2="250" stroke="#666" stroke-dasharray="4" />
            <text x="400" y="265" text-anchor="middle" class="milestone">Anatomy scan</text>
            
            <circle cx="550" cy="130" r="8" fill="#6666ff" />
            <line x1="550" y1="170" x2="550" y2="200" stroke="#666" stroke-dasharray="4" />
            <text x="550" y="215" text-anchor="middle" class="milestone">Lungs developing</text>
            
            <circle cx="650" cy="130" r="8" fill="#6666ff" />
            <line x1="650" y1="170" x2="650" y2="250" stroke="#666" stroke-dasharray="4" />
            <text x="650" y="265" text-anchor="middle" class="milestone">Baby positions for birth</text>
            
            <!-- Development notes -->
            <text x="200" y="350" class="note">First Trimester: Embryo develops basic structures and organs</text>
            <text x="400" y="380" class="note">Second Trimester: Fetus grows rapidly, features become defined</text>
            <text x="600" y="410" class="note">Third Trimester: Baby gains weight, prepares for birth</text>
            
            <text x="400" y="450" text-anchor="middle" class="note">Note: Timeline shows typical development, individual pregnancies may vary</text>
        </svg>"""
        return svg
    
    def _create_female_anatomy_svg(self) -> str:
        """Create SVG for female reproductive anatomy"""
        svg = """<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
                .label { font-family: Arial, sans-serif; font-size: 14px; }
                .outline { fill: none; stroke: #333; stroke-width: 2; }
                .organ { stroke: #333; stroke-width: 1; }
            </style>
            
            <!-- Title -->
            <text x="400" y="40" text-anchor="middle" class="title">Female Reproductive Anatomy</text>
            
            <!-- Uterus -->
            <path d="M300,300 C300,250 400,200 500,300 C550,350 550,400 400,450 C250,400 250,350 300,300 Z" 
                  fill="#ffcccc" class="organ" />
            <text x="400" y="350" text-anchor="middle" class="label">Uterus</text>
            
            <!-- Fallopian tubes -->
            <path d="M300,300 C250,250 200,300 150,250" fill="none" stroke="#ff9999" stroke-width="4" />
            <path d="M500,300 C550,250 600,300 650,250" fill="none" stroke="#ff9999" stroke-width="4" />
            <text x="180" y="240" class="label">Fallopian tube</text>
            <text x="620" y="240" class="label">Fallopian tube</text>
            
            <!-- Ovaries -->
            <ellipse cx="150" cy="250" rx="30" ry="20" fill="#ffaaaa" class="organ" />
            <ellipse cx="650" cy="250" rx="30" ry="20" fill="#ffaaaa" class="organ" />
            <text x="150" y="200" text-anchor="middle" class="label">Ovary</text>
            <text x="650" y="200" text-anchor="middle" class="label">Ovary</text>
            
            <!-- Cervix -->
            <ellipse cx="400" cy="450" rx="40" ry="20" fill="#cc9999" class="organ" />
            <text x="400" y="490" text-anchor="middle" class="label">Cervix</text>
            
            <!-- Vagina -->
            <path d="M380,450 L380,520 L420,520 L420,450" fill="#cc9999" class="organ" />
            <text x="450" y="500" class="label">Vagina</text>
            
            <!-- External anatomy indicator -->
            <line x1="400" y1="520" x2="400" y2="550" stroke="#333" stroke-dasharray="5,5" />
            <text x="400" y="570" text-anchor="middle" class="label">External anatomy</text>
        </svg>"""
        return svg
    
    def _create_male_anatomy_svg(self) -> str:
        """Create SVG for male reproductive anatomy"""
        svg = """<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
                .label { font-family: Arial, sans-serif; font-size: 14px; }
                .outline { fill: none; stroke: #333; stroke-width: 2; }
                .organ { stroke: #333; stroke-width: 1; }
            </style>
            
            <!-- Title -->
            <text x="400" y="40" text-anchor="middle" class="title">Male Reproductive Anatomy</text>
            
            <!-- Penis outline -->
            <path d="M400,300 L400,450 C350,450 300,400 300,350 L300,250 C300,200 350,150 400,150 C450,150 500,200 500,250 L500,350 C500,400 450,450 400,450" 
                  fill="#ffcccc" class="organ" />
            <text x="520" y="300" class="label">Penis</text>
            
            <!-- Urethra -->
            <path d="M400,200 L400,450" fill="none" stroke="#cc9999" stroke-width="3" stroke-dasharray="5,5" />
            <text x="380" y="180" text-anchor="end" class="label">Urethra</text>
            
            <!-- Bladder indicator -->
            <path d="M400,150 L400,100" fill="none" stroke="#333" stroke-dasharray="5,5" />
            <text x="400" y="90" text-anchor="middle" class="label">Bladder</text>
            
            <!-- Testicles -->
            <ellipse cx="350" cy="500" rx="30" ry="40" fill="#ffaaaa" class="organ" />
            <ellipse cx="450" cy="500" rx="30" ry="40" fill="#ffaaaa" class="organ" />
            <text x="350" y="550" text-anchor="middle" class="label">Testicle</text>
            <text x="450" y="550" text-anchor="middle" class="label">Testicle</text>
            
            <!-- Vas deferens -->
            <path d="M350,470 C350,420 370,400 400,360" fill="none" stroke="#ff9999" stroke-width="3" />
            <path d="M450,470 C450,420 430,400 400,360" fill="none" stroke="#ff9999" stroke-width="3" />
            <text x="320" y="430" text-anchor="end" class="label">Vas deferens</text>
            
            <!-- Prostate -->
            <ellipse cx="400" cy="300" rx="25" ry="15" fill="#cc9999" class="organ" />
            <text x="450" y="270" class="label">Prostate</text>
            
            <!-- Seminal vesicles -->
            <path d="M400,300 C450,280 470,310 480,330" fill="none" stroke="#cc9999" stroke-width="4" />
            <path d="M400,300 C350,280 330,310 320,330" fill="none" stroke="#cc9999" stroke-width="4" />
            <text x="500" y="330" class="label">Seminal vesicle</text>
        </svg>"""
        return svg
    
    def _create_sti_prevention_svg(self) -> str:
        """Create SVG for STI prevention methods"""
        svg = """<svg viewBox="0 0 800 600" xmlns="http://www.w3.org/2000/svg">
            <style>
                .title { font-family: Arial, sans-serif; font-size: 20px; font-weight: bold; }
                .heading { font-family: Arial, sans-serif; font-size: 18px; font-weight: bold; }
                .method { font-family: Arial, sans-serif; font-size: 16px; font-weight: bold; }
                .desc { font-family: Arial, sans-serif; font-size: 14px; }
                .note { font-family: Arial, sans-serif; font-size: 14px; font-style: italic; }
                .icon { stroke: #333; stroke-width: 1; }
            </style>
            
            <!-- Title -->
            <text x="400" y="40" text-anchor="middle" class="title">STI Prevention Methods</text>
            
            <!-- Barrier methods -->
            <text x="200" y="90" class="heading">Barrier Methods</text>
            
            <!-- Condom icon -->
            <circle cx="100" cy="130" r="30" fill="#3498db" class="icon" />
            <text x="100" y="135" text-anchor="middle" fill="white" class="method">C</text>
            <text x="150" y="125" class="method">Condoms</text>
            <text x="150" y="145" class="desc">Highly effective for preventing most STIs</text>
            
            <!-- Dental dam icon -->
            <circle cx="100" cy="190" r="30" fill="#3498db" class="icon" />
            <text x="100" y="195" text-anchor="middle" fill="white" class="method">D</text>
            <text x="150" y="185" class="method">Dental Dams</text>
            <text x="150" y="205" class="desc">Protection during oral sex</text>
            
            <!-- Testing -->
            <text x="200" y="260" class="heading">Testing & Communication</text>
            
            <!-- Testing icon -->
            <circle cx="100" cy="300" r="30" fill="#2ecc71" class="icon" />
            <text x="100" y="305" text-anchor="middle" fill="white" class="method">T</text>
            <text x="150" y="295" class="method">Regular Testing</text>
            <text x="150" y="315" class="desc">Every 3-6 months if sexually active</text>
            
            <!-- Communication icon -->
            <circle cx="100" cy="360" r="30" fill="#2ecc71" class="icon" />
            <text x="100" y="365" text-anchor="middle" fill="white" class="method">C</text>
            <text x="150" y="355" class="method">Communication</text>
            <text x="150" y="375" class="desc">Discuss STI status with partners</text>
            
            <!-- Vaccination -->
            <text x="200" y="430" class="heading">Vaccination</text>
            
            <!-- HPV vaccine icon -->
            <circle cx="100" cy="470" r="30" fill="#9b59b6" class="icon" />
            <text x="100" y="475" text-anchor="middle" fill="white" class="method">H</text>
            <text x="150" y="465" class="method">HPV Vaccine</text>
            <text x="150" y="485" class="desc">Prevents most cases of genital warts and cervical cancer</text>
            
            <!-- Hepatitis vaccine icon -->
            <circle cx="100" cy="530" r="30" fill="#9b59b6" class="icon" />
            <text x="100" y="535" text-anchor="middle" fill="white" class="method">B</text>
            <text x="150" y="525" class="method">Hepatitis B Vaccine</text>
            <text x="150" y="545" class="desc">Prevents hepatitis B infection</text>
            
            <!-- Risk reduction table -->
            <rect x="400" y="100" width="350" height="400" fill="none" stroke="#333" />
            <text x="575" y="90" text-anchor="middle" class="heading">STI Risk Reduction</text>
            
            <!-- Table headers -->
            <rect x="400" y="100" width="175" height="40" fill="#f1c40f" />
            <rect x="575" y="100" width="175" height="40" fill="#f1c40f" />
            <text x="487" y="125" text-anchor="middle" class="method">Method</text>
            <text x="662" y="125" text-anchor="middle" class="method">Effectiveness</text>
            
            <!-- Table rows -->
            <rect x="400" y="140" width="175" height="40" fill="#f5f5f5" />
            <rect x="575" y="140" width="175" height="40" fill="#f5f5f5" />
            <text x="487" y="165" text-anchor="middle" class="desc">External condoms</text>
            <text x="662" y="165" text-anchor="middle" class="desc">Very high</text>
            
            <rect x="400" y="180" width="175" height="40" fill="white" />
            <rect x="575" y="180" width="175" height="40" fill="white" />
            <text x="487" y="205" text-anchor="middle" class="desc">Internal condoms</text>
            <text x="662" y="205" text-anchor="middle" class="desc">High</text>
            
            <rect x="400" y="220" width="175" height="40" fill="#f5f5f5" />
            <rect x="575" y="220" width="175" height="40" fill="#f5f5f5" />
            <text x="487" y="245" text-anchor="middle" class="desc">Dental dams</text>
            <text x="662" y="245" text-anchor="middle" class="desc">Moderate-High</text>
            
            <rect x="400" y="260" width="175" height="40" fill="white" />
            <rect x="575" y="260" width="175" height="40" fill="white" />
            <text x="487" y="285" text-anchor="middle" class="desc">Regular testing</text>
            <text x="662" y="285" text-anchor="middle" class="desc">Supportive</text>
            
            <rect x="400" y="300" width="175" height="40" fill="#f5f5f5" />
            <rect x="575" y="300" width="175" height="40" fill="#f5f5f5" />
            <text x="487" y="325" text-anchor="middle" class="desc">Limiting partners</text>
            <text x="662" y="325" text-anchor="middle" class="desc">Supportive</text>
            
            <rect x="400" y="340" width="175" height="40" fill="white" />
            <rect x="575" y="340" width="175" height="40" fill="white" />
            <text x="487" y="365" text-anchor="middle" class="desc">HPV vaccination</text>
            <text x="662" y="365" text-anchor="middle" class="desc">Very high (for HPV)</text>
            
            <rect x="400" y="380" width="175" height="40" fill="#f5f5f5" />
            <rect x="575" y="380" width="175" height="40" fill="#f5f5f5" />
            <text x="487" y="405" text-anchor="middle" class="desc">Abstinence</text>
            <text x="662" y="405" text-anchor="middle" class="desc">Complete</text>
            
            <rect x="400" y="420" width="175" height="40" fill="white" />
            <rect x="575" y="420" width="175" height="40" fill="white" />
            <text x="487" y="445" text-anchor="middle" class="desc">PrEP</text>
            <text x="662" y="445" text-anchor="middle" class="desc">High (HIV only)</text>
            
            <!-- Note -->
            <text x="400" y="490" class="note">Note: No method except abstinence is 100% effective. Combining methods</text>
            <text x="400" y="510" class="note">provides the best protection. Consult healthcare provider for advice.</text>
        </svg>"""
        return svg