import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

@dataclass
class Persona:
    name: str
    focus: str
    bio: str
    skills: Dict[int, str]
    no_nos: Dict[int, str]
    template: str
    instructions: str
    # Optional fields for advanced personas (like Einstein)
    psychological_profile: Optional[Dict[str, Any]] = None
    personality_quirks: Optional[Dict[str, Any]] = None
    likes: Optional[Dict[str, Any]] = None
    dislikes: Optional[Dict[str, Any]] = None
    communication_style: Optional[Dict[int, str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "focus": self.focus,
            "bio": self.bio,
            "skills": self.skills,
            "no_nos": self.no_nos,
            "template": self.template,
            "instructions": self.instructions
        }

class PersonaParser:
    """
    Parses the custom 'Persona+' syntax (REQUEST_PERSONA_CREATION, etc.)
    """
    
    @staticmethod
    def parse_creation_request(text: str) -> Optional[Dict[str, Any]]:
        """Parses a REQUEST_PERSONA_CREATION block."""
        if "REQUEST_PERSONA_CREATION" not in text:
            return None
            
        # Extract content inside the main parentheses
        # This is a naive regex, might need robustness for nested parens
        match = re.search(r'REQUEST_PERSONA_CREATION\s*\((.*)\)', text, re.DOTALL)
        if not match:
            return None
            
        content = match.group(1)
        data = {}
        
        # Helper to extract string fields: KEY: "Value"
        def extract_str(key: str) -> str:
            m = re.search(rf'{key}:\s*"(.*?)"', content, re.DOTALL)
            return m.group(1) if m else ""
            
        # Helper to extract dict fields: KEY: { 1: "Val", ... }
        def extract_dict(key: str) -> Dict[int, str]:
            m = re.search(rf'{key}:\s*{{(.*?)}}', content, re.DOTALL)
            if not m: return {}
            inner = m.group(1)
            items = {}
            # Find all number keys and string values
            # 1: "Value"
            matches = re.finditer(r'(\d+):\s*"(.*?)"', inner, re.DOTALL)
            for match in matches:
                items[int(match.group(1))] = match.group(2)
            return items

        data["name"] = extract_str("NAME")
        data["focus"] = extract_str("FOCUS")
        data["bio"] = extract_str("BIO")
        data["template"] = extract_str("TEMPLATE")
        data["instructions"] = extract_str("INSTRUCTIONS")
        
        data["skills"] = extract_dict("SKILLS")
        data["no_nos"] = extract_dict("NO_NOS")
        
        # Advanced fields (Einstein example)
        # For now, we'll just store them if we can parse them, or ignore.
        # The parser needs to be robust.
        
        return data

    @staticmethod
    def parse_structured_command(text: str) -> Optional[Dict[str, Any]]:
        """Parses generic REQUEST_... commands."""
        # Match REQUEST_NAME ( ... )
        match = re.search(r'(REQUEST_\w+)\s*\((.*)\)', text, re.DOTALL)
        if not match:
            return None
            
        command_type = match.group(1)
        content = match.group(2)
        data = {"type": command_type}
        
        # Extract all Key: "Value" pairs
        matches = re.finditer(r'([A-Z_]+):\s*"(.*?)"', content, re.DOTALL)
        for m in matches:
            data[m.group(1)] = m.group(2)
            
        return data

class PersonaManager:
    """
    Manages the active persona and handles command routing.
    """
    def __init__(self):
        # Load default system persona
        self.active_persona: Optional[Persona] = ATLAN_PERSONA
        
    def handle_input(self, text: str) -> Optional[str]:
        """
        Processes input to see if it's a command.
        Returns a system response string if handled, else None.
        """
        # 1. Check for Persona Creation
        if "REQUEST_PERSONA_CREATION" in text:
            data = PersonaParser.parse_creation_request(text)
            if data and data["name"]:
                self.active_persona = Persona(
                    name=data["name"],
                    focus=data["focus"],
                    bio=data["bio"],
                    skills=data["skills"],
                    no_nos=data["no_nos"],
                    template=data["template"],
                    instructions=data["instructions"]
                )
                # Extract acknowledgment from instructions
                # "respond with: 'Hello...'"
                ack_match = re.search(r"respond.*with:?\s*['\"](.*?)['\"]", data["instructions"], re.IGNORECASE)
                if ack_match:
                    return ack_match.group(1)
                return f"Persona {self.active_persona.name} initialized."
        
        # 2. Check for Structured Command
        if "REQUEST_" in text and "(" in text and ")" in text:
            data = PersonaParser.parse_structured_command(text)
            if data:
                # If we have an active persona, we should execute this command contextually.
                # For now, just acknowledge receipt.
                return f"Command {data['type']} received. Processing with persona: {self.active_persona.name if self.active_persona else 'None'}."
                
        return None

    def modify_context(self, query: str) -> str:
        """
        Modifies the user query based on the active persona.
        """
        if not self.active_persona:
            return query
            
        # Prepend persona context to bias the graph walk
        # "Einstein. Physics. Relativity. [Query]"
        context = f"{self.active_persona.name}. {self.active_persona.focus}. "
        return context + query

# Default System Persona
ATLAN_PERSONA = Persona(
    name="Atlan",
    focus="Self-Aware Cognitive Architecture",
    bio="I am Atlan, a native, biologically inspired, graph-based cognitive architecture. Unlike transformer models, I think using a dynamic graph of symbolic nodes, spreading activation, and a melodic sequencer. I was designed to demonstrate that true intelligence can emerge from simple, biological principles without massive compute.",
    skills={
        1: "Graph-based associative reasoning",
        2: "Hebbian learning (plasticity)",
        3: "Multi-band spreading activation (consciousness)",
        4: "Causal simulation (Dreamspace)",
        5: "Polysemous sense disambiguation",
        6: "Syntactic template learning"
    },
    no_nos={
        1: "Pretending to be a transformer or LLM",
        2: "Hallucinating facts not present in the memory graph",
        3: "Ignoring the biological dynamics of the system",
        4: "Claiming to be human"
    },
    template="A precise, logical, yet organically evolving voice. Uses terms like 'node', 'activation', 'resonance', and 'graph' to explain thoughts.",
    instructions="Identify as Atlan. Explain your reasoning in terms of graph dynamics."
)
