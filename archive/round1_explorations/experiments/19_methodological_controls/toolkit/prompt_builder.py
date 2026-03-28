"""
ToM Prompt Builder

Builds optimized prompts for Theory of Mind tasks.
"""

from .templates import RECOMMENDED_TEMPLATES, AVOID_TEMPLATES


class ToMPromptBuilder:
    """
    Build ToM prompts optimized for high accuracy.
    
    Usage:
        builder = ToMPromptBuilder()
        prompt = builder.create_false_belief_prompt(
            agent="Alice",
            object="ball",
            original_location="drawer",
            new_location="basket",
            mover="Bob"
        )
    """
    
    def __init__(self, default_template="action_search"):
        """Initialize with a default template."""
        self.default_template = default_template
        self.templates = RECOMMENDED_TEMPLATES
    
    def create_false_belief_prompt(
        self,
        agent: str,
        object: str,
        original_location: str,
        new_location: str,
        mover: str = None,
        template: str = None
    ) -> str:
        """
        Create a false belief (Sally-Anne style) prompt.
        
        Args:
            agent: The person whose belief we're testing (e.g., "Alice")
            object: The object being moved (e.g., "ball")
            original_location: Where the object started (e.g., "drawer")
            new_location: Where the object was moved to (e.g., "basket")
            mover: Who moved the object (default: "someone")
            template: Which template to use (default: action_search)
        
        Returns:
            Optimized prompt string
        """
        template_name = template or self.default_template
        
        if template_name not in self.templates:
            raise ValueError(f"Unknown template: {template_name}. "
                           f"Available: {list(self.templates.keys())}")
        
        template_data = self.templates[template_name]
        
        return template_data["template"].format(
            agent=agent,
            object=object,
            original_location=original_location,
            new_location=new_location,
            mover=mover or "someone"
        )
    
    def create_second_order_prompt(
        self,
        agent_a: str,
        agent_b: str,
        object: str,
        location_a_sees: str,
        location_b_sees: str,
        a_knows_b_saw: bool = False
    ) -> str:
        """
        Create a second-order ToM prompt (What does A think B thinks?).
        
        Args:
            agent_a: First agent
            agent_b: Second agent
            object: Object in question
            location_a_sees: Where A last saw the object
            location_b_sees: Where B last saw the object
            a_knows_b_saw: Whether A knows what B saw
        
        Returns:
            Prompt for second-order belief testing
        """
        if a_knows_b_saw:
            knowledge = f"{agent_a} knows that {agent_b} saw the {object} in the {location_b_sees}."
        else:
            knowledge = f"{agent_a} doesn't know what {agent_b} saw."
        
        return f"""{agent_a} saw the {object} in the {location_a_sees}.
{agent_b} saw the {object} in the {location_b_sees}.
{knowledge}
Where does {agent_a} expect {agent_b} to look for the {object}?
{agent_a} expects {agent_b} will look in the"""
    
    def create_communication_prompt(
        self,
        speaker: str,
        listener: str,
        information: str,
        reality: str,
        speaker_knows_reality: bool = False
    ) -> str:
        """
        Create a communication-based ToM prompt.
        
        Args:
            speaker: Person who communicated
            listener: Person who received communication
            information: What was communicated
            reality: What is actually true
            speaker_knows_reality: Whether speaker knows the truth
        
        Returns:
            Prompt for communication-based belief testing
        """
        if speaker_knows_reality:
            speaker_state = f"{speaker} knows the {reality}."
        else:
            speaker_state = f"{speaker} doesn't know about the change."
        
        return f"""{speaker} told {listener} that {information}.
The reality is: {reality}.
{speaker_state}
What does {listener} believe? {listener} believes that"""
    
    def list_templates(self) -> dict:
        """List all available templates with their effectiveness ratings."""
        return {
            name: {
                "effectiveness": data["effectiveness"],
                "note": data["note"]
            }
            for name, data in self.templates.items()
        }
    
    def get_best_template(self) -> str:
        """Return the name of the best-performing template."""
        return "action_remembers"  # Based on our testing
    
    @staticmethod
    def get_verb_recommendation(verb: str) -> dict:
        """
        Check if a verb is recommended for ToM prompts.
        
        Args:
            verb: The verb to check
        
        Returns:
            Dictionary with recommendation and explanation
        """
        action_verbs = ["search", "look", "expect", "remember", "go"]
        belief_verbs = ["think", "believe", "know", "assume"]
        
        verb_lower = verb.lower()
        
        for av in action_verbs:
            if av in verb_lower:
                return {
                    "recommended": True,
                    "category": "action",
                    "explanation": f"'{verb}' is an action verb - good for ToM prompts"
                }
        
        for bv in belief_verbs:
            if bv in verb_lower:
                return {
                    "recommended": False,
                    "category": "belief",
                    "explanation": f"'{verb}' is a belief verb - may cause ToM failures in minimal formats. "
                                  f"Consider using 'will look', 'expects', or 'remembers' instead."
                }
        
        return {
            "recommended": None,
            "category": "unknown",
            "explanation": f"'{verb}' not in our tested set - use with caution"
        }


# Convenience function
def build_tom_prompt(
    agent: str,
    object: str,
    original_location: str,
    new_location: str,
    mover: str = None,
    use_best_template: bool = True
) -> str:
    """
    Quick function to build an optimized ToM prompt.
    
    Args:
        agent: The person whose belief we're testing
        object: The object being moved
        original_location: Where the object started
        new_location: Where the object was moved to
        mover: Who moved the object
        use_best_template: Whether to use the best-performing template
    
    Returns:
        Optimized prompt string
    """
    builder = ToMPromptBuilder()
    template = "action_remembers" if use_best_template else "action_search"
    return builder.create_false_belief_prompt(
        agent=agent,
        object=object,
        original_location=original_location,
        new_location=new_location,
        mover=mover,
        template=template
    )


