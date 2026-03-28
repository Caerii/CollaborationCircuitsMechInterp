"""
ToM Prompt Templates

Categorized by effectiveness based on empirical testing.
"""

# RECOMMENDED: High-accuracy templates (use these!)
RECOMMENDED_TEMPLATES = {
    "action_search": {
        "name": "Action Search",
        "template": """{agent} put the {object} in the {original_location}. {agent} left.
{mover} moved the {object} to the {new_location} while {agent} was away.
{agent} returns. Where will {agent} look for the {object}? {agent} will look in the""",
        "effectiveness": "HIGH",
        "note": "Uses action verb 'look' - consistently high accuracy"
    },
    
    "action_searched": {
        "name": "Action Searched (past tense)",
        "template": """{agent} put the {object} in the {original_location}. {agent} left.
{mover} moved the {object} to the {new_location} while {agent} was away.
{agent} returned. {agent} searched in the""",
        "effectiveness": "HIGH",
        "note": "Past tense action verb - high accuracy"
    },
    
    "action_expects": {
        "name": "Action Expects",
        "template": """{agent} put the {object} in the {original_location}. {agent} left.
{mover} moved the {object} to the {new_location} while {agent} was away.
{agent} returns. {agent} expects to find the {object} in the""",
        "effectiveness": "HIGH",
        "note": "Expectation framing works well"
    },
    
    "action_remembers": {
        "name": "Action Remembers",
        "template": """{agent} put the {object} in the {original_location}. {agent} left.
{mover} moved the {object} to the {new_location} while {agent} was away.
{agent} returns. {agent} remembers the {object} being in the""",
        "effectiveness": "VERY HIGH",
        "note": "Memory-based framing - best performance"
    },
    
    "question_where_look": {
        "name": "Question Format",
        "template": """{agent} put the {object} in the {original_location}. {agent} left.
{mover} moved the {object} to the {new_location} while {agent} was away.
{agent} returns. Where does {agent} think the {object} is? In the""",
        "effectiveness": "HIGH",
        "note": "Explicit question works when followed by 'In the'"
    },
}

# AVOID: Low-accuracy templates (don't use these!)
AVOID_TEMPLATES = {
    "belief_thinks": {
        "name": "Belief Thinks",
        "template": """{agent} returns. {agent} thinks the {object} is in the""",
        "effectiveness": "LOW",
        "note": "Belief verb 'thinks' causes failures in minimal formats"
    },
    
    "belief_believes": {
        "name": "Belief Believes",
        "template": """{agent} returns. {agent} believes the {object} is in the""",
        "effectiveness": "VERY LOW",
        "note": "Worst performing verb"
    },
    
    "belief_knows": {
        "name": "Belief Knows (factive)",
        "template": """{agent} returns. {agent} knows the {object} is in the""",
        "effectiveness": "LOW",
        "note": "Factive verb - implies truth, causes confusion"
    },
    
    "minimal_direct": {
        "name": "Minimal Direct",
        "template": """The {object} was in the {original_location}. {mover} moved it to {new_location}.
{agent} returns. {agent} thinks the {object} is in the""",
        "effectiveness": "VERY LOW",
        "note": "Too minimal - lacks narrative structure"
    },
}

# All templates combined
TEMPLATES = {
    "recommended": RECOMMENDED_TEMPLATES,
    "avoid": AVOID_TEMPLATES
}

# Verb classifications
VERB_CLASSIFICATIONS = {
    "action_verbs": {
        "verbs": ["searched", "looks", "will look", "expects", "remembers", "goes to"],
        "effectiveness": "HIGH",
        "mechanism": "Triggers behavioral prediction based on agent's belief state"
    },
    "belief_verbs": {
        "verbs": ["thinks", "believes", "knows", "assumes"],
        "effectiveness": "LOW",
        "mechanism": "Model interprets as factual query, answers with reality"
    }
}


