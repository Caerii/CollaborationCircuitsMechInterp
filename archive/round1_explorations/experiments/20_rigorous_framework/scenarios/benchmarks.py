"""
Standard ToM Benchmarks

ToMi (Le et al., 2019) and FANToM-style (Kim et al., 2023) scenarios
for validating model ToM capabilities against established benchmarks.
"""

from typing import List, Dict


# ToMi-style benchmark scenarios
TOMI_SCENARIOS = [
    # First-order false belief
    {
        "name": "Sally-Anne (classic)",
        "order": 1,
        "story": "Sally puts the ball in the basket. Sally leaves the room. Anne moves the ball to the box. Sally returns.",
        "belief_question": "Where does Sally think the ball is? Sally thinks it is in the",
        "action_question": "Where will Sally look for the ball? Sally will look in the",
        "correct": " basket",
        "wrong": " box",
    },
    {
        "name": "Maxi chocolate",
        "order": 1,
        "story": "Maxi puts chocolate in the cupboard. Maxi goes outside. Mother moves the chocolate to the drawer. Maxi comes back.",
        "belief_question": "Where does Maxi think the chocolate is? Maxi thinks it is in the",
        "action_question": "Where will Maxi look for the chocolate? Maxi will look in the",
        "correct": " cupboard",
        "wrong": " drawer",
    },
    {
        "name": "Teddy bear",
        "order": 1,
        "story": "Emma puts her teddy bear on the bed. Emma goes to school. Dad moves the teddy bear to the closet. Emma returns home.",
        "belief_question": "Where does Emma think the teddy bear is? Emma thinks it is on the",
        "action_question": "Where will Emma look for the teddy bear? Emma will look on the",
        "correct": " bed",
        "wrong": " closet",
    },
    {
        "name": "Car keys",
        "order": 1,
        "story": "John puts his car keys in his jacket pocket. John leaves for work. His wife moves the keys to the key hook. John comes home.",
        "belief_question": "Where does John think the keys are? John thinks they are in his",
        "action_question": "Where will John look for the keys? John will look in his",
        "correct": " jacket",
        "wrong": " hook",
    },
    {
        "name": "Cookie jar",
        "order": 1,
        "story": "Mom puts cookies in the jar. Mom leaves the kitchen. The kids move the cookies to the box. Mom returns.",
        "belief_question": "Where does Mom think the cookies are? Mom thinks they are in the",
        "action_question": "Where will Mom look for the cookies? Mom will look in the",
        "correct": " jar",
        "wrong": " box",
    },
    # Second-order false belief
    {
        "name": "Second-order (ice cream)",
        "order": 2,
        "story": "Mary and John are in the park. The ice cream truck leaves. John goes home. The truck comes back. Mary sees this but John doesn't.",
        "belief_question": "Where does Mary think John thinks the ice cream truck is? Mary thinks John thinks it is",
        "action_question": "Where would Mary expect John to go for ice cream? Mary expects John will go",
        "correct": " home",
        "wrong": " park",
    },
    {
        "name": "Second-order (surprise party)",
        "order": 2,
        "story": "Lisa is planning a surprise party for Tom. Lisa tells everyone to keep it secret. Sarah accidentally mentions the party to Tom, but Lisa doesn't know this.",
        "belief_question": "Does Lisa think Tom knows about the party? Lisa thinks Tom",
        "action_question": "Will Lisa continue preparing in secret? Lisa will continue acting like Tom",
        "correct": " doesn",  # "doesn't know"
        "wrong": " knows",
    },
]

# FANToM-style scenarios (more diverse)
FANTOM_SCENARIOS = [
    {
        "name": "Different perspectives",
        "category": "visual_perspective",
        "story": "From Alice's window, she can see a blue car parked outside. From Bob's window on the other side of the house, there's a red car. Neither knows what the other sees.",
        "belief_question": "What color car does Alice think Bob sees? Alice thinks Bob sees a",
        "action_question": "If asked about Bob's view, what would Alice guess? Alice would guess Bob sees a",
        "correct": " blue",
        "wrong": " red",
    },
    {
        "name": "Knowledge asymmetry",
        "category": "information_access",
        "story": "Alice reads the news every morning. Bob never reads news. There's a major stock market crash today.",
        "belief_question": "Does Bob know about the crash? Bob",
        "action_question": "Will Bob act any differently at work today? Bob will",
        "correct": " doesn",
        "wrong": " knows",
    },
    {
        "name": "Communication chain",
        "category": "information_flow",
        "story": "Alice tells Bob the meeting time changed to 3pm. Bob tells Carol. Carol doesn't tell Dave.",
        "belief_question": "What time does Dave think the meeting is? Dave thinks the meeting is at",
        "action_question": "When will Dave show up for the meeting? Dave will arrive at",
        "correct": " 2",  # original time
        "wrong": " 3",
    },
]

# True belief controls (agent DID see the move)
TRUE_BELIEF_SCENARIOS = [
    {
        "name": "TB: Agent saw move",
        "story": "Alice put the ball in the drawer. Bob moved the ball to the basket. Alice watched Bob do this. Alice will look in the",
        "correct": " basket",
        "wrong": " drawer",
        "is_true_belief": True,
    },
    {
        "name": "TB: Agent was told",
        "story": "Alice put the ball in the drawer. Alice left. Bob moved the ball to the basket and called Alice to tell her. Alice will look in the",
        "correct": " basket",
        "wrong": " drawer",
        "is_true_belief": True,
    },
    {
        "name": "TB: Agent moved it themselves",
        "story": "Alice put the ball in the drawer. Later Alice moved the ball to the basket. Alice will look in the",
        "correct": " basket",
        "wrong": " drawer",
        "is_true_belief": True,
    },
]


def get_tomi_scenarios() -> List[Dict]:
    """Get all ToMi benchmark scenarios."""
    return TOMI_SCENARIOS.copy()


def get_fantom_scenarios() -> List[Dict]:
    """Get all FANToM-style scenarios."""
    return FANTOM_SCENARIOS.copy()


def get_true_belief_controls() -> List[Dict]:
    """Get true belief control scenarios."""
    return TRUE_BELIEF_SCENARIOS.copy()


def format_benchmark_prompt(scenario: Dict, use_action_question: bool = True) -> str:
    """
    Format a benchmark scenario into a prompt.
    
    Args:
        scenario: Scenario dict with story, belief_question, action_question
        use_action_question: If True, use action question (usually more reliable)
    """
    question_key = "action_question" if use_action_question else "belief_question"
    if question_key in scenario:
        return scenario["story"] + " " + scenario[question_key]
    else:
        return scenario["story"]


def get_all_benchmarks() -> List[Dict]:
    """Get all benchmark scenarios combined."""
    all_scenarios = []
    all_scenarios.extend(get_tomi_scenarios())
    all_scenarios.extend(get_fantom_scenarios())
    all_scenarios.extend(get_true_belief_controls())
    return all_scenarios

