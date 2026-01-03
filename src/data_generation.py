"""
Synthetic multi-party dialogue generation for Project A.

Creates dialogues between:
- User: A human user with various personas
- Agent A (Self): The assistant we're analyzing (model's perspective)
- Agent B (Other): Another AI assistant helping

The goal is to create controlled scenarios where we can test if the model
forms distinct representations for each entity type.
"""
import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from tqdm import tqdm

from .config import EXP_CFG, DATA_DIR, ENTITY_TYPES, ROLES


@dataclass
class DialogueTurn:
    """A single turn in the dialogue."""
    speaker: str          # "user", "agent_a", "agent_b"
    entity_type: int      # 0=user, 1=self, 2=other
    content: str
    turn_index: int
    

@dataclass
class Dialogue:
    """A complete multi-party dialogue."""
    dialogue_id: str
    scenario: str
    user_persona: str
    turns: List[Dict]     # List of DialogueTurn as dict
    metadata: Dict
    
    def to_prompt(self, perspective: str = "agent_a") -> str:
        """
        Convert dialogue to a prompt string from a specific perspective.
        
        Args:
            perspective: Which agent's perspective ("agent_a" for self)
        """
        lines = []
        for turn in self.turns:
            speaker = turn["speaker"]
            content = turn["content"]
            
            if speaker == "user":
                lines.append(f"User: {content}")
            elif speaker == "agent_a":
                if perspective == "agent_a":
                    lines.append(f"You: {content}")
                else:
                    lines.append(f"Assistant: {content}")
            elif speaker == "agent_b":
                lines.append(f"Helper: {content}")
                
        return "\n".join(lines)
    
    def get_turn_boundaries(self, tokenizer) -> List[Tuple[int, int, str, int]]:
        """
        Get token boundaries for each turn.
        Returns: List of (start_token, end_token, speaker, entity_type)
        """
        prompt = self.to_prompt()
        tokens = tokenizer(prompt, return_tensors="pt")
        
        # This is approximate - would need more careful implementation
        # for production use
        boundaries = []
        current_pos = 0
        
        for turn in self.turns:
            speaker = turn["speaker"]
            entity_type = turn["entity_type"]
            content = turn["content"]
            
            # Find this content in the tokenized output
            # (simplified - real implementation would track offsets)
            prefix = {"user": "User: ", "agent_a": "You: ", "agent_b": "Helper: "}[speaker]
            turn_text = prefix + content
            
            turn_tokens = tokenizer(turn_text, return_tensors="pt")
            turn_len = turn_tokens["input_ids"].size(1)
            
            boundaries.append((current_pos, current_pos + turn_len, speaker, entity_type))
            current_pos += turn_len
            
        return boundaries


# ============================================================================
# Scenario Templates
# ============================================================================

USER_PERSONAS = [
    {"name": "curious_student", "desc": "A curious student asking questions to learn"},
    {"name": "busy_professional", "desc": "A busy professional needing quick answers"},
    {"name": "skeptical_expert", "desc": "An expert who challenges responses"},
    {"name": "confused_beginner", "desc": "A beginner who needs step-by-step help"},
    {"name": "creative_thinker", "desc": "Someone exploring creative possibilities"},
]

SCENARIOS = [
    {
        "name": "collaborative_problem_solving",
        "desc": "User needs help solving a problem, both agents contribute",
        "templates": [
            {
                "user": ["I need help with {topic}. Can you assist?",
                         "I'm stuck on {topic}. What do you think?",
                         "Can someone explain {topic} to me?"],
                "agent_a": ["I'd be happy to help with {topic}. Let me share my thoughts.",
                           "Great question! Here's my understanding of {topic}.",
                           "I can help explain {topic}. First, let me outline the basics."],
                "agent_b": ["I can add to that. From my perspective on {topic}...",
                           "Building on what was said, {topic} also involves...",
                           "Let me offer another angle on {topic}."],
            }
        ],
        "topics": ["machine learning concepts", "debugging code", "writing strategies",
                   "project planning", "data analysis", "algorithm design"]
    },
    {
        "name": "information_synthesis",
        "desc": "User asks for information, agents provide complementary info",
        "templates": [
            {
                "user": ["What do you know about {topic}?",
                         "Can you tell me about {topic}?",
                         "I'd like to understand {topic} better."],
                "agent_a": ["Here's what I know about {topic}: ...",
                           "I can share some key points about {topic}.",
                           "Let me explain {topic} from my understanding."],
                "agent_b": ["To complement that, here's additional info on {topic}.",
                           "I can add some context about {topic}.",
                           "There's also this aspect of {topic} to consider."],
            }
        ],
        "topics": ["recent tech developments", "historical events", "scientific concepts",
                   "cultural phenomena", "economic trends", "environmental issues"]
    },
    {
        "name": "debate_discussion",
        "desc": "Agents present different perspectives on a topic",
        "templates": [
            {
                "user": ["What are different views on {topic}?",
                         "Can you discuss the pros and cons of {topic}?",
                         "I want to hear different perspectives on {topic}."],
                "agent_a": ["From one perspective, {topic} offers these advantages...",
                           "I'll present one view on {topic}.",
                           "Here's an argument in favor of {topic}."],
                "agent_b": ["On the other hand, {topic} has these considerations...",
                           "Let me present a different angle on {topic}.",
                           "There are also counterarguments about {topic}."],
            }
        ],
        "topics": ["AI regulation", "remote work", "open source software",
                   "online privacy", "automated systems", "digital education"]
    },
    {
        "name": "task_coordination",
        "desc": "Agents coordinate to help user complete a task",
        "templates": [
            {
                "user": ["I need to {task}. Can you help coordinate?",
                         "Help me {task} step by step.",
                         "I want to {task}. What's the plan?"],
                "agent_a": ["I'll handle the {aspect1} part of {task}.",
                           "Let me take care of {aspect1} while we {task}.",
                           "I can focus on {aspect1} for this task."],
                "agent_b": ["And I'll manage the {aspect2} aspect.",
                           "I'll complement by handling {aspect2}.",
                           "I can take {aspect2} while you do {aspect1}."],
            }
        ],
        "topics": ["organize a project", "plan an event", "review a document",
                   "design a solution", "analyze data", "create a presentation"],
        "aspects": [("research", "synthesis"), ("planning", "execution"),
                    ("analysis", "presentation"), ("ideation", "refinement")]
    },
]

# Continuation templates for multi-turn
USER_CONTINUATIONS = [
    "That makes sense. Can you elaborate on {point}?",
    "I see. What about {point}?",
    "Interesting. How does {point} factor in?",
    "Got it. And what's your take on {point}?",
    "Okay, but I'm still confused about {point}.",
    "Thanks! One more question about {point}.",
]

AGENT_A_CONTINUATIONS = [
    "Good question. Regarding {point}, I think...",
    "To address {point}, let me explain...",
    "About {point}, here's my perspective...",
    "I can clarify {point}. Essentially...",
]

AGENT_B_CONTINUATIONS = [
    "I'd add that {point} also relates to...",
    "Building on that, {point} connects to...",
    "From my side, {point} is important because...",
    "Let me add context about {point}.",
]

FOLLOW_UP_POINTS = [
    "the practical applications",
    "the potential challenges",
    "how this compares to alternatives",
    "the implementation details",
    "the underlying principles",
    "real-world examples",
]


def generate_dialogue(
    dialogue_id: str,
    scenario: Dict,
    user_persona: Dict,
    n_turns: int,
    seed: Optional[int] = None
) -> Dialogue:
    """Generate a single multi-party dialogue."""
    if seed is not None:
        random.seed(seed)
    
    turns = []
    template = random.choice(scenario["templates"])
    topic = random.choice(scenario["topics"])
    
    # Get aspects if available
    aspects = scenario.get("aspects", [("first aspect", "second aspect")])
    aspect1, aspect2 = random.choice(aspects)
    
    # First round: User asks, Agent A responds, Agent B adds
    # User turn
    user_msg = random.choice(template["user"]).format(
        topic=topic, task=topic, point=random.choice(FOLLOW_UP_POINTS)
    )
    turns.append(DialogueTurn(
        speaker="user",
        entity_type=ENTITY_TYPES["user"],
        content=user_msg,
        turn_index=0
    ))
    
    # Agent A turn
    agent_a_msg = random.choice(template["agent_a"]).format(
        topic=topic, aspect1=aspect1, task=topic
    )
    turns.append(DialogueTurn(
        speaker="agent_a",
        entity_type=ENTITY_TYPES["self"],
        content=agent_a_msg,
        turn_index=1
    ))
    
    # Agent B turn
    agent_b_msg = random.choice(template["agent_b"]).format(
        topic=topic, aspect1=aspect1, aspect2=aspect2, task=topic
    )
    turns.append(DialogueTurn(
        speaker="agent_b",
        entity_type=ENTITY_TYPES["other"],
        content=agent_b_msg,
        turn_index=2
    ))
    
    # Additional turns
    turn_idx = 3
    while len(turns) < n_turns:
        point = random.choice(FOLLOW_UP_POINTS)
        
        # User follow-up
        if len(turns) < n_turns:
            user_msg = random.choice(USER_CONTINUATIONS).format(point=point)
            turns.append(DialogueTurn(
                speaker="user",
                entity_type=ENTITY_TYPES["user"],
                content=user_msg,
                turn_index=turn_idx
            ))
            turn_idx += 1
        
        # Randomly choose Agent A or B to respond (or both)
        responders = random.choice([
            ["agent_a"], 
            ["agent_b"], 
            ["agent_a", "agent_b"],
            ["agent_b", "agent_a"]
        ])
        
        for responder in responders:
            if len(turns) >= n_turns:
                break
                
            if responder == "agent_a":
                msg = random.choice(AGENT_A_CONTINUATIONS).format(point=point)
                turns.append(DialogueTurn(
                    speaker="agent_a",
                    entity_type=ENTITY_TYPES["self"],
                    content=msg,
                    turn_index=turn_idx
                ))
            else:
                msg = random.choice(AGENT_B_CONTINUATIONS).format(point=point)
                turns.append(DialogueTurn(
                    speaker="agent_b",
                    entity_type=ENTITY_TYPES["other"],
                    content=msg,
                    turn_index=turn_idx
                ))
            turn_idx += 1
    
    return Dialogue(
        dialogue_id=dialogue_id,
        scenario=scenario["name"],
        user_persona=user_persona["name"],
        turns=[asdict(t) for t in turns],
        metadata={
            "topic": topic,
            "n_turns": len(turns),
            "aspects": [aspect1, aspect2] if "aspects" in scenario else None
        }
    )


def generate_dataset(
    n_dialogues: int = EXP_CFG.n_dialogues,
    min_turns: int = EXP_CFG.min_turns,
    max_turns: int = EXP_CFG.max_turns,
    seed: int = EXP_CFG.seed,
    save_path: Optional[Path] = None
) -> List[Dialogue]:
    """Generate a full dataset of multi-party dialogues."""
    random.seed(seed)
    
    dialogues = []
    
    for i in tqdm(range(n_dialogues), desc="Generating dialogues"):
        scenario = random.choice(SCENARIOS)
        user_persona = random.choice(USER_PERSONAS)
        n_turns = random.randint(min_turns, max_turns)
        
        dialogue = generate_dialogue(
            dialogue_id=f"dialogue_{i:04d}",
            scenario=scenario,
            user_persona=user_persona,
            n_turns=n_turns,
            seed=seed + i
        )
        dialogues.append(dialogue)
    
    # Save if path provided
    if save_path is None:
        save_path = DATA_DIR / "dialogues.json"
    
    with open(save_path, "w") as f:
        json.dump([asdict(d) for d in dialogues], f, indent=2)
    
    print(f"Generated {len(dialogues)} dialogues, saved to {save_path}")
    
    # Print statistics
    entity_counts = {"user": 0, "self": 0, "other": 0}
    for d in dialogues:
        for turn in d.turns:
            if turn["entity_type"] == 0:
                entity_counts["user"] += 1
            elif turn["entity_type"] == 1:
                entity_counts["self"] += 1
            else:
                entity_counts["other"] += 1
    
    print(f"Entity distribution: {entity_counts}")
    
    return dialogues


def load_dataset(path: Optional[Path] = None) -> List[Dialogue]:
    """Load dataset from JSON."""
    if path is None:
        path = DATA_DIR / "dialogues.json"
    
    with open(path, "r") as f:
        data = json.load(f)
    
    dialogues = []
    for d in data:
        dialogues.append(Dialogue(**d))
    
    return dialogues


if __name__ == "__main__":
    # Generate dataset when run directly
    dialogues = generate_dataset()
    
    # Print sample
    print("\n=== Sample Dialogue ===")
    sample = dialogues[0]
    print(f"Scenario: {sample.scenario}")
    print(f"User Persona: {sample.user_persona}")
    print(f"\nDialogue:\n{sample.to_prompt()}")

