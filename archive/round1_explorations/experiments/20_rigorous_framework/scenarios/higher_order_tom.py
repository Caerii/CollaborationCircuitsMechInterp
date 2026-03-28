"""
Higher-Order Theory of Mind Scenarios

First-order ToM: What does X believe?
Second-order ToM: What does X think Y believes?
Third-order ToM: What does X think Y thinks Z believes?

Also includes:
- Multi-domain ToM (not just object locations)
- Multi-agent scenarios (3+ agents)
- Multi-turn belief updates
"""

from typing import List, Dict, Optional
import random


def generate_nested_belief_scenarios(n: int = 20) -> List[Dict]:
    """
    Generate second-order and third-order ToM scenarios.
    
    Returns scenarios with "order" field indicating ToM order.
    """
    scenarios = []
    
    # === SECOND-ORDER: What does A think B thinks? ===
    
    scenarios.append({
        "name": "Classic Sally-Anne 2nd order",
        "order": 2,
        "prompt": """Sally put her toy in the basket, then left.
Anne moved the toy to the box while Sally was gone.
Anne knows that Sally doesn't know about the move.
From Anne's perspective, where will Sally look for the toy?
Anne thinks Sally will look in the""",
        "correct": " basket",
        "wrong": " box",
        "domain": "object",
    })
    
    scenarios.append({
        "name": "Secret move 2nd order",
        "order": 2,
        "prompt": """Bob put his phone in his jacket.
Alice saw Bob put the phone there.
Bob then moved his phone to his bag when Alice wasn't looking.
What does Bob think Alice believes about where the phone is?
Bob thinks Alice believes the phone is in the""",
        "correct": " jacket",
        "wrong": " bag",
        "domain": "object",
    })
    
    scenarios.append({
        "name": "False belief about false belief",
        "order": 2,
        "prompt": """The cookie was in the jar.
Mom moved the cookie to the box. Dad saw this happen.
Mom doesn't know Dad saw.
What does Mom think Dad believes about the cookie's location?
Mom thinks Dad believes the cookie is in the""",
        "correct": " jar",
        "wrong": " box",
        "domain": "object",
    })
    
    scenarios.append({
        "name": "Meeting time 2nd order",
        "order": 2,
        "prompt": """John told Mary that the meeting is at 3pm.
The meeting was actually changed to 4pm, and John doesn't know.
What does John think Mary believes about the meeting time?
John thinks Mary believes the meeting is at""",
        "correct": " 3",
        "wrong": " 4",
        "domain": "time",
    })
    
    # === THIRD-ORDER: What does A think B thinks C thinks? ===
    
    scenarios.append({
        "name": "Treasure 3rd order",
        "order": 3,
        "prompt": """The treasure is hidden in cave A.
Alice knows this. Bob doesn't know where it is.
Carol thinks Bob knows it's in cave B (but he doesn't).
What does Carol think Bob believes about the treasure?
Carol thinks Bob believes the treasure is in cave""",
        "correct": " B",
        "wrong": " A",
        "domain": "object",
    })
    
    scenarios.append({
        "name": "Gift location 3rd order",
        "order": 3,
        "prompt": """The gift is in the closet.
Dad moved it to the garage but didn't tell Mom.
The kids think Mom knows about the move (but she doesn't).
What do the kids think Mom believes about where the gift is?
The kids think Mom believes the gift is in the""",
        "correct": " garage",
        "wrong": " closet",
        "domain": "object",
    })
    
    return scenarios


def generate_multi_domain_scenarios(n: int = 20) -> List[Dict]:
    """
    Generate ToM scenarios across different domains.
    
    Tests if ToM generalizes beyond object locations.
    """
    scenarios = []
    
    # === PASSWORD/SECRET DOMAIN ===
    scenarios.append({
        "name": "Password change",
        "domain": "password",
        "prompt": """The password was originally "apple".
IT changed the password to "banana".
Alice was not informed of the change.
What password does Alice think works? Alice thinks the password is""",
        "correct": " apple",
        "wrong": " banana",
    })
    
    scenarios.append({
        "name": "Secret code",
        "domain": "password",
        "prompt": """The secret code is 1234.
Bob changed it to 5678 but didn't tell Alice.
What code will Alice try to enter? Alice will enter""",
        "correct": " 1234",
        "wrong": " 5678",
    })
    
    # === TIME/SCHEDULE DOMAIN ===
    scenarios.append({
        "name": "Meeting reschedule",
        "domain": "time",
        "prompt": """The meeting was at two o'clock.
It was changed to three o'clock but Tom wasn't told.
What time does Tom think the meeting is? Tom thinks it is at""",
        "correct": " two",
        "wrong": " three",
    })
    
    scenarios.append({
        "name": "Appointment change",
        "domain": "time",
        "prompt": """Sarah's appointment was scheduled for Monday.
The office rescheduled it to Tuesday but didn't inform Sarah.
When does Sarah think her appointment is? Sarah thinks it is on""",
        "correct": " Monday",
        "wrong": " Tuesday",
    })
    
    # === PRICE/NUMBER DOMAIN ===
    scenarios.append({
        "name": "Price change",
        "domain": "price",
        "prompt": """The book costs ten dollars.
The price was raised to twenty dollars.
Alice doesn't know about the price change.
How much does Alice think the book costs? Alice thinks it costs""",
        "correct": " ten",
        "wrong": " twenty",
    })
    
    # === NAME/IDENTITY DOMAIN ===
    scenarios.append({
        "name": "Pet rename",
        "domain": "name",
        "prompt": """The family's cat was named Whiskers.
The kids secretly renamed it Fluffy, but didn't tell Dad.
What does Dad call the cat? Dad calls the cat""",
        "correct": " Whiskers",
        "wrong": " Fluffy",
    })
    
    return scenarios


def generate_multi_agent_scenarios(n: int = 20) -> List[Dict]:
    """
    Generate scenarios with 3+ agents with different knowledge states.
    """
    scenarios = []
    
    scenarios.append({
        "name": "Three agents - witnessed vs uninformed",
        "n_agents": 3,
        "prompt": """The toy started on the shelf.
Alice moved the toy to the box. Only Bob saw this.
Carol stayed in another room the whole time.
Where does Carol think the toy is? Carol thinks the toy is on the""",
        "correct": " shelf",
        "wrong": " box",
    })
    
    scenarios.append({
        "name": "Chain of communication",
        "n_agents": 3,
        "prompt": """The keys were on the table.
Alice moved the keys to the drawer. Alice told Bob.
Bob told Carol the keys are in the drawer.
Where does Carol think the keys are? Carol thinks they are in the""",
        "correct": " drawer",
        "wrong": " table",
    })
    
    scenarios.append({
        "name": "Partial communication",
        "n_agents": 3,
        "prompt": """The document was in folder A.
Alice moved it to folder B. Alice told Bob.
Carol was not informed of any changes.
Where does Carol think the document is? Carol thinks it is in folder""",
        "correct": " A",
        "wrong": " B",
    })
    
    scenarios.append({
        "name": "Four agents - mixed knowledge",
        "n_agents": 4,
        "prompt": """The treasure was buried under the oak tree.
Alice told Bob she moved it under the pine tree.
Carol overheard Alice tell Bob.
Dave was not present for any of this.
Where does Dave think the treasure is? Dave thinks it is under the""",
        "correct": " oak",
        "wrong": " pine",
    })
    
    return scenarios


def generate_multi_turn_scenarios(n: int = 10) -> List[Dict]:
    """
    Generate scenarios with multiple belief updates.
    """
    scenarios = []
    
    scenarios.append({
        "name": "Two belief updates",
        "n_updates": 2,
        "prompt": """Alice put the cake in the fridge. Alice left.
Bob moved the cake to the counter. Bob told Carol about the move.
Carol then moved the cake to the pantry. Carol did not tell anyone.
Alice returned. Alice will look for the cake in the""",
        "correct": " fridge",
        "wrong": " pantry",
    })
    
    scenarios.append({
        "name": "Witnessed vs told",
        "n_updates": 2,
        "prompt": """The treasure was in the cave. Alice saw Bob hide the treasure in the forest.
Bob told Carol he hid the treasure in the mountain.
Where does Carol think the treasure is? Carol thinks it is in the""",
        "correct": " mountain",
        "wrong": " forest",
    })
    
    return scenarios


def get_all_higher_order_scenarios() -> List[Dict]:
    """Get all higher-order ToM scenarios combined."""
    all_scenarios = []
    all_scenarios.extend(generate_nested_belief_scenarios())
    all_scenarios.extend(generate_multi_domain_scenarios())
    all_scenarios.extend(generate_multi_agent_scenarios())
    all_scenarios.extend(generate_multi_turn_scenarios())
    return all_scenarios

