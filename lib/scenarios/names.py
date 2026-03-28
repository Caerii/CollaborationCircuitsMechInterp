"""Novel name pools for stimulus generation.

Never use common names (Alice, Bob, Charlie) or common locations (drawer, basket, box)
to avoid triggering training priors from existing ToM benchmarks.
"""

import random

AGENT_NAMES = [
    "Zara", "Kael", "Priya", "Orin", "Lumi", "Dex", "Naia", "Voss",
    "Mira", "Talon", "Suki", "Ren", "Ione", "Cael", "Yuki", "Bram",
    "Petra", "Joss", "Wren", "Soren", "Elia", "Knox", "Thea", "Rune",
    "Lyra", "Ash", "Maren", "Zev", "Tova", "Quinn", "Sage", "Kai",
]

OBJECTS = [
    "marble", "figurine", "compass", "lantern", "crystal", "medallion",
    "feather", "pebble", "ring", "shell", "coin", "acorn", "ribbon",
    "whistle", "brooch", "chalk", "spool", "prism", "flask", "locket",
]

LOCATIONS = [
    "alcove", "cupboard", "cabinet", "shelf", "vault", "crate",
    "niche", "chest", "locker", "hutch", "recess", "compartment",
    "cubby", "hamper", "armoire", "bureau", "credenza", "wardrobe",
    "canister", "coffer",
]


def sample_names(n_agents: int = 2, seed: int | None = None) -> dict:
    """Sample non-overlapping names for a scenario.

    Returns dict with keys: agents (list), object (str), locations (list of 2).
    """
    rng = random.Random(seed)
    agents = rng.sample(AGENT_NAMES, n_agents)
    obj = rng.choice(OBJECTS)
    locs = rng.sample(LOCATIONS, 2)
    return {"agents": agents, "object": obj, "locations": locs}
