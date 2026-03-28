"""
Novel Name Generator

Generates unfamiliar names for agents, locations, and objects to break
learned priors from training data.

Key insight from PROPER_METHODOLOGY.md:
- Using familiar names like "Alice", "Bob", "drawer", "basket" allows
  models to rely on training data patterns rather than actual reasoning
- Novel names force the model to process the actual scenario content
- This is critical for distinguishing ToM from memorization/heuristics
"""

import random
from typing import List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class NameSet:
    """A complete set of novel names for a scenario."""
    agents: List[str]
    locations: List[str]
    objects: List[str]
    
    def get_agent(self, idx: int = 0) -> str:
        return self.agents[idx % len(self.agents)]
    
    def get_location(self, idx: int = 0) -> str:
        return self.locations[idx % len(self.locations)]
    
    def get_object(self, idx: int = 0) -> str:
        return self.objects[idx % len(self.objects)]


class NovelNameGenerator:
    """
    Generate procedural names that don't appear in common training data.
    
    Uses patterns designed to be:
    - Pronounceable but unfamiliar
    - Unlikely to have semantic associations
    - Consistent within a scenario
    
    Example:
        gen = NovelNameGenerator(seed=42)
        names = gen.generate_set()
        
        print(names.agents)     # ['Zyx', 'Qar']
        print(names.locations)  # ['Container-Alpha', 'Zone-Beta']
        print(names.objects)    # ['orb', 'cube']
    """
    
    # Agent name components - alien/fantasy style
    AGENT_PREFIXES = [
        "Zyx", "Qar", "Blip", "Vorn", "Krix", "Thex", "Norv", "Plex",
        "Jax", "Rix", "Dax", "Zeph", "Mox", "Nyx", "Vex", "Brix"
    ]
    
    AGENT_SUFFIXES = [
        "", "on", "ix", "ar", "os", "ax", "el", "is"
    ]
    
    # Location patterns - technical/container style
    LOCATION_PREFIXES = [
        "Container", "Zone", "Area", "Unit", "Sector", "Module",
        "Chamber", "Vault", "Bin", "Cell", "Pod", "Bay"
    ]
    
    LOCATION_SUFFIXES = [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta",
        "Omega", "Sigma", "Theta", "Kappa", "Lambda", "Rho"
    ]
    
    # Alternative location style - made-up words
    LOCATION_WORDS = [
        "gloxx", "freln", "skorp", "drenn", "plonk", "blorx",
        "spliff", "grunk", "fleem", "blort", "snarg", "pliff"
    ]
    
    # Object names - simple geometric or abstract
    OBJECT_NAMES = [
        "orb", "cube", "prism", "disc", "shard", "node",
        "token", "rune", "glyph", "sigil", "mark", "seal"
    ]
    
    # Alternative object style - nonsense words
    OBJECT_WORDS = [
        "glorp", "bleem", "skrib", "froon", "plonx", "drib",
        "snork", "plib", "grex", "flob", "zib", "krep"
    ]
    
    def __init__(
        self,
        seed: Optional[int] = None,
        use_technical_locations: bool = True,
        use_geometric_objects: bool = True
    ):
        """
        Initialize generator.
        
        Args:
            seed: Random seed for reproducibility
            use_technical_locations: Use "Container-Alpha" style vs "gloxx" style
            use_geometric_objects: Use "orb", "cube" style vs "glorp" style
        """
        self.rng = random.Random(seed)
        self.use_technical_locations = use_technical_locations
        self.use_geometric_objects = use_geometric_objects
        
        # Track used names to avoid repetition
        self._used_agents = set()
        self._used_locations = set()
        self._used_objects = set()
    
    def reset(self):
        """Reset tracking of used names."""
        self._used_agents.clear()
        self._used_locations.clear()
        self._used_objects.clear()
    
    def _generate_agent_name(self) -> str:
        """Generate a single agent name."""
        for _ in range(100):  # Avoid infinite loop
            prefix = self.rng.choice(self.AGENT_PREFIXES)
            suffix = self.rng.choice(self.AGENT_SUFFIXES)
            name = prefix + suffix
            
            if name not in self._used_agents:
                self._used_agents.add(name)
                return name
        
        # Fallback: add number
        base = self.rng.choice(self.AGENT_PREFIXES)
        return f"{base}{len(self._used_agents)}"
    
    def _generate_location_name(self) -> str:
        """Generate a single location name."""
        for _ in range(100):
            if self.use_technical_locations:
                prefix = self.rng.choice(self.LOCATION_PREFIXES)
                suffix = self.rng.choice(self.LOCATION_SUFFIXES)
                name = f"{prefix}-{suffix}"
            else:
                name = self.rng.choice(self.LOCATION_WORDS)
            
            if name not in self._used_locations:
                self._used_locations.add(name)
                return name
        
        # Fallback
        return f"location-{len(self._used_locations)}"
    
    def _generate_object_name(self) -> str:
        """Generate a single object name."""
        for _ in range(100):
            if self.use_geometric_objects:
                name = self.rng.choice(self.OBJECT_NAMES)
            else:
                name = self.rng.choice(self.OBJECT_WORDS)
            
            if name not in self._used_objects:
                self._used_objects.add(name)
                return name
        
        return f"object-{len(self._used_objects)}"
    
    def generate_agents(self, n: int = 2) -> List[str]:
        """Generate n unique agent names."""
        return [self._generate_agent_name() for _ in range(n)]
    
    def generate_locations(self, n: int = 2) -> List[str]:
        """Generate n unique location names."""
        return [self._generate_location_name() for _ in range(n)]
    
    def generate_objects(self, n: int = 1) -> List[str]:
        """Generate n unique object names."""
        return [self._generate_object_name() for _ in range(n)]
    
    def generate_set(
        self,
        n_agents: int = 2,
        n_locations: int = 2,
        n_objects: int = 1
    ) -> NameSet:
        """
        Generate a complete set of novel names for a scenario.
        
        Args:
            n_agents: Number of agent names
            n_locations: Number of location names
            n_objects: Number of object names
            
        Returns:
            NameSet with agents, locations, objects
        """
        return NameSet(
            agents=self.generate_agents(n_agents),
            locations=self.generate_locations(n_locations),
            objects=self.generate_objects(n_objects)
        )
    
    def generate_sets(self, n: int, **kwargs) -> List[NameSet]:
        """Generate n independent name sets."""
        self.reset()
        return [self.generate_set(**kwargs) for _ in range(n)]


# Convenience instances
_default_generator = NovelNameGenerator(seed=42)


def get_novel_agents(n: int = 2, seed: Optional[int] = None) -> List[str]:
    """Get n novel agent names."""
    gen = NovelNameGenerator(seed=seed) if seed else _default_generator
    gen.reset()
    return gen.generate_agents(n)


def get_novel_locations(n: int = 2, seed: Optional[int] = None) -> List[str]:
    """Get n novel location names."""
    gen = NovelNameGenerator(seed=seed) if seed else _default_generator
    gen.reset()
    return gen.generate_locations(n)


def get_novel_objects(n: int = 1, seed: Optional[int] = None) -> List[str]:
    """Get n novel object names."""
    gen = NovelNameGenerator(seed=seed) if seed else _default_generator
    gen.reset()
    return gen.generate_objects(n)


def get_name_set(seed: Optional[int] = None, **kwargs) -> NameSet:
    """Get a complete name set for a scenario."""
    gen = NovelNameGenerator(seed=seed) if seed else _default_generator
    gen.reset()
    return gen.generate_set(**kwargs)

