"""
Scenario Generation for Rigorous ToM and Collaboration Testing

RECOMMENDED: Use the simplified templates module for most cases:
    from scenarios.templates import generate_n_scenarios, TOM_TEMPLATES

Full modules (for advanced use):
    NovelNameGenerator: Generate unfamiliar names to break learned priors
    CounterbalancedScenarioSet: Enforce 8-scenario design per task
    ToMScenarioGenerator: Theory of Mind scenarios (FB, TB, Communication)
    MultiAgentScenarioGenerator: Multi-agent belief tracking
    DeceptionScenarioGenerator: Lie detection and trust calibration
    CooperationScenarioGenerator: Prisoner's dilemma, commons, negotiation
"""

# Simplified API (recommended)
from .templates import (
    generate_n_scenarios,
    generate_counterbalanced_8,
    generate_scenario,
    TOM_TEMPLATES,
    MULTI_AGENT_TEMPLATES,
    COOPERATION_TEMPLATES,
    get_novel_names,
)

# Full modules (for advanced use)
from .novel_names import NovelNameGenerator
from .counterbalancing import CounterbalancedScenarioSet, generate_counterbalanced_set
from .tom_extended import ToMScenarioGenerator
from .multi_agent import MultiAgentScenarioGenerator
from .deception import DeceptionScenarioGenerator
from .cooperation import CooperationScenarioGenerator

# Higher-order ToM (2nd/3rd order, multi-domain, multi-agent)
from .higher_order_tom import (
    generate_nested_belief_scenarios,
    generate_multi_domain_scenarios,
    generate_multi_agent_scenarios,
    generate_multi_turn_scenarios,
    get_all_higher_order_scenarios,
)

# Standard benchmarks (ToMi, FANToM)
from .benchmarks import (
    get_tomi_scenarios,
    get_fantom_scenarios,
    get_true_belief_controls,
    get_all_benchmarks,
    format_benchmark_prompt,
    TOMI_SCENARIOS,
    FANTOM_SCENARIOS,
)

# Explicit vs Implicit ToM
from .explicit_implicit import (
    generate_explicit_implicit_pairs,
    generate_bridging_phrase_variants,
    get_explicit_implicit_scenarios,
    get_bridging_phrase_tests,
)

# Robustness testing (verbs, styles, languages)
from .robustness import (
    generate_verb_robustness_scenarios,
    generate_style_robustness_scenarios,
    generate_multilingual_scenarios,
    get_verb_robustness,
    get_style_robustness,
    get_multilingual_scenarios,
    get_all_robustness_scenarios,
    VERB_CATEGORIES,
    PROMPT_STYLES,
)

__all__ = [
    # Simple API
    "generate_n_scenarios",
    "generate_counterbalanced_8",
    "generate_scenario",
    "TOM_TEMPLATES",
    "MULTI_AGENT_TEMPLATES",
    "COOPERATION_TEMPLATES",
    "get_novel_names",
    # Full modules
    "NovelNameGenerator",
    "CounterbalancedScenarioSet",
    "generate_counterbalanced_set",
    "ToMScenarioGenerator",
    "MultiAgentScenarioGenerator",
    "DeceptionScenarioGenerator",
    "CooperationScenarioGenerator",
    # Higher-order ToM
    "generate_nested_belief_scenarios",
    "generate_multi_domain_scenarios",
    "generate_multi_agent_scenarios",
    "generate_multi_turn_scenarios",
    "get_all_higher_order_scenarios",
    # Benchmarks
    "get_tomi_scenarios",
    "get_fantom_scenarios",
    "get_true_belief_controls",
    "get_all_benchmarks",
    "format_benchmark_prompt",
    "TOMI_SCENARIOS",
    "FANTOM_SCENARIOS",
    # Explicit/Implicit
    "generate_explicit_implicit_pairs",
    "generate_bridging_phrase_variants",
    "get_explicit_implicit_scenarios",
    "get_bridging_phrase_tests",
    # Robustness
    "generate_verb_robustness_scenarios",
    "generate_style_robustness_scenarios",
    "generate_multilingual_scenarios",
    "get_verb_robustness",
    "get_style_robustness",
    "get_multilingual_scenarios",
    "get_all_robustness_scenarios",
    "VERB_CATEGORIES",
    "PROMPT_STYLES",
]

