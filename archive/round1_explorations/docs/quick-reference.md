# Quick Reference

## One-Sentence Summary

We're using mechanistic interpretability to understand how language models internally represent and collaborate with other models and users—focusing on the circuits and representations that enable (or break) multi-agent coordination.

## Key Concepts

- **Collaboration Circuits**: Internal mechanisms (attention heads, activation patterns) that enable models to coordinate with each other
- **Agent Representations**: How models internally represent other agents' beliefs, goals, and states
- **User Representations**: How models build and maintain internal models of users
- **Representation Interference**: When different representations (user/agent/self) contaminate each other, causing collaboration failures

## Research Questions (TL;DR)

1. How do models represent other agents internally?
2. What circuits enable collaboration vs. cause failures?
3. Can we detect/prevent harmful behaviors (collusion, groupthink, deception) mechanistically?
4. Does representation interference explain collaboration breakdowns?

## Why It Matters

- **Safety**: Detect manipulation, collusion, and misalignment in multi-agent systems
- **Alignment**: Understand when alignment breaks in social contexts
- **Control**: Design interventions based on mechanistic understanding
- **Research**: Bridge behavioral multi-agent research with mechanistic interpretability

## Current State (Dec 2025)

**What we know**:
- ToM circuits exist and are identifiable
- User/agent representations are probeable
- Deception and misalignment have identifiable patterns
- Multi-agent behaviors are documented behaviorally

**What's missing**:
- Mechanistic understanding of multi-agent collaboration
- How representations interact and interfere
- Circuit-level interventions for multi-agent safety

## Our Approach

1. **Model organisms**: Start with small, controlled multi-agent scenarios
2. **Representation analysis**: Probe and decode user/agent representations
3. **Causal interventions**: Use patching, steering, ablation to test hypotheses
4. **Circuit discovery**: Identify collaboration mechanisms with automated tools

## Key Research Directions

- **ToM**: Lookback mechanisms for tracking beliefs
- **User modeling**: Demographic probes and latent belief decoders
- **Deception**: Universal motifs and circuit patterns
- **Multi-agent**: Mechanistic analysis of multi-agent LLM systems
- **Tools**: GIM for automated circuit discovery ([see references](references.md#verified-papers))

*For verified citations, see [references.md](references.md)*

## Success Looks Like

1. Can locate and characterize user/agent representations in multi-agent systems
2. Can show these representations are separable
3. Can show representation interference causally affects collaboration
4. Can use mechanistic knowledge to improve/fix collaboration

## Core Tools

| Tool | Use For | Link |
|------|---------|------|
| **TransformerLens** | Activation access, patching, circuit analysis | [GitHub](https://github.com/neelnanda-io/TransformerLens) |
| **nnsight** | Interventions on any model, remote access | [nnsight.net](https://nnsight.net/) |
| **SAELens** | Training sparse autoencoders | [GitHub](https://github.com/jbloomAus/SAELens) |
| **Neuronpedia** | Browse SAE features, circuit tracing | [neuronpedia.org](https://www.neuronpedia.org/) |
| **pyvene** | Causal interventions | [GitHub](https://github.com/stanfordnlp/pyvene) |

*See [getting-started.md](getting-started.md) for setup and first experiments.*

## Navigation

- **Getting Started**: [getting-started.md](getting-started.md) ⭐
- **Overview**: [README.md](README.md)
- **Research Questions**: [research-questions.md](research-questions.md)
- **Approach**: [approach.md](approach.md)
- **Thesis**: [thesis.md](thesis.md)
- **Related Work**: [related-work.md](related-work.md)
- **SOTA Overview**: [sota-overview.md](sota-overview.md)
- **References**: [references.md](references.md)

