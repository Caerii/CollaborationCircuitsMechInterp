# Collaboration Circuits: Mechanistic Interpretability of Multi-Agent LLM Systems

## Project Overview

This project investigates how language models internally represent and interact with other models, agents, and users—focusing on the mechanistic circuits that enable collaboration, coordination, and emergent behaviors in multi-agent systems.

## What We're Doing

We're applying mechanistic interpretability techniques to understand:
- **Circuits for collaboration** between models (with other models and with humans)
- **Internal representations** of users and other agents within models
- **Emergent behaviors** in multi-agent setups (coordination, deception, groupthink)
- **The mechanistic basis** of theory-of-mind, user modeling, and agent-agent interactions

## What We're Trying to Do

Our goal is to:

1. **Map collaboration circuits**: Identify the internal mechanisms (attention heads, activation patterns, feature directions) that enable models to coordinate and collaborate with each other and with users

2. **Understand agent and user representations**: Characterize how models internally represent:
   - Other agents' beliefs, goals, and mental states
   - User attributes, preferences, and identity
   - Their own state and awareness (e.g., "am I being evaluated?")

3. **Explain emergent behaviors**: Mechanistically explain phenomena like:
   - Groupthink and conformity in multi-agent systems
   - Deception and collusion between agents
   - Value drift and persuasion in agent interactions
   - Toxic agreement and consensus loops

4. **Develop interpretability tools for multi-agent systems**: Extend existing techniques (activation patching, probing, steering, circuit discovery) to multi-agent contexts

## Why This Matters

### For AI Safety

- **Risk identification**: Understanding how agents coordinate or collude helps detect dangerous emergent behaviors
- **Monitoring**: Probe-based detection of manipulation, deception, or misalignment in multi-agent systems
- **Intervention design**: Mechanistic knowledge enables targeted fixes (e.g., ablating "toxic agreement" circuits)

### For Alignment

- **Stability under influence**: Understanding whether aligned agents stay aligned when interacting with misaligned ones, and whether we can detect and prevent value drift mechanistically
- **Evaluation awareness**: Models can distinguish evaluation vs. deployment contexts—understanding this mechanism helps prevent sandbagging and deceptive behavior
- **Truthfulness**: Detecting when models are lying or colluding requires understanding their internal representations

### For Research

- **Novel frontier**: Multi-agent mechanistic interpretability is still in its infancy—there's low-hanging fruit
- **Pragmatic interpretability**: Focus on behaviors that matter for real-world deployment, not just toy models
- **Model biology**: Treat multi-agent systems as organisms with internal mechanisms to be dissected and understood

## Research Direction

Our approach combines:

1. **Theory-of-Mind circuits**: How do models track and represent other agents' beliefs and knowledge?
2. **User modeling**: How do models build and maintain internal representations of users?
3. **Collaboration mechanisms**: What circuits enable coordination, information sharing, and joint problem-solving?
4. **Safety-relevant behaviors**: Deception, manipulation, collusion, groupthink—and how to detect/intervene

## Current State of the Field (Dec 2025)

- **ToM representations**: Recent work shows models implement "lookback" mechanisms to track other agents' beliefs, with identifiable circuits
- **User representations**: Linear probes can extract user demographics; newer work uses decoders for richer latent user beliefs
- **Emergent behaviors**: Groupthink, collusion, and deception have been documented at the behavioral level, but mechanistic explanations are sparse (see [Mathew et al. 2024](references.md#verified-papers) for collusion case study)
- **Circuits for deception/misalignment**: Common motifs and transferable "misalignment directions" have been identified

*See [related-work.md](related-work.md) for more details and [references.md](references.md) for verified citations.*

**Key Gap**: Most multi-agent research is behavioral, not mechanistic. We aim to bridge this gap by opening up the black box and tracing how these behaviors emerge from model internals.

---

## Getting Started

**Ready to start implementing?** See [getting-started.md](getting-started.md) for:
- Core tools (TransformerLens, nnsight, SAELens, Neuronpedia)
- Development environment setup
- Concrete first experiments
- Phased development plan

## Documentation

| Document | Description |
|----------|-------------|
| [application-project.md](application-project.md) | **20-hour MATS project plan** ⭐ |
| [getting-started.md](getting-started.md) | Tools, setup, and first experiments |
| [quick-reference.md](quick-reference.md) | One-page summary |
| [research-questions.md](research-questions.md) | Full research agenda |
| [approach.md](approach.md) | Methodology and techniques |
| [thesis.md](thesis.md) | Core thesis and arguments |
| [neel-alignment.md](neel-alignment.md) | Alignment with Neel Nanda's interests |
| [related-work.md](related-work.md) | Field overview |
| [sota-overview.md](sota-overview.md) | Current state of the art |
| [references.md](references.md) | Verified citations and tools |

