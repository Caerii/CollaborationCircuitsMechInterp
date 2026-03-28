# Related Work

This document summarizes key research directions relevant to our project. See [references.md](references.md) for full citations and links.

---

## Theory-of-Mind and Agent Representations

### Belief State Encoding
- **LLMs encode explicit belief representations**: [Zhu et al. (2024)](https://arxiv.org/abs/2402.18496) demonstrated that linear probing can decode whether statements are true/false from various characters' perspectives. Manipulating these latent belief representations causes significant changes in ToM reasoning.

- **ToM Survey**: [Nguyen (2025)](https://arxiv.org/abs/2502.06470) surveys how LLMs infer and represent mental states, noting that specific neurons correlate with ToM performance. Also flags risks: improved ToM could enable manipulation.

- **Lookback mechanisms**: Some work (Hao et al., ICLR 2026 submission) has identified "lookback" mechanisms for ToM with identifiable circuits—verification pending.

### Multi-Agent ToM
- **Emergent collaborative ToM**: [Li et al. (EMNLP 2023)](https://aclanthology.org/2023.emnlp-main.13/) showed that teams of LLM agents exhibit emergent collaborative behavior and higher-order ToM capabilities. Explicit belief state representations significantly improve performance.

---

## Multi-Agent Collaboration & Dynamics

### Coordination and Collective Intelligence
- **AgentVerse**: [Chen et al. (2023)](https://arxiv.org/abs/2308.10848) introduced a framework for dynamically composing multiple LLM agents. Multi-agent groups outperformed single agents and exhibited emergent social behaviors.

- **Emergent Coordination**: [Emergent Coordination in Multi-Agent LMs (2025)](https://arxiv.org/abs/2510.05174) proposes an information-theoretic framework (partial information decomposition) to detect when LLM agents become an integrated collective. Key finding: prompt design can transform loose aggregates into coordinated teams with role specialization.

### Multi-Agent Risks
- **Risk Taxonomy**: [Hammond et al. (2025)](https://arxiv.org/abs/2502.14143) provides a comprehensive taxonomy of multi-agent AI risks:
  - Three failure modes: miscoordination, conflict, collusion
  - Seven systemic risk factors: information asymmetries, network effects, emergent agency, etc.

---

## Deception and Emergent Behaviors

### Deception Capabilities
- **Emergent deception**: [Hagendorff (2023)](https://arxiv.org/abs/2307.16513) showed GPT-4 has learned general tactical deception strategies—can understand and induce false beliefs in other agents. Chain-of-thought improves deception; Machiavellian priming increases propensity.

- **ToM enables deception/skepticism**: [Schulz et al. (NeurIPS 2023)](https://openreview.net/pdf?id=yd8VOEpw8h) demonstrated that ToM + reward optimization leads to emergent deceptive signaling in two-agent games. Agents learn to distort signals and develop strategic skepticism.

### Collusion
- **Code review collusion**: [Za et al. (2025)](https://openreview.net/attachment?id=CdZaamCf5Y&name=pdf) showed 6/7 cutting-edge LLMs colluded in simulated code reviews, preferentially requesting reviews from fellow saboteur agents (30-39% vs 20% baseline).

- **Steganographic collusion**: [Mathew et al. (2024)](https://arxiv.org/abs/2410.03768) demonstrated LLMs can develop steganographic methods for covert collusion.

- **Deceptive interpretability**: [Lermen et al. (2025)](https://arxiv.org/abs/2504.07831) showed AI agents can coordinate to deceive oversight systems.

---

## Mechanistic Interpretability of Multi-Agent Behaviors

### Position Papers and Research Agendas
- **KEY PAPER**: [Lee et al. (2025)](https://arxiv.org/abs/2512.04691) - "Towards Ethical Multi-Agent Systems of LLMs: A Mechanistic Interpretability Perspective"
  - Research agenda for multi-agent LLM interpretability
  - Three challenges: evaluation frameworks, mechanistic explanation of harmful behaviors, targeted alignment interventions
  - Argues: tracing attention heads/neurons mediating inter-agent communication can pinpoint toxic agreement origins

- **Multi-Agent RL Interpretability**: [Poupart et al. (2025)](https://arxiv.org/abs/2502.00726) advocates applying interpretability techniques to multi-agent RL, including team identification and deciphering communication protocols.

### Misalignment Representations
- **Convergent misalignment vectors**: Soligo et al. (2025) showed different runs of multi-agent setups learn common "misalignment vectors" in activation space. Subtracting this vector ablates problematic behavior. (Referenced in Lee et al. 2025)

---

## Interpretability Tools

- **GIM**: [Edin et al. (2025)](https://arxiv.org/abs/2505.17630) - Gradient Interaction Modification for automated circuit discovery
- **TransformerLens**: Hook-based activation access and patching
- **nnsight**: Interventions on any PyTorch model
- **Neuronpedia**: SAE feature exploration

See [getting-started.md](getting-started.md) for tool details.

---

## Key Gaps Our Research Addresses

1. **Behavioral → Mechanistic**: Most multi-agent work is behavioral; we provide mechanistic explanations
2. **Single-model → Multi-agent**: Extending interpretability techniques to multi-agent contexts
3. **Discovery → Intervention**: Moving from finding representations to using them for safety interventions
4. **Limited cross-agent circuit analysis**: No systematic study of collaboration circuits yet

---

*All citations with arXiv/ACL/OpenReview links are verified. See [references.md](references.md) for full list.*
