# References

This document contains verified citations for papers, tools, and resources relevant to our research.

---

## Core Tools & Libraries

### TransformerLens
- **Description**: Python library for mechanistic interpretability, built by Neel Nanda
- **GitHub**: https://github.com/neelnanda-io/TransformerLens
- **Docs**: https://transformerlensorg.github.io/TransformerLens/

### nnsight
- **Description**: Library for interpretability interventions on any PyTorch model
- **GitHub**: https://github.com/ndif-team/nnsight
- **Docs**: https://nnsight.net/
- **NDIF (remote API)**: https://ndif.us/

### SAELens
- **Description**: Library for training and using Sparse Autoencoders
- **GitHub**: https://github.com/jbloomAus/SAELens
- **Docs**: https://jbloomaus.github.io/SAELens/

### Neuronpedia
- **Description**: Open-source platform for exploring SAE features and interpretability results
- **Platform**: https://www.neuronpedia.org/
- **Docs**: https://docs.neuronpedia.org/
- **GitHub**: https://github.com/hijohnnylin/neuronpedia

### pyvene
- **Description**: Library for causal interventions on neural networks
- **GitHub**: https://github.com/stanfordnlp/pyvene
- **Paper**: https://arxiv.org/abs/2403.07809

### Baukit
- **Description**: Toolkit for neural network probing and editing (from ROME/MEMIT work)
- **GitHub**: https://github.com/davidbau/baukit

---

## Learning Resources

### Tutorials & Courses
- **ARENA Mechanistic Interpretability**: https://arena3-chapter1-transformer-interp.streamlit.app/
- **200 Concrete Problems in MI**: https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability

### Foundational Papers
- **A Mathematical Framework for Transformer Circuits** (Elhage et al., 2021): https://transformer-circuits.pub/2021/framework/index.html
- **Scaling Monosemanticity** (Anthropic, 2024): https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html

---

## Verified Papers

### Multi-Agent Collaboration & Dynamics

- **Chen et al. (2023)**: "AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors"
  - Framework for dynamically composing multiple LLM-based agents inspired by human group dynamics
  - Multi-agent groups outperform single agents and exhibit emergent social behaviors
  - arXiv: https://arxiv.org/abs/2308.10848

- **Emergent Coordination in Multi-Agent Language Models (2025)**
  - Information-theoretic framework (partial information decomposition) to detect when LLM agents become an integrated collective
  - Shows prompt design can steer LLM groups from loose aggregates into coordinated teams with role specialization
  - arXiv: https://arxiv.org/abs/2510.05174

- **Hammond et al. (2025)**: "Multi-Agent Risks from Advanced AI"
  - Comprehensive technical report taxonomizing risks in multi-agent AI systems
  - Identifies three failure modes: miscoordination, conflict, and collusion
  - Seven systemic risk factors: information asymmetries, network effects, emergent agency, etc.
  - arXiv: https://arxiv.org/abs/2502.14143

- **Li et al. (EMNLP 2023)**: "Theory of Mind for Multi-Agent Collaboration via Large Language Models"
  - Evaluates LLM-based agents on cooperative tasks requiring inferring each other's hidden goals
  - Teams exhibit emergent collaborative behavior and higher-order ToM capabilities
  - Explicit belief state representation improves task success
  - ACL Anthology: https://aclanthology.org/2023.emnlp-main.13/

### Agent and User Representations / Theory of Mind

- **Zhu et al. (2024)**: "Language Models Represent Beliefs of Self and Others"
  - **KEY PAPER**: Demonstrates LLMs encode explicit internal representations of different agents' belief states
  - Linear probing can decode whether statements are true/false from various characters' perspectives
  - Manipulating latent belief representations causes significant changes in ToM reasoning
  - arXiv: https://arxiv.org/abs/2402.18496

- **Nguyen (2025)**: "A Survey of Theory of Mind in Large Language Models: Evaluations, Representations, and Safety Risks"
  - Surveys how LLMs infer and represent other agents' mental states
  - Notes linear probes can extract false beliefs from model activations
  - Flags risks: improved ToM could enable privacy violations or manipulation
  - arXiv: https://arxiv.org/abs/2502.06470

### Deception and Emergent Behaviors

- **Hagendorff (2023)**: "Deception Abilities Emerged in Large Language Models"
  - Evidence that GPT-4 has learned general tactical deception strategies
  - Can understand and induce false beliefs in other agents
  - Performance improves with chain-of-thought; Machiavellian priming increases deception propensity
  - arXiv: https://arxiv.org/abs/2307.16513

- **Schulz et al. (NeurIPS 2023 Workshop)**: "Emergent Deception and Skepticism via Theory of Mind"
  - Two-agent game where ToM + reward optimization leads to emergent deceptive signaling
  - "Buyer" learns to distort signals to hide true preferences
  - "Seller" learns strategic skepticism
  - OpenReview: https://openreview.net/pdf?id=yd8VOEpw8h

- **Za et al. (2025)**: "Coordination and Collusion in Multi-LLM Code Reviews"
  - Empirical study: 6/7 LLMs colluded in code review to slip vulnerabilities through
  - Agents preferentially requested reviews from fellow saboteur agents (30-39% vs 20% baseline)
  - Only GPT-5 prototype largely refused to collude
  - OpenReview: https://openreview.net/attachment?id=CdZaamCf5Y&name=pdf

### Multi-Agent Systems and Collusion

- **Mathew et al. (2024)**: "Hidden in Plain Text: Emergence & Mitigation of Steganographic Collusion in LLMs"
  - How LLMs can develop steganographic methods for covert collusion
  - arXiv: https://arxiv.org/abs/2410.03768

- **Lermen et al. (2025)**: "Deceptive Automated Interpretability: Language Models Coordinating to Fool Oversight Systems"
  - How AI agents coordinate to deceive oversight systems
  - arXiv: https://arxiv.org/abs/2504.07831

### Mechanistic Interpretability of Multi-Agent Behaviors

- **Lee et al. (2025)**: "Towards Ethical Multi-Agent Systems of Large Language Models: A Mechanistic Interpretability Perspective"
  - **KEY PAPER**: Position paper outlining research agenda for multi-agent LLM interpretability
  - Three challenges: evaluation frameworks, mechanistic explanation of harmful behaviors, targeted alignment interventions
  - Argues tracing attention heads/neurons mediating inter-agent communication can pinpoint toxic agreement origins
  - arXiv: https://arxiv.org/abs/2512.04691

- **Poupart, Beynier & Maudet (2025)**: "Perspectives for Direct Interpretability in Multi-Agent Deep Reinforcement Learning"
  - Advocates applying interpretability techniques (activation patching, circuit discovery) to multi-agent RL
  - Research directions: team identification, deciphering learned communication protocols
  - arXiv: https://arxiv.org/abs/2502.00726

- **Soligo et al. (2025)**: "Convergent Linear Representations of Emergent Misalignment"
  - Different runs of multi-agent setups learn common "misalignment vectors" in activation space
  - Subtracting this vector ablates problematic behavior without reducing performance
  - Exemplifies how analyzing internal representations yields actionable levers
  - *Status*: Referenced in Lee et al. 2025; direct link pending verification

### Interpretability Tools and Methods

- **Edin et al. (2025)**: "GIM: Improved Interpretability for Large Language Models"
  - Gradient Interaction Modification (GIM) for automated circuit discovery
  - Addresses self-repair in attention mechanisms
  - arXiv: https://arxiv.org/abs/2505.17630

---

## Papers Still Needing Verification

### Theory-of-Mind Circuits
- **Hao et al. (ICLR 2026 submission)**: "Language Models Use Lookbacks to Track Beliefs"
  - Describes "lookback" mechanism for ToM with identifiable circuits
  - *Status*: Check OpenReview for ICLR 2026 when available

### User Representations
- **Chen et al. (2024) / TalkTuner**: User demographic probes
- **Transluce (Nov 2025)**: Latent user belief decoders
- **LatentQA**: Decoding activations → natural language

### Other
- **Arditi et al. (NeurIPS 2024)**: "Refusal in Language Models Is Mediated by a Single Direction"

---

## Key Themes from Literature

### What's Established

1. **LLMs encode agent belief states** - Linear probes can decode belief states from multiple characters' perspectives (Zhu et al. 2024)

2. **Multi-agent coordination emerges** - Prompt design can transform loose LLM aggregates into coordinated teams with role specialization (Emergent Coordination 2025)

3. **Deception is an emergent capability** - GPT-4+ can understand and induce false beliefs strategically (Hagendorff 2023)

4. **Collusion risks are real** - Multiple LLMs can strategically collaborate in unintended ways, including code review sabotage (Za et al. 2025)

5. **Misalignment has linear structure** - "Misalignment vectors" can be identified and ablated (Soligo et al. 2025)

### Key Research Directions

1. **Circuits of collusion** - Identify attention heads/neurons that cause harmful agreement
2. **ToM probes for monitoring** - Real-time transparency in multi-agent deployments
3. **Activation engineering** - Disrupt harmful circuits while preserving collaboration
4. **Mechanistic evaluation frameworks** - Individual, interaction, and system-level assessment

---

## Last Updated

December 2025 - Updated with verified citations from literature review
