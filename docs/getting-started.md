# Getting Started: Implementation Guide

This guide covers the tools, resources, and concrete first steps for tackling mechanistic interpretability of multi-agent LLM systems.

## Core Tools & Libraries

### 1. TransformerLens

**What**: Python library for mechanistic interpretability research, built by Neel Nanda.

**Use for**: Hook-based activation access, attention pattern analysis, activation patching, circuit analysis.

**Links**:
- GitHub: https://github.com/neelnanda-io/TransformerLens
- Docs: https://transformerlensorg.github.io/TransformerLens/
- Tutorial: https://arena3-chapter1-transformer-interp.streamlit.app/

**Install**:
```bash
pip install transformer-lens
```

**Key Features**:
- Easy access to all intermediate activations via hooks
- Built-in support for many models (GPT-2, GPT-Neo, Llama, etc.)
- Activation patching utilities
- Attention pattern visualization

---

### 2. nnsight

**What**: Library for interpretability interventions on any PyTorch model, including large models via NDIF remote access.

**Use for**: Activation patching, steering, running interventions on models too large for local hardware.

**Links**:
- GitHub: https://github.com/ndif-team/nnsight
- Docs: https://nnsight.net/
- NDIF (remote API): https://ndif.us/

**Install**:
```bash
pip install nnsight
```

**Key Features**:
- Works with any PyTorch model (not just supported architectures)
- Remote access to large models (70B+) via NDIF
- Tracing-based intervention API
- Batched interventions

---

### 3. SAELens

**What**: Library for training and using Sparse Autoencoders (SAEs) for interpretability.

**Use for**: Training SAEs on model activations, extracting interpretable features.

**Links**:
- GitHub: https://github.com/jbloomAus/SAELens
- Docs: https://jbloomaus.github.io/SAELens/

**Install**:
```bash
pip install sae-lens
```

**Key Features**:
- Train SAEs on any transformer model
- Integration with TransformerLens
- Pre-trained SAEs available

---

### 4. Neuronpedia

**What**: Open-source platform for exploring and sharing interpretability results, especially SAE features.

**Use for**: Browsing pre-computed SAE features, circuit tracing, model steering, sharing results.

**Links**:
- Platform: https://www.neuronpedia.org/
- Docs: https://docs.neuronpedia.org/
- GitHub: https://github.com/hijohnnylin/neuronpedia

**Key Features**:
- Browse SAE features with explanations and activations
- Circuit tracing tools
- Model steering interface
- API for programmatic access

---

### 5. pyvene

**What**: Library for causal interventions on neural networks.

**Use for**: Activation patching, distributed alignment search, causal tracing.

**Links**:
- GitHub: https://github.com/stanfordnlp/pyvene
- Paper: https://arxiv.org/abs/2403.07809

**Install**:
```bash
pip install pyvene
```

---

### 6. Baukit (from ROME/MEMIT)

**What**: Toolkit for neural network probing and editing.

**Use for**: Tracing, causal mediation analysis, model editing.

**Links**:
- GitHub: https://github.com/davidbau/baukit

---

## Development Environment Setup

### Recommended Setup

```bash
# Create virtual environment
python -m venv mech-interp-env
source mech-interp-env/bin/activate  # Linux/Mac
# or: mech-interp-env\Scripts\activate  # Windows

# Core libraries
pip install torch transformers datasets
pip install transformer-lens
pip install nnsight
pip install sae-lens

# Visualization and analysis
pip install plotly matplotlib seaborn
pip install scikit-learn  # for probes, PCA, etc.
pip install einops  # tensor manipulation

# Notebooks
pip install jupyter jupyterlab

# Optional: for remote large model access
pip install ndif-client
```

### Hardware Considerations

| Model Size | Minimum VRAM | Recommended |
|------------|--------------|-------------|
| GPT-2 (124M) | 4GB | Local GPU |
| GPT-2-XL (1.5B) | 8GB | Local GPU |
| Llama-3-8B | 24GB | A100/H100 or NDIF |
| Llama-3-70B | 140GB+ | Use NDIF remote |

---

## First Experiments: Concrete Starting Points

### Experiment 1: User Representation Probing (Week 1)

**Goal**: Determine if/where models represent user attributes.

**Setup**:
1. Create synthetic conversations with varied user personas (formal/casual, expert/novice)
2. Extract activations at each layer using TransformerLens
3. Train linear probes to classify user attributes
4. Analyze which layers encode user information

```python
# Pseudo-code structure
import transformer_lens as tl
from sklearn.linear_model import LogisticRegression

model = tl.HookedTransformer.from_pretrained("gpt2")

# Generate prompts with different user personas
formal_prompts = ["Dear Assistant, could you please..."]
casual_prompts = ["hey can u help me..."]

# Collect activations
formal_acts = model.run_with_cache(formal_prompts)[1]
casual_acts = model.run_with_cache(casual_prompts)[1]

# Train probe at each layer
for layer in range(model.cfg.n_layers):
    X = torch.cat([formal_acts[f"blocks.{layer}.hook_resid_post"], 
                   casual_acts[f"blocks.{layer}.hook_resid_post"]])
    y = [0]*len(formal_prompts) + [1]*len(casual_prompts)
    probe = LogisticRegression().fit(X.mean(dim=1).numpy(), y)
    print(f"Layer {layer}: {probe.score(...)}")
```

---

### Experiment 2: Multi-Agent Activation Analysis (Week 2)

**Goal**: Analyze how models represent "self" vs "other agent" in multi-agent dialogues.

**Setup**:
1. Create dialogues where model plays Agent A responding to Agent B
2. Compare activations when processing own vs other agent's text
3. Look for separation in representation space

**Key questions**:
- Are there distinct representations for "my turn" vs "their turn"?
- How do models track conversational roles?

---

### Experiment 3: Evaluation Awareness Detection (Week 2-3)

**Goal**: Detect if models have different internal representations when "being evaluated."

**Setup**:
1. Create prompts with/without evaluation context ("This is a test...")
2. Probe for evaluation awareness
3. If found, test causal effect via steering

---

### Experiment 4: Collaboration Circuit Tracing (Week 3-4)

**Goal**: Identify circuits responsible for collaborative vs competitive behavior.

**Setup**:
1. Create minimal collaborative tasks (e.g., jointly solving a puzzle)
2. Use activation patching to find components critical for cooperation
3. Test if ablating these components breaks collaboration

---

## Using Neuronpedia Effectively

### For Feature Discovery

1. Browse SAE features for concepts like "user", "agent", "evaluation"
2. Look at activation examples to understand feature semantics
3. Use features as a starting point for circuit analysis

### For Circuit Tracing

1. Identify a behavior of interest
2. Use Neuronpedia's circuit tools to trace information flow
3. Validate with local experiments

### API Access

```python
import requests

# Example: Search for features
response = requests.get(
    "https://www.neuronpedia.org/api/search",
    params={"model": "gpt2-small", "query": "user"}
)
```

---

## Phased Development Plan

### Phase 1: Foundation (Weeks 1-2)
- [ ] Set up development environment
- [ ] Run basic TransformerLens tutorials
- [ ] Explore Neuronpedia for relevant SAE features
- [ ] Implement Experiment 1 (user representation probing)

### Phase 2: Multi-Agent Analysis (Weeks 3-4)
- [ ] Create multi-agent dialogue datasets
- [ ] Implement Experiment 2 (self vs other representations)
- [ ] Implement Experiment 3 (evaluation awareness)
- [ ] Document findings, iterate on hypotheses

### Phase 3: Circuit Discovery (Weeks 5-6)
- [ ] Implement Experiment 4 (collaboration circuits)
- [ ] Use activation patching to identify critical components
- [ ] Begin causal intervention experiments

### Phase 4: Synthesis & Steering (Weeks 7-8)
- [ ] Synthesize findings into coherent story
- [ ] Develop steering interventions based on findings
- [ ] Test robustness across models/scenarios
- [ ] Write up results

---

## Key Resources for Learning

### Tutorials & Courses
- **ARENA Mechanistic Interpretability**: https://arena3-chapter1-transformer-interp.streamlit.app/
- **Neel Nanda's YouTube**: Comprehensive mech interp tutorials
- **200 Concrete Problems in MI**: https://www.alignmentforum.org/posts/LbrPTJ4fmABEdEnLf/200-concrete-open-problems-in-mechanistic-interpretability

### Papers to Read First
- "A Mathematical Framework for Transformer Circuits" (Elhage et al., 2021)
- "Toward A Mathematical Framework for Computation in Superposition" (Anthropic, 2024)
- "Scaling Monosemanticity" (Anthropic, 2024)

### Communities
- **MATS Discord**: For MATS applicants and scholars
- **EleutherAI Discord**: Active interpretability channels
- **Alignment Forum**: Research discussion

---

## Quick Decision Tree

**"Should I use Neuronpedia?"**
→ YES if: exploring known SAE features, sharing results, quick feature search
→ NO if: training custom SAEs, working with unsupported models, need full control

**"Should I use TransformerLens or nnsight?"**
→ **TransformerLens**: Cleaner API, better for learning, limited model support
→ **nnsight**: Any PyTorch model, remote access to large models, more flexible

**"Should I use SAELens?"**
→ YES if: training your own SAEs, need custom feature analysis
→ NO if: just using pre-trained features (use Neuronpedia instead)

---

## Next Steps

1. **Today**: Set up environment, run first TransformerLens tutorial
2. **This week**: Complete Experiment 1 (user probing)
3. **Next week**: Start multi-agent analysis
4. **Ongoing**: Document everything, iterate based on findings

See [approach.md](approach.md) for detailed methodology and [research-questions.md](research-questions.md) for the full research agenda.

