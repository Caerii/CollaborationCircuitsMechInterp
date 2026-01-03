# Research Approach

## Methodology

### 1. Model Organism Setup

Start with controlled, simplified scenarios:
- Small transformer models (7B or smaller) for tractability
- Well-defined multi-agent tasks (collaborative puzzles, debates, negotiations)
- Synthetic but realistic dialogue datasets with known ground truth

### 2. Representation Analysis

**Probing**:
- Train linear probes to detect user attributes, agent states, evaluation context
- Use decoders (LatentQA-style) for richer, structured representations
- Map where these representations appear across layers and token positions

**Visualization**:
- Use dimensionality reduction (PCA, t-SNE) to visualize representation spaces
- Identify clusters corresponding to different agents/users/personas
- Track how representations evolve across conversation turns

### 3. Causal Interventions

**Activation Patching/Resampling**:
- Test which parts of one agent's output causally affect another agent's behavior
- Identify specific attention heads or layers that mediate collaboration

**Steering**:
- Extract "directions" for user models, agent representations, etc.
- Add/subtract these directions to see causal effects on behavior
- Test conditional steering (only when a probe fires) to reduce side effects

**Ablation**:
- Remove specific components (heads, layers) to test their necessity
- Identify minimal circuits required for collaboration

### 4. Circuit Discovery

Use automated tools (GIM, attribution graphs) to identify:
- Critical components for multi-agent behaviors
- Cross-agent information flow pathways
- Circuits that generalize across different collaboration scenarios

## Experimental Design Principles

1. **Start simple, scale up**: Begin with toy scenarios, then move to more realistic ones
2. **Control for confounds**: Use paraphrase controls, off-distribution checks, baseline comparisons
3. **Causal verification**: Always test hypotheses with interventions, not just correlations
4. **Multiple models**: Validate findings across different model families/sizes when possible
5. **Reproducibility**: Clear experimental protocols, released code/models

## Key Techniques

- **Activation patching/resampling**: Test causal importance of specific activations
- **Linear probing**: Detect what information is encoded where
- **Activation steering**: Add/subtract vectors to change behavior
- **Circuit tracing**: Map causal pathways through the network
- **Automated circuit discovery**: Scale up analysis with gradient-based methods

