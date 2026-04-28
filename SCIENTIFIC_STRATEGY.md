# Scientific Strategy: Epistemic-State Circuits

This project should make narrower, causal claims about epistemic-state tracking,
not broad claims that a model "has Theory of Mind."

## Research Question

What mechanisms let a language model maintain, confuse, or override separate
representations of:

- reality: where the object actually is
- belief: where an agent thinks the object is
- knowledge: what an agent knows because they observed or were told something
- perspective: which agent or self-perspective a question asks about

The strongest target is belief-reality separation and collapse: when the model
answers from the agent's belief versus when it defaults to reality.

## Literature Position

The behavioral ToM literature is mixed. Kosinski-style false-belief batteries show
that modern LLMs can pass many classic tasks, while stress tests such as Shapira
et al. and Ullman-style trivial alterations show brittle performance and heuristic
dependence. Hu, Sosa, and Ullman argue that the field often conflates two
questions: whether models match human behavior and whether they use human-like or
otherwise faithful computations.

The recent KaBLE result is especially relevant: models can struggle to distinguish
belief, knowledge, and fact, with large gaps between first-person and third-person
false beliefs. This project should therefore treat "belief," "knowledge," and
"reality" as separate experimental variables.

Mechanistic interpretability also requires caution. Attribution graphs and
SAE/transcoder features are hypothesis generators, not final evidence. Their own
literature emphasizes reconstruction error, missing attention/QK mechanisms,
inactive or inhibitory features, graph complexity, and the need for intervention
validation.

## Claim Ladder

Claims should advance only when the previous rung is satisfied:

1. Behavioral contrast: a controlled task family produces reliable success/failure
   differences.
2. Local mechanism: attribution or probing yields a candidate internal mechanism
   that explains the contrast.
3. Causal validation: interventions on that mechanism change the belief-vs-reality
   logit margin in the predicted direction.
4. Stimulus generalization: the intervention works across held-out names, objects,
   location orders, and surface forms.
5. Construct generalization: the mechanism extends beyond Sally-Anne to belief
   update, knowledge attribution, first-person belief, and multi-agent binding.
6. Cross-model homology: analogous functional mechanisms appear in at least two
   model families, even if layer/head identities differ.

Anything below rung 3 is exploratory.

## Primary Metric

Use a continuous contrast whenever possible:

```text
belief_reality_margin = logit(belief_location) - logit(reality_location)
```

For false-belief questions, a positive margin favors belief-based answering. A
negative margin favors reality bias. This metric is better for circuit work than
free-form accuracy because it gives patching, attribution, and steering methods a
stable target.

## High-Level Causal Model

Before interpreting any graph, use a simple symbolic model as the hypothesized
computation:

```text
observed_move(agent, object, loc2)
absent(agent, move_event)
belief(agent, object) = loc1 if absent else loc2
query_type in {belief, reality}
answer = belief(agent, object) if query_type asks mental state else reality(object)
```

Mechanistic results should be phrased as partial evidence that a model realizes
parts of this causal model, not as evidence for a full human-like ToM faculty.

## Experimental Phases

### Phase 1: Lock the Behavioral Contrast

Run n=100 per primary condition on the primary model:

- false belief
- true belief
- reality check
- explicit belief question
- first-person false belief
- communication update
- two-agent conflicting beliefs

Every item must include heuristic metadata: first mention, recency, reality, and
explicit-statement baselines.

### Phase 2: Thin-Slice Circuit Tracing

Use matched examples where:

- true-belief controls are correct
- false-belief cases split into correct and reality-biased answers
- location order and surface form are counterbalanced

For each graph, record:

- replacement score and completeness score
- top feature contributors to the target logit or contrast target
- reconstruction-error contribution
- overlap of features between correct and failed false-belief cases
- signed contribution toward belief versus reality

### Phase 3: Intervention Validation

For each candidate circuit or feature group:

- Necessity: suppress/noise it and test whether false-belief margin collapses.
- Sufficiency: preserve/denoise it under corruption and test whether margin recovers.
- Specificity: confirm smaller effects on reality checks and irrelevant controls.
- Rescue: amplify belief-path features in reality-bias failures.
- Random controls: match layer, attribution magnitude, activation magnitude, and
  feature count.

Both noising and denoising are required because redundant OR-like mechanisms can
make a real circuit look unnecessary under one-sided tests.

### Phase 4: Cross-Construct and Cross-Model Tests

Do not require identical heads or layers across models. Require functional
homology:

- same behavioral dissociation
- same high-level variable recoverability
- same intervention direction
- analogous graph roles such as agent binding, observation/presence, belief
  content, reality update, and answer-policy selection

## Near-Term Repo Tasks

1. Finish `studies/01_circuit_atlas/thin_slice.py --compare`.
2. Trace matched false-belief successes and failures with explicit belief and
   reality targets.
3. Export a per-graph summary with graph scores and top feature contributors.
4. Select 3-5 candidate feature groups for causal tests.
5. Add a behavioral battery for first-person belief and communication updates.

## Reference Anchors

- Kosinski, 2024, PNAS: false-belief task battery for LLMs.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11551352/
- Gandhi et al., 2023, NeurIPS: BigToM procedural social-reasoning benchmark.
  https://arxiv.org/abs/2306.15448
- Ullman, 2023: trivial alterations to false-belief tasks.
  https://arxiv.org/abs/2302.08399
- Shapira et al., 2024, EACL: Clever Hans or Neural Theory of Mind.
  https://aclanthology.org/2024.eacl-long.138/
- Suzgun et al., 2025, Nature Machine Intelligence: KaBLE belief/knowledge/fact
  benchmark.
  https://www.nature.com/articles/s42256-025-01113-8
- Riemer et al., 2025, ICML: ToM benchmarks are broken for LLMs.
  https://research.ibm.com/publications/position-theory-of-mind-benchmarks-are-broken-for-large-language-models
- Anthropic, 2025: circuit tracing and attribution graphs with transcoders.
  https://www.anthropic.com/research/open-source-circuit-tracing
- Karvonen et al., 2025, ICML: SAEBench.
  https://arxiv.org/abs/2503.09532
- Conmy et al., 2023, NeurIPS: automated circuit discovery.
  https://arxiv.org/abs/2304.14997
