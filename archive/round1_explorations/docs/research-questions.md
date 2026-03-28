# Research Questions

## Primary Questions

### 1. Collaboration Circuits

- **What are the internal mechanisms that enable models to collaborate?**
  - Which attention heads/layers mediate cross-agent information flow?
  - How do models represent the state of other agents they're collaborating with?
  - Can we identify sparse circuits responsible for coordination vs. competition?

### 2. Agent and User Representations

- **How do models internally represent other agents and users?**
  - Where in the network are "other agent" representations stored?
  - How persistent are user representations across conversation turns?
  - Can we separate representations of self, user, and other agents?
  - How does "evaluation awareness" (knowing you're being tested) manifest mechanistically?

### 3. Emergent Behaviors

- **What causes groupthink, collusion, and deception in multi-agent systems?**
  - Is there a "conformity circuit" that causes agents to agree with the majority?
  - How do agents develop hidden communication channels for collusion?
  - What internal mechanisms enable deceptive behavior, and how is it different from errors?

### 4. Safety and Intervention

- **Can we detect and prevent harmful multi-agent behaviors mechanistically?**
  - Can probes detect when an agent is building a manipulative model of the user?
  - Can we use steering vectors to prevent value drift when agents interact?
  - How do we monitor for collusion or toxic agreement in real-time?

## Secondary Questions

- How do representations of "user" vs. "other agent" interact or interfere?
- Do models form separate representations for different agents in a conversation, or do they get conflated?
- Can we trace information flow from one agent's output through another agent's processing?
- How do evaluation-awareness representations affect collaboration dynamics?
- Are there "collaboration primitives" (like attention patterns) that generalize across tasks?

## Hypothesis for MATS Application

**Core Hypothesis**: In multi-agent interactions, models maintain separable latent representations for (i) the user, (ii) partner agents, and (iii) evaluation context. Collaboration failures correlate with representational interference, which can be detected and causally fixed via targeted steering.

**Testable Claims**:
- User, agent, and self-representations are decodable via probes/decoders
- These representations can be separated and don't always interfere
- Interference (when it occurs) predicts collaboration failures
- Targeted steering can restore proper separation and improve collaboration

