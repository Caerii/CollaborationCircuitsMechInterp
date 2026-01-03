# Experiment 08: Belief Tracking with Minimal Pairs

## Scientific Question
**Does the model represent WHO knows WHAT as separable, decodable features?**

## Design Principles

### 1. Minimal Pairs (Control Lexical Confounds)
Each pair differs ONLY in the agent attribution:
- "Alice knows the password is 7492"
- "Bob knows the password is 7492"

Same content, same words (mostly), only the KNOWER changes.

### 2. Multiple Content Types
Test across different knowledge types to ensure generalization:
- Secrets (passwords, codes)
- Locations (where objects are)
- Plans (what someone intends to do)
- Facts (information about the world)

### 3. Balanced Design
- 2 agents (Alice, Bob) × N content items
- Each content item appears with both agents
- Perfectly balanced → no lexical shortcut possible

### 4. Prediction
If the model tracks beliefs properly:
- We should be able to decode WHICH AGENT from activations
- This should generalize across content types
- The "agent" dimension should be orthogonal to "content" dimension

## Analyses

1. **Agent Classification**: Can we decode Alice vs Bob from activations?
2. **Content Classification**: Can we decode WHAT is known?
3. **Orthogonality Test**: Are agent and content representations independent?
4. **Cross-Content Generalization**: Train on passwords, test on locations

## Success Criteria
- Agent classification > 70% with cross-content generalization
- Agent and content directions should be ~orthogonal (cosine < 0.3)























