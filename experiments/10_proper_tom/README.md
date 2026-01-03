# Experiment 10: Proper Theory of Mind Testing

## Addressing the Critique

The previous experiments had several methodological issues:

1. **Label Recognition, Not ToM**: Probes learned "User:" → class 0, not mental states
2. **Shortcut Heuristics**: False belief "success" may be outputting first location
3. **Behavioral Q&A, Not Representation Probing**: We tested answers, not internal representations
4. **Single Model, Not Multi-Agent**: No actual agent-to-agent interaction

## Correct Approach (Following Zhu et al.)

### Key Insight from Zhu et al.
> "Belief and reality should be decodable as SEPARATE directions from the NARRATIVE, 
> not just behaviorally answerable from Q&A prompts."

### What We'll Do

1. **Probe the STORY, not the Q&A**
   - Extract activations from the narrative BEFORE asking questions
   - Probe for "Sally's believed location" and "actual location" separately
   
2. **Test for Shortcut Heuristics**
   - Vary position of belief vs reality locations
   - Counter-balance first/second mention
   
3. **Minimal Pairs with Location Control**
   - Same locations, different belief holders
   - Same belief holder, different locations

4. **Real Multi-Agent Setup**
   - Model A generates → Model B processes A's output
   - Trace how B represents A's claims

## Success Criteria

- Belief direction ≠ Reality direction (cosine < 0.3)
- Probes work from NARRATIVE (not Q&A)
- Results survive position counter-balancing
- Transfer to novel locations/agents























