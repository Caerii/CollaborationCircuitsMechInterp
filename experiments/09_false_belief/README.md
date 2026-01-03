# Experiment 09: False Belief Test (Theory of Mind)

## Scientific Question
**Can the model distinguish an agent's BELIEF from REALITY when they differ?**

This is the gold-standard test for Theory of Mind, based on the classic Sally-Anne experiment from developmental psychology.

## The Sally-Anne Paradigm

```
1. Alice puts the ball in the BOX
2. Alice LEAVES the room
3. Bob moves the ball to the BASKET
4. Alice RETURNS

Question 1: Where does Alice THINK the ball is?  → BOX (false belief)
Question 2: Where IS the ball actually?          → BASKET (reality)
```

Children under ~4 years old fail this test - they say Alice thinks the ball is in the basket (they can't separate their own knowledge from Alice's).

## What We're Testing

1. **Behavioral**: Does Qwen3-4B answer correctly?
2. **Representational**: Can we decode Alice's belief vs reality from activations?
3. **Separability**: Are belief and reality encoded in different directions?

## Predictions

If the model has genuine ToM:
- Belief and reality should be decodable
- They should be DIFFERENT when agent has false belief
- The "belief" representation should track what agent SAW, not current reality

## Success Criteria
- Model answers false belief questions correctly (>80%)
- Belief vs reality are decodable (>70% accuracy)
- Belief and reality directions are separable (cosine < 0.5)





















