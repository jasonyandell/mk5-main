# Sample Clipboard Output - Test Submission

This is what gets copied to clipboard when user clicks "Submit for Grading" on the test form:

```markdown
# Comprehension Test Submission

Topic: Intermediate AI Monte Carlo Implementation

---

## Question 1.1 (5 pts) - Monte Carlo simulation basics

**Answer:**
a) 42 game. specifically, we are starting from a state, inferring what dominoes other players COULD have, generating a random hand, playing it through ("rollout") with beginnerAI, measuring the outcome.
b) team points
c) this is a partnership game. the team is what's actually scored. the player's hand is just a contributor

**Discussion:**
(none)

---

## Question 1.2 (5 pts) - Rejection sampling

**Answer:**
Invariant: there is always a valid move for every player. If we get to the end and there are no moves that we know of, we have a bug in the simulator.

**Discussion:**
I'm not sure why we need 1000 failed attempts to understand that. It seems we could sample once if we truly had no bug?

---

## Question 2.1 (10 pts) - The critical bug with doesDominoFollowSuit

**Answer:**
A) true. doesDominoFollowSuit is misnamed. it doesn't fully check if a domino follows suit because it doesn't check for trumps.
b) it inferred that the 4-0 could follow suit against the 4-3
c) because it would incorrectly think that it MUST play that domino to follow suit

**Discussion:**
This is confusing - can you walk through a concrete example?

---

Please grade and respond to any discussion points.
```

---

# Sample Clipboard Output - Follow-up Discussion

This is what gets copied when user clicks "Submit Follow-up" on the review page:

```markdown
# Follow-up Discussion

Topic: Intermediate AI Monte Carlo Implementation

---

I'm still confused about 2.1. You said the constraint tracker infers P1 is "void in 0s" but I thought it was about whether 4-0 could follow suit.

Also, for 1.3, you mentioned infinite recursion but there's a guard for exactly 1 choice. Doesn't that prevent it?

Can we trace through the actual code for these?

---

Please respond to these follow-up points.
```
