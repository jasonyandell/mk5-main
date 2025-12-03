---
name: comprehension-test
description: Code comprehension testing - activates when user wants to be tested on code changes. Triggers on "test me on", "quiz me", "comprehension test", "check my understanding" for uncommitted changes, recent commits, or commit ranges. Generates interactive HTML tests, grades answers, facilitates discussion.
---

# Code Comprehension Test Skill

Generate interactive tests over code changes to verify deep understanding, then grade and discuss.

## Activation Examples

- "Test me on the uncommitted changes"
- "Quiz me on the last 3 commits"
- "Make a comprehension test for the intermediate AI code"
- "Check my understanding of commit abc123..def456"

## Workflow

### Step 1: Read the Content

Determine what to test based on user request.

Read related docs or files to understand context deeply.

### Step 2: Generate Test

Create questions that test **deep understanding, not surface recall**. The structure should emerge from the content - let the code tell you what matters.

Generate HTML at `scratch/comprehension-test-{timestamp}.html` with:
- Questions with point values
- Answer textarea for each question
- Dark theme (#1a1a2e background)
- Submit button → copies to clipboard → alert: "Copied! Paste results back into Claude Code"

See `templates/test.html` for the HTML pattern.

### Step 3: Tell User How to Open

```
npx playwright open file:///absolute/path/to/scratch/comprehension-test-{timestamp}.html
```

### Step 4: Grade Submission

When user pastes results:
- Grade answers fairly (alternative correct answers get credit)
- Respond to discussion notes (user might be right - check your work!)
- Generate review HTML at `scratch/comprehension-review-{timestamp}.html`

See `templates/review.html` for the HTML pattern. Color scheme:
- Question: #00d9ff (cyan)
- User answer: #f39c12 (orange)
- Expected: #2ecc71 (green)
- Feedback: #9b59b6 (purple)
- Discussion: #e74c3c (red)

Review page also has Submit button for follow-up discussion.

### Step 5: Continue Discussion

User may want to dig deeper on specific points. You have full context.

## Reference

- `templates/test.html` - HTML pattern for test form
- `templates/review.html` - HTML pattern for graded review
- `examples/sample-output.md` - What the clipboard output looks like
