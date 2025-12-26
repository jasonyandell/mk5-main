---
name: grinder
description: Grinder bead management - activates for batching multiple beads together in isolated worktrees. Use when user says "grind X and Y together", wants to batch related work, or when implementing a bead with [grinder] label. Creates self-documenting epic beads with outcome reports.
---

# Grinder Skill

Create and execute **grinder beads** - meta-tasks that batch multiple beads for isolated execution in a git worktree.

## When to Use

- User says "grind these beads", "batch these together", "work on X, Y, Z together"
- User says "implement" a bead that has `[grinder]` label or "grinder bead" in description

## Creating a Grinder Bead

When user wants to batch beads together:

```bash
bd create \
  --type=epic \
  --title="Grind: <brief description>" \
  --labels=grinder \
  --description="<use template below>"
```

### Description Template

```markdown
## Grinder Bead

This is a **grinder bead** - a meta-task that processes multiple beads together
in an isolated git worktree.

### Goal
<What we're trying to achieve with this batch of work>

### Child Beads
- <bead-id>: <title>
- <bead-id>: <title>

### Execution Instructions

**Setup worktree:**
git worktree add ../mk9-worktrees/<this-bead-id> -b bead-grinder/<this-bead-id>
cd ../mk9-worktrees/<this-bead-id>
npm install  # if needed

**For each child bead:**
1. bd show <child-id>
2. Implement the task
3. npm run test:all
4. git add -A && git commit -m "<child-id>: <summary>"
5. bd close <child-id>

**After all child beads complete:**
1. Run final `npm run test:all`
2. Write Outcome Report (see below)
3. Assess vs Goal - did we achieve it?
4. File follow-up beads for anything discovered/incomplete
5. Update this bead with outcome report
6. Close this bead with summary

### Outcome Report Template
After grinding, update this bead description by appending:

---
## Outcome Report

### Completed
- ✅ <bead-id>: <what was done>
- ✅ <bead-id>: <what was done>

### Changes
- `path/to/file.ts` - <brief description>
- `path/to/file.ts` (new) - <brief description>

### Test Results
<X> tests pass / <any failures or notes>

### Goal Assessment
<Did we achieve the goal stated above? Fully/Partially/No?>
<Brief explanation>

### Follow-up Beads Filed
- <bead-id>: <title> (discovered during grind)
- None (if nothing needed)

### Branch
`bead-grinder/<this-bead-id>` ready to merge

### Acceptance Criteria
- [ ] All child beads implemented and closed
- [ ] All tests pass
- [ ] Outcome report written
- [ ] Goal assessed
- [ ] Follow-ups filed
- [ ] All changes committed to grinder branch
```

## Executing a Grinder Bead

When user says "implement <grinder-bead-id>":

1. **Read the bead**: `bd show <id>` - the description contains execution instructions
2. **Setup**: Create worktree, cd into it, npm install if needed
3. **Grind**: Work through each child bead (implement, test, commit, close)
4. **Review**:
   - Run final test suite
   - Assess outcome vs stated goal
   - File follow-up beads for discovered work or incomplete items
5. **Report**: Update grinder bead with outcome report
6. **Close**: Close the grinder bead with summary

## Example

**User:** "Grind t42-s1o and t42-liuw together - they're both AI performance work"

**Response:** Create grinder bead with goal "Improve AI performance" and child beads listed.

**User:** "Implement t42-grind-xyz"

**Response:**
1. Create worktree
2. Work each child bead
3. Write outcome report
4. Assess: "Goal fully achieved - AI now uses endgame enumeration and smarter bidding"
5. File follow-ups if any
6. Close with summary

## Notes

- Grinder beads use sibling worktrees (`../mk9-worktrees/`) to avoid nesting
- Each child bead gets its own commit for clean history
- The grinder branch can be merged or cherry-picked as needed
- If a child bead fails, stop and report - don't continue blindly
- The outcome report makes the bead a complete record of what happened
- Always assess vs goal - this catches drift and ensures we actually solved the problem
- Follow-up beads should use `discovered-from` dependency type
