# Beads - AI-Native Issue Tracking

Welcome to Beads! This repository uses **Beads** for issue tracking - a modern, AI-native tool designed to live directly in your codebase alongside your code.

## What is Beads?

Beads is issue tracking that lives in your repo, making it perfect for AI coding agents and developers who want their issues close to their code. No web UI required - everything works through the CLI and integrates seamlessly with git.

**Learn more:** [github.com/steveyegge/beads](https://github.com/steveyegge/beads)

## Quick Start

### Essential Commands

```bash
# Create new issues
bd create "Add user authentication"

# View all issues
bd list

# View issue details
bd show <issue-id>

# Update issue status
bd update <issue-id> --claim
bd update <issue-id> --status done

# Sync with Dolt remote
bd dolt push
```

### Working with Issues

Issues in Beads are:
- **Git-native**: Stored in Dolt database with version control and branching
- **AI-friendly**: CLI-first design works perfectly with AI coding agents
- **Branch-aware**: Issues can follow your branch workflow
- **Always in sync**: Auto-syncs with your commits

## Why Beads?

✨ **AI-Native Design**
- Built specifically for AI-assisted development workflows
- CLI-first interface works seamlessly with AI coding agents
- No context switching to web UIs

🚀 **Developer Focused**
- Issues live in your repo, right next to your code
- Works offline, syncs when you push
- Fast, lightweight, and stays out of your way

🔧 **Git Integration**
- Automatic sync with git commits
- Branch-aware issue tracking
- Dolt-native three-way merge resolution

## Get Started with Beads

Try Beads in your own projects:

```bash
# Install Beads
curl -sSL https://raw.githubusercontent.com/steveyegge/beads/main/scripts/install.sh | bash

# Initialize in your repo
bd init

# Create your first issue
bd create "Try out Beads"
```

## Learn More

- **Documentation**: [github.com/steveyegge/beads/docs](https://github.com/steveyegge/beads/tree/main/docs)
- **Quick Start Guide**: Run `bd quickstart`
- **Examples**: [github.com/steveyegge/beads/examples](https://github.com/steveyegge/beads/tree/main/examples)

---

## Backup (this repo)

Issues are backed up to DoltHub (`jasonyandell/t42-beads`, public, free):

- Auto-backup runs every 15 min on issue writes (`backup.enabled=true` in config.yaml).
- Manual sync: `bd backup sync`
- Status: `bd backup status`

### Disaster recovery

The local working copy is `.beads/embeddeddolt/` (gitignored, fast).
The local backup copy is `.beads/backup/*.darc` (gitignored, the canonical
`bd backup restore` source). The DoltHub remote is the off-machine copy
of last resort.

If `.beads/embeddeddolt/` is gone but `.beads/backup/` is intact:

    bd init --prefix=t42       # in a fresh dir or after wiping embeddeddolt
    bd backup restore .beads/backup --force

If even `.beads/backup/` is gone (lost machine), pull from DoltHub:

    dolt clone jasonyandell/t42-beads /tmp/restore
    # then copy /tmp/restore/.dolt chunks into a new bd workspace —
    # see the beads docs (this path is fiddly; the local backup
    # dir is the recommended recovery source).

---

*Beads: Issue tracking that moves at the speed of thought* ⚡
