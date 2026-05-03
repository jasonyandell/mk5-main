# Secrets

Values live in the macOS login Keychain, not in this repo. This file is
just an index of service names so future-you (or a fresh machine setup)
knows what to look for.

Keychain access requires the user's login password / Touch ID, so
knowing a service name is useless without the keyring itself.

## Inventory

| Service name           | Account        | What it is                          |
|------------------------|----------------|-------------------------------------|
| `huggingface-token`    | `jasonyandell` | HF write token (current name: `yolo`) |
| `wandb-api-key`        | `user`         | Weights & Biases API key            |
| `vastai-api-key`       | `jasonyandell` | Vast.ai API key                     |

## Read

```bash
export HF_TOKEN=$(security find-generic-password -s huggingface-token -w)
export WANDB_API_KEY=$(security find-generic-password -s wandb-api-key -w)
export VAST_API_KEY=$(security find-generic-password -s vastai-api-key -w)
```

## Add or rotate

```bash
security add-generic-password -U \
  -a <account> -s <service-name> -w <secret-value> \
  -j "<short note about source / purpose>"
```

`-U` updates an existing entry in place. `-j` stores a free-form
description retrievable with `security find-generic-password -s <name> -g`.

## SSH keypair

Lives at `~/.ssh/id_ed25519` / `~/.ssh/id_ed25519.pub` (label
`jason@lambda`, fingerprint `SHA256:WY/3AOOy6cBv0S2fPHsOQAMurihgbR57LNMtSZt2lnU`).
Used for Vast.ai and other remote hosts.

`~/.ssh/config` is set with `AddKeysToAgent yes` + `UseKeychain yes`,
and the key is loaded into the agent via
`ssh-add --apple-use-keychain ~/.ssh/id_ed25519`. macOS auto-restores
the agent on login, so SSH stays "always logged in" — no per-session
unlock.

## Persistence model

Each CLI tool reads from its own native credential file:

| Tool       | File                              | Refresh command                     |
|------------|-----------------------------------|-------------------------------------|
| HF         | `~/.cache/huggingface/token`      | `hf auth login --token $HF_TOKEN`   |
| W&B        | `~/.netrc` (machine `api.wandb.ai`) | `wandb login $WANDB_API_KEY`      |
| Vast.ai    | `~/.config/vastai/vast_api_key`   | (write the value into the file)     |
| SSH        | `~/.ssh/id_ed25519` + agent       | `ssh-add --apple-use-keychain …`    |

Keychain is the master vault. If any per-tool file gets wiped, rehydrate
with `security find-generic-password -s <name> -w > <file>`.

## Don't add to this file

- Token values, API keys, passwords, or anything that looks like one
- Internal hostnames or paths that imply a token's permission scope
  (e.g. "the prod-billing token" — name it neutrally)
