#!/bin/zsh
# Regenerate the jud v1 seed corpus on the REPAIRED sampler (post-4123b2d5).
# Same seeds/composition as the 2026-07-06 gen_corpus.sh so chunk m1 doubles as
# a same-seed field-drift probe (pre-repair vs post-repair sampler).
set -e
cd /Users/jason/code/mk5-main/.claude/worktrees/fix-jud
PY=/Users/jason/code/mk5-main/forge/venv/bin/python
CKPT="/Users/jason/code/mk5-main/forge/models/domino-large-817k-valuehead-acc97.8-qgap0.07.ckpt"
MODEL="/Users/jason/code/mk5-main/champion/margin_net_r8.pt"
mkdir -p scratch/jud-v1/corpus scratch/jud-v1/gen_tmp
gen() {  # $1 team spec  $2 seed  $3 out name  $4 n_games
  PYTHONPATH=$PWD $PY -u -m arena.cli \
    --team-a "$1" --team-b "$1" \
    --n-games $4 --n-samples 10 --device mps --checkpoint $CKPT \
    --base-seed $2 --out-dir scratch/jud-v1/gen_tmp \
    --emit-snapshots scratch/jud-v1/corpus/$3
  echo "=== done $3 ==="
}
gen "margin:wp,model=$MODEL+lens:ev" 10000000 snaps_m1.json 500
gen "margin:wp,model=$MODEL+lens:ev" 10050000 snaps_m2.json 500
gen "margin:wp,model=$MODEL+lens:ev" 10100000 snaps_m3.json 500
gen "net:wp+lens:ev" 11000000 snaps_n1.json 500
gen "net:wp+lens:ev" 11050000 snaps_n2.json 500
gen "random+random" 12000000 snaps_r1.json 500
gen "random+random" 12050000 snaps_r2.json 500
echo "=== ALL CORPUS DONE (repaired sampler) ==="
