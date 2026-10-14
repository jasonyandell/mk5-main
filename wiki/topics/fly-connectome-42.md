---
title: The fly connectome as a 42 player — what exists, whether it can learn 42, and a reservoir pilot
kind: topic
first_seen: 2026-10-14
last_updated: 2026-10-14
status: active
---

## What Jason meant

"That working fly neuron map" is the adult fruit fly's whole-brain wiring
diagram, run as a simulation. There are three layers to it:

1. **The map: FlyWire.** It covers one female *Drosophila* brain at synapse
   resolution: about 139k neurons and about 50M chemical synapses, every
   neuron typed, and a transmitter predicted for each one. It was published in
   *Nature* 634 (Oct 2024), and the current release is v783. The connectivity
   is on Zenodo under **CC BY 4.0**
   ([record 10676866](https://zenodo.org/records/10676866), version 783.0,
   2024-06-02, checked through the Zenodo API on 2026-10-14). The cell-type
   annotations are at
   [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
   That repo shows no license on GitHub, so treat it as the paper's
   supplement.
2. **The brain model: Shiu et al.** "A leaky integrate-and-fire
   computational model based on the connectome of the entire adult
   Drosophila brain," *Nature* 634:210–219 (2024). It has every neuron, uses
   synapse counts as weights and takes signs from the transmitter. Its weights
   are **not trained**. Activating sugar or bitter receptors in silico predicts
   which neurons fire, including the feeding motor neuron, and the paper
   reports about 91% agreement with optogenetic experiments. The code and the
   packaged connectome (`Connectivity_783.parquet`, 15.1M neuron pairs) are
   **MIT**:
   [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model),
   last pushed 2024-09-14.
3. **The "working fly": Eon Systems' embodied emulation** (2026-03-10). This
   is what went viral. It runs the Shiu model in a closed loop with a
   physics-simulated body (NeuroMechFly v2 in MuJoCo, 87 joints) and syncs
   every 15 ms. Gustatory, olfactory and mechanosensory neurons are driven by
   the virtual world. A handful of descending neurons (DNa01/DNa02 to steer,
   oDN1 to walk, the antennal-grooming DNs, MN9 to feed) drive controllers
   that were themselves trained by imitation. The fly grooms when dusted and
   steers to food. In Eon's own words, there is **no plasticity, no learning
   and no internal state**, and the brain-to-body mapping was mostly chosen by
   hand
   ([Eon write-up](https://eon.systems/updates/embodied-brain-emulation)).
   Their simulator is
   [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain)
   (GPL-2.0, with Brian2, Brian2CUDA, PyTorch, NEST GPU and GeNN backends,
   pushed 2026-08-29). The README gives no real-time factor. A separate
   hobbyist repo,
   [erojasoficial-byte/fly-brain](https://github.com/erojasoficial-byte/fly-brain)
   (MIT), does a similar embodiment with added Hebbian plasticity. It makes
   large claims that have not been reviewed and needs a CUDA GPU.

**The "teach it" literature** (trained models built on the connectome):

- [Lappalainen et al., *Nature* 634:1132 (2024)](https://ideas.repec.org/a/nat/nature/v634y2024i8036d10.1038_s41586-024-07939-3.html)
  fixed the visual-system wiring (64 cell types) and trained only the neuron
  and synapse parameters with backprop on a motion task. It predicted neural
  tuning across 26 studies. This is the clearest case of "connectome
  constrains, gradient descent fills in".
- [FlyGM, Jin et al., arXiv 2602.17997](https://arxiv.org/abs/2602.17997)
  (Feb–Mar 2026; NeurIPS 2025 precursor "flyGNN"). The whole-brain connectome
  is a fixed message-passing graph. The encoder, decoder, per-neuron
  descriptors and a shared update MLP are trained by imitation and then PPO
  to make a simulated fly walk and fly. It beats a degree-preserving rewired
  graph, an Erdős–Rényi graph and an MLP. Example: heading error 8.3° against
  13.6° (rewired) and 13.9° (MLP) at the hardest turn. **Caveat in their own
  text:** the graph baselines use unit-weight edges, so they are not matched
  to the signed, weighted connectome. The ablation with unweighted FlyGM
  (11.0°) shows that part of the gap is the weights, not the topology. The
  number of seeds is not stated.
- Reservoir-computing tests of fly connectomes. Morra & Daley
  ([arXiv 2201.09359](https://arxiv.org/abs/2201.09359), 2022) found a
  connectome echo-state network beat a random one on Mackey–Glass prediction.
  A 2024 study of *C. elegans* and larval mushroom bodies
  ([Ulster](https://pure.ulster.ac.uk/en/publications/random-and-biological-network-connectivity-for-reservoir-computin))
  found random topologies often match. A 2025 paper built a reservoir from the
  whole FlyWire brain (*Biomimetics* 10(5):341); I did not read its full
  text. Overall: **mixed, and no matched-null win on a symbolic task.**
- The mushroom body is the fly's associative memory. About 50 glomerular
  channels expand onto about 2,000 Kenyon cells per side, each with about 7
  claws, and the code is sparse because APL inhibition enforces it. Dopamine
  neurons teach the KC→MBON synapses, which is a three-factor rule. Two
  findings matter here. The PN→KC wiring is statistically close to random
  (Caron et al., *Nature* 2013). That random-expansion-plus-winner-take-all
  scheme is "FlyHash" (Dasgupta et al., *Science* 2017). In other words, the
  one part of the fly built for learning arbitrary associations is the part
  whose wiring is closest to random. **This is from memory. I did not re-fetch
  either paper on 2026-10-14.**

**Could not verify:** Eon's compute cost (not published). Whether
`fly-brain` (Eon) runs faster than real time on a laptop CPU. The license of
the annotation table. Full text of the 2025 whole-brain reservoir paper.
Anything about "teaching" the Eon fly a task (I found none: their model has
no learning).

## The verdict: viable as a toy, not as a player

**The brain cannot be the player (option d), and it cannot be a component
that Zeb would keep (option b).** Four reasons:

1. **There is nowhere to put 42.** The fly's interfaces are odours, tastes,
   bristles, photoreceptors and a few dozen descending command neurons. A 42
   position is about 160 bits of symbolic state, and every encoding onto
   receptors is an arbitrary choice. Whatever the brain then "knows" about 42
   is whatever the readout learns. The Eon demo works because grooming and
   feeding are what those circuits evolved to do, and 42 is not.
2. **The fixed brain does not learn.** Shiu's model and Eon's have no
   plasticity. Every trained connectome result above trains *around* the
   wiring (the readouts, encoders, neuron parameters, and FlyGM's shared MLP).
   So "teaching it 42" means option (a) or (b), and the wiring is then a
   prior, not a player.
3. **What 42 needs is not what the fly offers.** Zeb's strength is the
   plan: one action per information set over sampled worlds, with a rung
   model of the other seats ([[scenario-player]]). The only learned part is
   the field net, a 28-way policy over about 360 bit features with 128×64
   hidden units, run as integer arithmetic on a phone. The fly offers sparse
   coding, fast sensorimotor reflexes and one-shot associative memory. A
   connectome-shaped field net would be about 40k–140k units against the
   current 192, run through a recurrent graph at every modeled move inside
   rollouts. That is roughly a 10⁴× slowdown per field call, for a prior with
   no argument for why it suits tiles.
4. **The pilot (below) found nothing for the wiring to carry.** On the 5-5
   belief task, the real connectome reads no better than the same graph
   rewired or randomized. Through the mushroom body it reads slightly
   *worse* (+1.7 mnat [+1.0, +2.4] against rewired).

**Viable as a toy:** yes, and the toy is honest science. "Does real wiring
read a card table better than scrambled wiring?" can be answered in a
minute, with the right nulls, on a laptop CPU.

## The options, ranked

| rank | option | what it would be | verdict |
|---|---|---|---|
| 1 | (a) fixed reservoir + trained linear readout | the connectome as a frozen feature map for a 42 sub-task, against rewired and random nulls | **the pilot** — cheap, clean nulls, answers "does structure earn its keep" |
| 2 | (c) mushroom-body associative learning | a KC-like sparse expansion with a dopamine-style three-factor rule at the readout, learning e.g. "this partner's lead means X" across hands, online | a real idea for cross-hand opponent modeling ([[book-assumptions]] A1/A2: no opponent models you across hands). But the expansion is near-random in the fly too, so the connectome adds little over FlyHash. Build it as a random-expansion learner if the opponent-modeling program wants one. |
| 3 | (b) connectome-constrained trainable net as the field net | FlyGM-style: fixed graph, trained per-neuron parameters and a shared update | the field net is a 192-unit integer MLP for phone speed. A 40k-node graph net per modeled move is orders of magnitude slower. Run it only if (a) shows a structural edge. |
| 4 | (d) the brain *is* the player | sensory encoding in, descending neurons out, play by spikes | not viable: no learning in the model, no native channel for symbols, and nothing that plans across hidden worlds |

**Fence check.** Every option above trains on information-state targets:
Zeb's own plan values, game outcomes, or (in the pilot) the exact posterior
under the uniform belief. None uses a per-world clairvoyant value. The pilot
never reads a deal's hidden holder, only the counted posterior. The
[[strategy-fusion-fence]] is untouched, and `scenario/flybrain_pilot.py`
imports no clairvoyant API.

## The first experiment (pre-registered before the measured run)

**Task.** Seat 0 (sixes trump) holds no 5-5 and the 5-5 is unplayed. Which
of seats 1–3 holds it? Positions come from random legal play, after 4–23
plays of a random deal. The input is 164 bits: own hand, tiles gone, the
voids proved for each seat, and each seat's hand size. The target is the
**exact posterior** under the uniform-given-record belief, Zeb's default
(g ≡ 0), counted by `scenario.fiber`. This is information-state-native, and
it has a known ceiling.

**Representations.** All share the same L2-logistic readout on soft
targets, with L2 picked on a validation split.

- Raw bits (linear).
- A 256-unit MLP: the field net's kind of model.
- FlyHash: 5,177 units, 7 random claws each, ReLU above the median.
- The connectome reservoir. This is the non-optic FlyWire brain: 41,692
  neurons and 5.5M connected pairs, with the optic lobes and photoreceptors
  dropped to stay CPU-sized. It is a rate model, x ← tanh(2·Ŵx + u), run for
  4 steps, with rows L1-normalized. It is **not** Shiu's spiking LIF. It has
  two arms:
  - **mb**: the bits go to the 2,279 olfactory receptor neurons, and the
    readout is the 5,177 Kenyon cells.
  - **wb**: the bits go to the 5,497 non-visual sensory neurons, and the
    readout is the 1,299 descending neurons.

  Each arm runs on three graphs:
  - **real**: the connectome as is.
  - **rewired**: postsynaptic ends permuted. This keeps both degrees and
    each neuron's sign.
  - **er**: uniform endpoints with shuffled weights.

**Metrics.** On 1/4 of the positions held out:

- Expected accuracy E[p(argmax q)]. Chance is 1/3. The Bayes ceiling is
  E[max p].
- KL(p‖q) in millinats. The ceiling is 0. **KL is the primary metric**,
  because the accuracy headroom is small.
- Paired bootstrap 95% intervals over test positions.

**Decision rule** (fixed here, before the measured run):

- **The wiring earns its keep** if `mb:real` (or `wb:real`) has lower KL than
  **both** its `rewired` and its `er` null, with both paired intervals below
  0.
- **The fly is useful at all** if, in addition, it beats the raw-bit linear
  readout and FlyHash with the interval below 0.
- Anything else is a **null for structure**. A null is the expected outcome,
  and it would be a good one: it says the fly's value here is generic
  sparse-random expansion, which the project can have without the fly.

**The measured run.** `--n 6000 --seed 2`. That is 4,500 train+validation
positions and 1,500 test positions. Seed 2 is fresh: the smoke test used
`--n 800 --seed 1` and is disclosed below. Command:

    timeout 300 python -u -m scenario.flybrain_pilot --data scratch/fly/data --n 6000 --seed 2

**Data provenance** (downloaded 2026-10-14 to `scratch/fly/data/`, which is
gitignored and never committed):

| file | source | license | sha256 |
|---|---|---|---|
| `Connectivity_783.parquet` | philshiu/Drosophila_brain_model @ main | MIT repo; FlyWire data CC BY 4.0 | `efeb23fb…af0347` |
| `Completeness_783.csv` | same | same | `bbb847a4…a311` |
| `neuron_annotations.tsv` | flyconnectome/flywire_annotations `supplemental_files/Supplemental_file1_neuron_annotations.tsv` | none stated on GitHub (Schlegel et al. 2024 supplement) | `b214970b…83cd` |

## Pilot result (measured, `--n 6000 --seed 2`): a null for structure

The pre-registration above was committed at 2bae0410, before the run.

| representation | acc | KL (mnat) |
|---|---|---|
| uniform (chance) | 0.329 | 87.4 |
| hand sizes only | 0.356 | 81.8 |
| raw bits, linear | 0.409 | 9.1 |
| MLP 256 | **0.414** | **7.7** |
| FlyHash 5,177 | 0.408 | 10.4 |
| **mb: real** / rewired / er | 0.408 / 0.410 / 0.410 | **10.9** / 9.2 / 8.8 |
| **wb: real** / rewired / er | 0.410 / 0.411 / 0.411 | **9.2** / 9.0 / 8.9 |
| exact posterior (Bayes) | 0.419 | 0 |

Paired differences on the 1,500 test positions, as mean [95% bootstrap].
Negative KL means the first representation is better.

| comparison | KL (mnat) | acc |
|---|---|---|
| mb: real − rewired | +1.7 [+1.0, +2.4] | −0.001 [−0.003, +0.000] |
| mb: real − er | +2.0 [+1.3, +2.8] | −0.002 [−0.004, −0.001] |
| wb: real − rewired | +0.2 [−0.1, +0.5] | −0.002 [−0.004, +0.000] |
| wb: real − er | +0.3 [+0.1, +0.5] | −0.001 [−0.002, +0.000] |
| mb: real − raw | +1.7 [+1.0, +2.5] | −0.001 [−0.003, +0.001] |
| mb: real − FlyHash | +0.4 [−0.3, +1.2] | +0.000 [−0.002, +0.002] |
| wb: real − raw | +0.1 [−0.1, +0.3] | +0.000 [−0.001, +0.001] |
| wb: real − FlyHash | −1.2 [−1.7, −0.8] | +0.001 [−0.001, +0.003] |
| MLP − raw | −1.4 [−2.1, −0.9] | +0.005 [+0.003, +0.007] |

**By the pre-registered rule, the wiring does not earn its keep.** Neither
arm beats its nulls. The mushroom-body arm is reliably *worse* than both
nulls by 2 mnat, and the whole-brain arm ties them within 0.5 mnat. Neither
arm beats raw bits read linearly. The one reservoir win, wb over FlyHash, is
over the weakest baseline. The only representation that beats raw bits is the
ordinary MLP, the field net's kind of model.

**Why the mushroom body loses, as a reading, not a test.** Fly olfaction
pools receptor neurons of one type into one glomerulus. 2,279 ORNs make
about 53 types per side, so the 164 bits are squeezed through about 100
channels before they reach a Kenyon cell. Bits that land in the same
glomerulus are summed together. The rewired and ER graphs have no such
funnel. This is the encoding problem made concrete: the fly's input wiring
is built to merge receptors that report the same thing, and 42's bits are
not that kind of thing. Activity at the Kenyon cells was also 15× larger on
the real graph than on the rewired one (mean |x| 0.128 against 0.009, on a 64-position check). The
readout is standardized, so this is a difference in regime, not in scale.

**What this does not show.** The task has little headroom. Raw bits are
already within 9 mnat of exact. Every representation recovers about 88–95%
of the accuracy gap from chance to Bayes. So the test can see an edge of a
few mnat and no more. A harder sub-task, where the linear readout sits far
from the ceiling, could still find structure helping. It is one seed, one
gain (G = 2), one step count (T = 4), and a rate model rather than Shiu's
spiking LIF. A null here is a null for this reservoir on this task, not for
connectomes in general.

**Cost.** The real graph, pruned exactly to the neurons on input→readout
paths of at most 3 edges, is 9,255 neurons and 0.79M edges for mb, and
26,561 neurons and 3.4M edges for wb. The scrambled graphs do not prune:
40–42k neurons and 5.5M edges. The measured run did not fit one 300-s
process. It ran as five processes (`--graphs`/`--arms`; the seeds make that
identical to one run, and the baselines matched bit for bit across all five)
of about 2–4 min each. The machine's load average was about 60 from other
sessions. The smoke run at 800 positions took 51 s. Pruning was checked
exact against the unpruned network: max |Δx| 0 on real and 4·10⁻⁹ on
rewired. The first one-process attempt at seed 2 hit the 300-s timeout after
the real graph's two fits. It printed no scores, and the split run is the
measured one.

Commands:

    for g in real rewired er; do for a in mb wb; do
      timeout 300 python -u -m scenario.flybrain_pilot --data scratch/fly/data --n 6000 --seed 2 \
        --graphs $g --arms $a --out scratch/fly/seed2-$g-$a.npz; done; done
    python -u -m scenario.flybrain_pilot --data - --combine scratch/fly/seed2-*.npz

(The run grouped `real` into one process, with both arms in it.)

## The Doom demos (September 2026 and spring 2026)

The pilot's sources missed two widely covered game demos. Both are consistent with the verdict.

- **A fly connectome "plays" Doom.** Alex Wormuth connected the newer MaleCNS v1.0 connectome (Google Research and HHMI Janelia; ~166k neurons, ~125M synapses) to 1993's Doom. Game frames drive sensory neurons, and the resulting activity is read out as movement controls. Damage sends a pulse to two dopaminergic PPL1-01 neurons, the mushroom body's punishment pathway, as the reinforcement signal. Coverage reports no clear improvement after 6,385 attempts. The project's own documentation calls learning unproven. Similar hobby setups drive Super Mario 64 and Beat Saber. ([HotHardware](https://hothardware.com/news/google-mapped-a-fruit-fly-brain-so-engineers-taught-it-to-play-doom), [TechSpot](https://www.techspot.com/news/113780-google-mapped-fly-nervous-system-developers-using-play.html), [TweakTown](https://www.tweaktown.com/news/113507/googles-mapped-fruit-fly-brain-has-played-doom-more-than-6000-times-and-still-has-not-beaten-it/index.html))
- **Living human neurons play Doom.** Cortical Labs' CL1 holds ~200k cultured human neurons on a multi-electrode array. Game state goes in as stimulation, and firing comes out as actions. The neurons play poorly, with some reported in-session adaptation. This is the same group that taught neuron cultures Pong in 2022. ([Military.com](https://secure.military.com/feature/2026/03/10/scientists-teach-human-brain-cells-play-doom-sci-fi-experiment.html), [Decrypt](https://decrypt.co/359697/human-brain-cells-play-doom-cortical-labs-experiment?amp=1))

Neither demo changes the verdict. In both, an engineered encoding and readout carry the game, and the wiring is a fixed substrate. The Doom setup does make option (c) concrete. It shows a dopamine-neuron reinforcement pulse into the mushroom body's own learning pathway as an online learning signal. That is the three-factor rule the mushroom-body follow-up below proposes, built from random wiring rather than the connectome.

## What would come next (if anyone wants it)

The pilot answers the cheap question. The fly's wiring is not a better
feature map for 42's beliefs than random wiring of the same size. Two
follow-ups are worth having on file; neither is urgent.

- **Harder task, same nulls.** Use the field net's own target: the pour's
  margin softmax, graded by regret under q
  (`scenario/field_train.py`), on a few thousand rows of
  `texas-42-zeb-pour-250k`. Readouts from the real, rewired and ER reservoirs
  go against the deployed `margin_d250k_h128x64`. That test has headroom, and
  it is the one Zeb would care about. Expect a null again. If one appeared,
  it would be the only route to option (b).
- **The mushroom body as a learner, not a map.** Option (c) as an
  opponent-modeling experiment. A random sparse expansion (FlyHash) with a
  three-factor, dopamine-style rule at the readout would learn one table's
  habits online across hands. That belongs to the
  [[book-assumptions]] program (A1/A2), and the pilot says to build it from
  random wiring, not from the connectome.

**Smoke test, disclosed** (`--n 800 --seed 1`; 200 test positions; not the
measured run):

| | uniform | sizes only | raw | mlp | flyhash | mb real / rewired / er | wb real / rewired / er | Bayes |
|---|---|---|---|---|---|---|---|---|
| acc | 0.342 | 0.371 | 0.406 | 0.406 | 0.400 | 0.402 / 0.407 / 0.407 | 0.406 / 0.404 / 0.407 | 0.422 |
| KL (mnat) | 96.0 | 86.7 | 15.0 | 19.7 | 20.8 | 16.9 / 15.8 / 15.9 | 17.1 / 15.9 / 15.3 | 0 |

It took 51 s wall time on the M5 Max CPU. Every reservoir sits within about
2 mnat of the others and of raw bits. The measured run shows the same picture
with tighter intervals.

## Links

[[scenario-player]] (Zeb, the field net) · [[strategy-fusion-fence]] ·
[[book-assumptions]] (A1/A2: cross-hand opponent modeling, where an
associative learner would live) · [[belief-policy-value-algebra]]
