# The Neighborhood Is the Rule

Code and per-game results for

> **The Neighborhood Is the Rule: Why Memorization Beats Neural World Models in ARC-AGI-3**
> Xinyi Fang and Jin Zhao, Saint Louis University. Submitted to ESANN 2027.

ARC-AGI-3 puts an agent in grid games it has never seen. In these games the next color of a cell is fixed by the colors around it and the action taken, so a neighborhood seen once is known. A table that stores what followed each 13×13 neighborhood, filled from 300 moves of a game, predicts 55% of later frames exactly. A residual network pretrained on the 24 other public games and finetuned on the same 300 moves predicts 45%. Asking the network only where the table has not seen the neighborhood gives 59%.

All numbers in the paper, and all 25 games one by one, are in [`results/summary.md`](results/summary.md).

## Contents

| Path | What it does | Where it appears in the paper |
|---|---|---|
| `code/lookup_core.py`, `code/exp_lookup.py` | the neighborhood table and its evaluation | Table 1, Fig. 2, Sec. 2 |
| `code/exp_hybrid.py` | widest-seen table; table first, network where unseen | Table 1, Fig. 3b, Sec. 3 |
| `code/exp_meta.py` | network from scratch, pretrained, Reptile | Table 1, Fig. 3a, Sec. 4 |
| `code/exp_rollout_ft.py` | networks fed their own predictions; networks with 13×13 and 25×25 views | Fig. 3b, Sec. 3 and 5 |
| `code/exp_pretrain.py` | transfer without finetuning, number of games, moves per game | Sec. 4 |
| `code/exp_loss.py`, `code/exp_worldmodel.py` | networks trained on one game with more data | Sec. 3 |
| `code/exp_learnability.py`, `code/exp_radius.py` | determinism; how wide a neighborhood has to be | Sec. 2 |
| `code/exp_progsyn.py`, `code/dump_transitions.py` | step functions written by language models | Sec. 3 |
| `code/exp_sched.py` | scheduled sampling against plain training, in paired runs | Sec. 5 |
| `code/exp_horizon.py` | our first scheduled-sampling script, replaced by `exp_sched.py` (see its docstring) | |
| `code/sg/`, `code/scratch_wm/` | agents that choose moves with the network | Sec. 5 |
| `code/wmcache.py`, `code/prep_cache.py`, `code/env.py` | data collection and cache shared by the scripts | Setup |
| `code/summarize.py`, `code/sum_rfx_hybrid.py`, `code/sum_recheck.py` | turn result files into the tables of `results/summary.md` | |

## Setup

We ran the experiments with Python 3.13, `torch`, `numpy`, `arc_agi` and `arcengine` (the code was written against `arc_agi` 0.9.8 and `arcengine` 0.9.3). `exp_progsyn.py` also needs `transformers`. The agents in `code/sg/` and `code/scratch_wm/` import `agents.agent` from the official [ARC-AGI-3-Agents](https://github.com/arcprize/ARC-AGI-3-Agents) framework, checked out as `code/ARC-AGI-3-Agents`.

```bash
cd code
python prep_cache.py download                       # fetch the 25 public games (needs internet)
OPERATION_MODE=offline python prep_cache.py fill 8  # play and cache all random runs (8 processes)
```

Every experiment reads the same cached random play, so all predictors see identical moves:

| Cache entry | Used for |
|---|---|
| 3000 moves per game, seed 11 | pretraining pool (the 24 other games) |
| 900 moves per game, seed 23 | first 300 train the predictors, last 600 test them |
| 3600 moves per game, seed 31 | tables filled from 50 to 1000 moves (Fig. 2) |
| 4000 moves per game, seed 99, cut into unbroken runs | windows for imagined steps (Fig. 3b) |

The random agent picks uniformly among the available actions; a click lands on a uniform cell.

## Reproducing the results

Run from `code/`, with `OPERATION_MODE=offline`. Each run fits on one GPU; the longest (three seeds of the 25×25 network over 25 games) took about three hours.

```bash
mkdir -p ../results

# Table: all widths, the 300-move split, the data curve of Fig. 2
LK_OUT=../results/lookup.jsonl python exp_lookup.py

# Network from scratch, pretrained and with Reptile, ten seeds
for s in 0 1 2 3 4 5 6 7 8 9; do MT_SEED=$s python exp_meta.py > ../results/meta_s$s.log; done

# Networks fed their own predictions (9x9 view), three seeds
for s in 0 1 2; do RO_SEED=$s RO_OUT=../results/rollout_ft_s$s.jsonl python exp_rollout_ft.py; done

# Networks with the table's 13x13 view and with a 25x25 view, three seeds each
for rf in 13 25; do for s in 0 1 2; do
  RO_RF=$rf RO_SEED=$s RO_OUT=../results/rfx_rf${rf}_s$s.jsonl python exp_rollout_ft.py
done; done

# Widest-seen table and table-first hybrids, on the same test steps and the same windows
for s in 0 1 2; do HY_SEED=$s HY_OUT=../results/hybrid_s$s.jsonl python exp_hybrid.py; done

# Summaries (from the repository root)
cd .. && python code/summarize.py > results/summary.txt && python code/sum_rfx_hybrid.py
```

The remaining numbers come from the scripts below. Game ids have the form `ls20-xxxxxxxx`; `python -c "import wmcache; print(wmcache.all_games())"` lists them.

```bash
# Sec. 3: a 25x25 network trained on 1200 moves of ls20 alone (1500 collected, last 300 for testing)
LS_GAMES=<ls20 id> LS_RFS=25 LS_STEPS=1500 LS_EPOCHS=30 LS_WEIGHTS=1 LS_WIDTH=96 python exp_loss.py

# Sec. 3: step functions written by two open-weight language models, best of four samples
G=cd82-fb555c5d,sp80-589a99af,ar25-0c556536,m0r0-492f87ba,tn36-ef4dde99,s5i5-18d95033
PS_MODEL=Qwen/Qwen3-Coder-Next PS_GAMES=$G PS_K=4 PS_MAXNEW=4096 python exp_progsyn.py
PS_MODEL=google/gemma-4-31B-it PS_GAMES=$G PS_K=4 PS_MAXNEW=4096 python exp_progsyn.py

# Sec. 4: one held-out game; add PT_N_GAMES=n for fewer pretraining games,
# change PT_PER_GAME for more or fewer moves per game
PT_HELD=<game id> PT_PER_GAME=3000 PT_TEST_N=300 PT_EPOCHS=20 python exp_pretrain.py

# Sec. 5: scheduled sampling against plain training, from the same initial weights and the same batches
# (8 games, view 25x25, 4000 moves, 25 epochs, rate 0.15; these are the defaults)
for s in 0 1 2; do SC_SEED=$s SC_ARMS=tf,ss_next SC_OUT=../results/sched_s$s.jsonl python exp_sched.py; done

# Networks without the second click plane (see "The predictors")
for s in 0 1 2; do RO_CLICK_T=0 RO_SEED=$s RO_OUT=../results/clickab_s$s.jsonl python exp_rollout_ft.py; done

# Paired differences by depth, the click-plane comparison, and the compounding law game by game
cd .. && python code/sum_recheck.py
```

## The predictors

**Table.** For every cell the key is the colors of its k×k neighborhood, a mark where a click lands inside it, and the action. The table stores the most frequent next color seen for that key; a cell whose key is not in the table keeps its color. Keys are hashed with two random integer convolutions, so wide neighborhoods cost no more to store than narrow ones. *Widest seen* answers from the widest of the 7×7, 13×13, 25×25 and 41×41 tables that contains the key.

**Network.** A residual convolutional network with group normalization and width 96. The input is the one-hot frame, seven action planes and two click planes; the loss is cross-entropy over the 16 colors of every cell. The first click plane marks the clicked cell at row y, column x, which is how `arcengine` indexes its frames. The second marks the transposed cell (row x, column y), so that the input holds the click under either coordinate order. The second plane does not change the results: with `RO_CLICK_T=0`, which leaves it empty, the pretrained network reproduces 0.450 and 0.447 of frames in two reruns of three seeds, against 0.454 with both planes (seed-to-seed standard deviation 0.014; see `results/summary.md`). The view is 9×9 unless stated. Pretraining is 4000 Adam steps on the pooled moves of the other 24 games; finetuning is 20 epochs on the 300 moves of the held-out game. Reptile spends the same 4000 steps as 400 outer updates of 10 inner steps (ε = 0.2).

**Table first, network where unseen.** The table answers every cell whose key it has seen; the network answers the rest.

## Scores

- **Exact frame**: share of test steps whose predicted next frame matches in all 64×64 cells.
- **Real events**: steps that change more than three cells. *Exact on real events* is the exact-frame score on those steps only.
- **Nothing changes**: the score of predicting the current frame again (0.184 on average).

## Games named in the paper

The paper names two of the 25 games, the two ends of its Fig. 3a.

- **ls20** (navigating a map) is where the table leads by most. 450 of its 600 test steps are real events. The 7×7 table reproduces 0.8% of frames and the 13×13 table 84%. After the same 300 moves the pretrained network reproduces 0.6% with its 9×9 view, 20 to 34% with the 13×13 view (three seeds) and none with the 25×25 view. Trained on 1200 moves of ls20 alone, a 25×25 network reproduces 93% of the next 300 frames.
- **wa30** is where the network leads by most: 0.53 against 0.36 for the 13×13 table. Handing the table's unseen cells to the network gives 0.56.

Two more games appear without their names. The game of Sec. 2 that needs a 41×41 neighborhood is **sp80**; at that width 5837 of its neighborhoods recur and none is followed by two different colors. The six games of the program experiment are cd82, sp80, ar25, m0r0, tn36 and s5i5.

The table of all 25 games is in [`results/summary.md`](results/summary.md).
