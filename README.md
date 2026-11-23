# DuoProp

Code for **DuoProp: Dual-Consistent Atoms and Neighbourhood Propagation for Multi-Hop Logical Reasoning over Knowledge Graphs**.

DuoProp is a module for QTO-style reasoners. It keeps the link predictor frozen and changes how its scores become truth values.

- **Duo** (atom level): weighted geometric mean of the head-side and tail-side calibration of every triple; rows without an observed answer are scaled by a learned matchability `m(h,r)^rho`; self-loops are removed for relations that are never reflexive in the training graph.
- **Prop** (set level): after every projection, each entity receives the weighted average truth of its nearest neighbours in an entity similarity graph.

## Datasets

DuoProp uses the BetaE query sets for **FB15k-237**, **FB15k** and **NELL995**. Stanford distributes all three in one archive. [Download KG_data.zip](https://snap.stanford.edu/betae/KG_data.zip) · [BetaE data documentation](https://github.com/snap-stanford/KGReasoning)

With `wget` and `unzip` installed, run the download script from the repository root.

```bash
bash scripts/download_data.sh
```

| Benchmark | Directory passed to `--data` |
| --- | --- |
| FB15k-237 | `data/FB15k-237-betae` |
| FB15k | `data/FB15k-betae` |
| NELL995 | `data/NELL-betae` |

For a manual download, extract the ZIP into `data/` and use the folders ending in `-betae`. Each contains the graph splits, entity and relation mappings, and the query and answer files. [Data layout](data/README.md)

## Main results

Selected comparisons from Table VII of the paper. Each cell gives test MRR (%) averaged over the nine positive query types and the five query types with negation, in that order. Under the standard protocol, training edges receive truth one. The consistent protocol also sets validation edges to one at test time.

| Protocol | Method | FB15k-237 | FB15k | NELL995 |
| --- | --- | ---: | ---: | ---: |
| Standard | QTO (published) | 33.5 / 15.5 | 74.0 / 49.2 | 32.9 / 12.9 |
| Standard | DuoProp, QTO training settings (five-seed mean) | 34.5 / 16.2 | 74.8 / 53.5 | 33.2 / 13.9 |
| Standard | DuoProp, validation-selected single model | 35.4 / 18.3 | 76.5 / 56.9 | 33.9 / 13.7 |
| Standard | DuoProp, top-4 ensemble | 35.6 / 18.3 | 76.9 / 57.2 | 34.3 / 13.9 |
| Standard | DuoProp, CKGC link predictor | 35.3 / 17.9 | 78.2 / 59.4 | 34.0 / 14.0 |
| Consistent | CKGC (published) | 34.8 / 25.3 | 80.6 / 71.0 | 35.3 / 19.9 |
| Consistent | QTO, CKGC link predictor | 34.5 / 25.0 | 77.0 / 66.6 | 34.6 / 19.6 |
| Consistent | DuoProp, CKGC link predictor | 36.5 / 26.6 | 80.6 / 70.9 | 36.7 / 20.7 |
| Consistent | DuoProp, validation-selected single model | 36.7 / 26.7 | 79.1 / 66.8 | 37.0 / 21.1 |

The final test records for the validation-selected single model include the per-query-type scores. Each file contains the Duo-only comparison followed by the selected DuoProp setting. [FB15k-237](results/final_selection/MS_lrt237_r2ke1500_finalP_coll_test.jsonl) · [FB15k](results/final_selection/MS_hp15k22_lr002_finalP_coll_test.jsonl) · [NELL995](results/final_selection/MS_lrtnell_r3klr005_finalP_coll_test.jsonl)

## Layout

| Path | Content |
|---|---|
| `duoprop/scorer.py` | batched scoring from SpherE, GQE, Query2Box and ComplEx checkpoints |
| `duoprop/match.py` | matchability probes `m(h,r)` |
| `duoprop/build.py`, `build_blocked.py` | per-relation truth matrices (`--mode forward`: QTO head side, `--mode dual`: Duo); sharded build, blocked variant for NELL995 |
| `duoprop/reason.py` | fuzzy-logic evaluator with QTO semantics over CSR truth matrices |
| `duoprop/aggregation.py`, `neighbours.py`, `cand_probe.py` | p-norm aggregation and the similarity graphs used by Prop |
| `duoprop/collective_eval.py` | Prop, validation selection (one-stage and two-stage) and test evaluation |
| `duoprop/ensemble.py` | average the Duo matrices of several runs |
| `duoprop/run_ckpt.py`, `final_select.py`, `reselect.py`, `summarize_valid.py`, `report.py` | sweeps, selection and reporting |
| `duoprop/gen_pipeline.py`, `gen_joint.py`, `gpuqueue.py`, `prune.py` | job generation, GPU queue and cache clean-up used to run the experiments on one multi-GPU machine |
| `tools/` | scripts of single experiments: QTO baseline matrices (`qto_forward*.py`), per-query results (`perquery.py`), consistent protocol (`obs_eval.py`, `obs_valid_s.py`), fixed-setting seeds (`fixed_eval.py`, `qto_fixed.py`), matrices split over GPUs (`mg_run.py`, `multigpu.py`), build benchmark, FIT solver (`fit_run.py`), CKGC link predictor (`ckgc2kbc.py`, `ckgc_duo_build.py`), atom diagnosis (`atom_diag.py`, `dual_diag.py`) |
| `analysis/` | statistics and examples of the diagnosis section |
| `third_party/sphere/` | SpherE training code (WWW 2025, MIT licence), upstream version |
| `scripts/download_data.sh`, `data/README.md` | BetaE benchmark download and directory layout |
| `results/final_selection/` | the three standard-protocol single-model test records linked above |

## Setup

    pip install -r requirements.txt
    mkdir -p cache/art cache/mats runs/results runs/paper
    export PYTHONPATH=$PWD             # the scripts in tools/ and analysis/ import duoprop

Tested with Python 3.11, torch 2.5.1 (CUDA 12.1), numpy 1.26.4, scipy 1.17.1. Truth matrices at the lowest storage thresholds of the paper need an 80 GB GPU (or several GPUs, see `tools/mg_run.py`); at the default threshold 1e-3 the FB15k-237 matrices fit in under 1 GB.

## Run the FB15k-237 single model

Supply a frozen ComplEx checkpoint in the QTO `best_valid.model` format. The reported FB15k-237 run used rank 2000 and 1,500 training epochs. The checkpoint can be trained with the [QTO implementation](https://github.com/bys0318/QTO).

```bash
DATA=data/FB15k-237-betae
CKPT=/path/to/best_valid.model
MATCH=cache/art/match_fb237.pt
PREFIX=cache/mats/fb237

python -m duoprop.match --data "$DATA" --ckpt "$CKPT" --out "$MATCH"
python -m duoprop.build --data "$DATA" --ckpt "$CKPT" --out "$PREFIX" \
  --mode dual --temps 0.75 --alphas 0.7 --eps 1e-3
python -m duoprop.build --data "$DATA" --out "$PREFIX" \
  --mode dual --temps 0.75 --alphas 0.7 --merge

python -m duoprop.collective_eval --data "$DATA" --name fb237_single \
  --tag fb237_single --ckpt "$CKPT" --kind rel+emb --k 20 \
  --cache "${PREFIX}_dual_T0.75_a0.7_b0.0.pt" --match "$MATCH" \
  --rb 2 --betas 0.1 --aggs 2 --extra= --neg_fixed 3 --irreflexive
```

The evaluator writes validation and test records under `runs/results/`. The last record in `fb237_single_coll_test.jsonl` contains DuoProp's per-query-type MRR, `AVG_p` and `AVG_n`. Use a distinct `--tag` for each checkpoint because the embedding neighbour graph is cached by tag.

## Training and validation search

Run everything from the repository root.

**1. Link predictor.** SpherE (FB15k-237, d = 256, seed 0):

    python third_party/sphere/main.py --cuda --data_path data/FB15k-237-betae --do_train --do_test --model sphere -n 128 -cenr '(none, 0.02)' --save_checkpoint_steps 50000 --valid_steps 100000000 --cpu_num 4 --tasks 1p.2p.3p --select_train_query 1p --select_test_query 1p.2p.3p -d 256 --seed 0 -lr 0.0001 -b 512 -g 24 --max_steps 450001 --prefix runs/ckpt/ms237_s0

The checkpoint is written to `runs/ckpt/ms237_s0/FB15k-237-betae/sphere/<settings>/<time>/checkpoint`.

ComplEx is trained with the kbc code of QTO (https://github.com/bys0318/QTO), for example FB15k-237 with rank 2000:

    cd QTO/kbc/src
    python main.py --dataset FB15K-237 --batch_size 1000 --lmbda 0.05 --w_rel 4 --score_rel True --model ComplEx --rank 2000 --learning_rate 0.1 --max_epochs 1500 --model_cache_path <run_dir>/kbc/

Below, `CKPT` is the checkpoint file: the SpherE, GQE or Query2Box `checkpoint`, or `kbc/best_valid.model` for ComplEx. The job generators (`duoprop/gen_pipeline.py`, `gen_joint.py`) expect it at `<run_dir>/r1/checkpoint`.

**2. Matchability.**

    python -m duoprop.match --data data/FB15k-237-betae --ckpt CKPT --out cache/art/match_NAME.pt

**3. Duo truth matrices** (a grid over T and alpha shares the link-predictor scores):

    python -m duoprop.build --mode dual --data data/FB15k-237-betae --ckpt CKPT --out cache/mats/NAME --match cache/art/match_NAME.pt --temps 3.0,4.0 --alphas 0.6,0.7 --eps 1e-3 --shard 0 --nshards 2
    python -m duoprop.build --mode dual --data data/FB15k-237-betae --ckpt CKPT --out cache/mats/NAME --match cache/art/match_NAME.pt --temps 3.0,4.0 --alphas 0.6,0.7 --eps 1e-3 --shard 1 --nshards 2
    python -m duoprop.build --mode dual --data data/FB15k-237-betae --out cache/mats/NAME --temps 3.0,4.0 --alphas 0.6,0.7 --nshards 2 --merge

This writes `cache/mats/NAME_dual_T{T}_a{alpha}_b0.0.pt`. `--mode forward` builds QTO's matrices for the baseline.

**4. Prop, selection and test.** Neighbour graph of the embedding view (the structural view is built on first use):

    python tools/emb_knn.py data/FB15k-237-betae CKPT NAME

One cache, validation only:

    python -m duoprop.collective_eval --kind rel+emb --data data/FB15k-237-betae --name MYRUN_T4.0_a0.6 --tag NAME --cache cache/mats/NAME_dual_T4.0_a0.6_b0.0.pt --match cache/art/match_NAME.pt --rb 2 --eps none --neg_scale 1 --betas 0,0.05,0.1,0.2,0.3 --aggs max,2,4 --extra= --limit 1000 --irreflexive --no_test

To select among several caches on validation and evaluate the selected setting once on test, run this for every cache (names with a common prefix, as above) and then `--select_from MYRUN_T --refine_top 5 --criterion pos` (`--criterion sum` for the AVG_p + AVG_n rule). `duoprop/gen_pipeline.py` and `duoprop/gen_joint.py` generate the complete job chains used for the paper. The negation scaling is fixed to the published value of the baseline reasoner (`--neg_fixed`, see the paper). Neighbour graphs are cached in `cache/art/`.

The scripts in `analysis/` read the selection records in `runs/results/` and the paths set at the top of each script; the case studies print entity names from `aux_names/FB15k-237_entity2text.txt` (`entity2text.txt` of FB15k-237 in https://github.com/yao8839836/kg-bert).

## Check of the released code

After the modules were renamed for this release, step 4 was re-run on the FB15k-237 SpherE checkpoint (seed 0, T = 4, alpha = 0.6, 1,000 validation queries per type, neighbour graphs and matrices from the original run). All 14 per-type validation MRRs agree with the records of the original run to within 6e-5; for beta = 0.3 with p = 4 the average over the nine positive types is 25.37 (AVG_n 14.28). Steps 1-3 were not re-run; apart from module names, comments and docstrings, their code (`scorer.py`, `match.py`, `build.py`, `build_blocked.py`) is identical to that of the original runs.

## Not included

Checkpoints, truth-matrix caches (about 0.9 TB in our runs), the datasets, and the code of QTO, FIT and CKGC; please use their official repositories. `tools/fit_run.py` reads the FIT code and its EFO-1 data from the directory in the environment variable `FIT_DIR` (see its docstring for the one-line change to FIT).

