# DuoProp

Code for **DuoProp: Dual-Consistent Atoms and Neighbourhood Propagation for Multi-Hop Logical Reasoning over Knowledge Graphs**.

DuoProp answers logical queries using a frozen link predictor. **Duo** combines the head-side and tail-side scores for each triple and scales predictions for entity–relation pairs with no observed answer. **Prop** updates entity scores after each projection using neighbours from the training graph and the embedding space.

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

## Run the FB15k-237 single model

The following example uses the standard protocol and the validation-selected ComplEx setting. The paper used Python 3.11 and PyTorch 2.5.1 with CUDA 12.1. Run the commands from the repository root on a machine with an NVIDIA GPU.

```bash
python -m pip install torch==2.5.1 numpy==1.26.4
mkdir -p data cache/art cache/mats runs/results
curl -fL https://snap.stanford.edu/betae/KG_data.zip -o data/KG_data.zip
unzip -n data/KG_data.zip -d data
```

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

## Source files

| Module in `duoprop/` | Purpose |
| --- | --- |
| `scorer.py` | Score triples using frozen ComplEx, SpherE, GQE or Query2Box checkpoints. |
| `match.py` | Fit per-relation matchability probes. |
| `build.py`, `build_blocked.py` | Build sparse Duo truth matrices. The blocked builder stages scores in host memory for larger graphs such as NELL995. |
| `reason.py` | Evaluate logical queries over sparse matrices and compute filtered ranks. |
| `collective_eval.py` | Construct entity neighbourhoods, apply propagation and select settings on validation. |
| `ensemble.py` | Average the sparse atom matrices of several runs before evaluation. |

Released under the [MIT licence](LICENSE).
