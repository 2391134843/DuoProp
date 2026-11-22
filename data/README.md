# Data

The benchmarks are the BetaE query sets for FB15k, FB15k-237 and NELL995. They are not stored in this repository.

    bash scripts/download_data.sh

Expected layout after unpacking:

    data/FB15k-237-betae/   data/FB15k-betae/   data/NELL-betae/

Each folder holds train.txt, valid.txt, test.txt, stats.txt, the id/relation mappings (ent2id.pkl, id2ent.pkl, rel2id.pkl, id2rel.pkl) and the query files {valid,test}-{queries,easy-answers,hard-answers}.pkl.
