#!/usr/bin/env bash
# Download the BetaE benchmark data (FB15k, FB15k-237, NELL995) into ./data
set -e
mkdir -p data
wget -c -P data http://snap.stanford.edu/betae/KG_data.zip
unzip -n -d data data/KG_data.zip
