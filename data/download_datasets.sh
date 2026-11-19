#!/usr/bin/env bash
# Downloads the real datasets used by FSAM into data_raw/ and builds data_processed/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
mkdir -p data_raw/prostate

# RIGA+ (Hu et al., Zenodo 6325549) — md5 4df43d5e8152776d32219dbb6b77d1e9
[[ -f data_raw/RIGAPlus.zip ]] || curl -L -o data_raw/RIGAPlus.zip https://zenodo.org/api/records/6325549/files/RIGAPlus.zip/content
echo "4df43d5e8152776d32219dbb6b77d1e9  data_raw/RIGAPlus.zip" | md5sum -c -
[[ -d data_raw/RIGAPlus/RIGA ]] || unzip -q data_raw/RIGAPlus.zip -d data_raw/RIGAPlus

# Multi-site prostate (Liu et al., SAML MICCAI 2020; https://liuquande.github.io/SAML/)
[[ -f data_raw/prostate/Processed_data_nii.zip ]] || \
  .venv/bin/gdown -O data_raw/prostate/Processed_data_nii.zip "https://drive.google.com/uc?id=1TtrjnlnJ1yqr5m4LUGMelKTQXtvZaru-"
echo "f03a4b77dbc39a36102435e73c28b649  data_raw/prostate/Processed_data_nii.zip" | md5sum -c -
[[ -d data_raw/prostate/RUNMC ]] || unzip -q data_raw/prostate/Processed_data_nii.zip -d data_raw/prostate

$PY data/preprocess_riga.py
$PY data/preprocess_prostate.py
$PY data/verify_datasets.py
