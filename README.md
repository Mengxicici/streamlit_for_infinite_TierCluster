
# interactive tool for single cell phenotyping

https://appforinfinitetiercluster-ajoxhaf28kdvklco7zkuj7.streamlit.app/

## Overview
this is a interactive tool for single cell phenotyping, can make infinite tiers of clustering based on selected marker intensity and morphology features (extracted from single cell segmentation results), the input is supposed to be anndata .pkl file, the satisfied clustering setting will be saved in .yaml file, and the output file will be .pkl file.

## Features
- come with visualization of spatial distribution of markers/clusters
- the visualization is interactable with plotly plugin
- come with the mann whitney utest p value to compare the selected features between groups
- st_cluster_v3 will visualize all regions in anndata, st_cluster_ROI_selection will only visualize selected ROIs to save time.

## Installation
To set up the environment for this project, follow these steps:

in current file path:
```bash

gitclone https://github.com/Mengxicici/streamlit_for_infinite_TierCluster.git


conda create --name st python=3.9
conda install -c conda-forge poppler
pip install -r requirements.txt

streamlit run st_cluster_v3.py
