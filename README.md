# 3D Correspondences via 2D Image Encoders

**Beyond the Mean: per-view feature matching and a 2D-backbone comparison for lifted 3D correspondence on HouseCorr3D.**

Deep Learning Lab project, University of Freiburg (supervisor: Leonhard Sommer).
Authors: Tushar Samal, Smrithi V Sambrekar, Prashanth Premakumar.
Poster: [docs/poster.pdf](docs/poster.pdf)

## The problem

3D semantic correspondence means finding matching points between two different 3D objects of
the same category (e.g. the nozzles of two hair dryers) from shape and appearance alone, with
no manual labels.

The HouseCorr3D benchmark (GenIntel lab) evaluates this without training a matcher. A frozen 2D
vision foundation model is run on rendered views of each mesh, and its features are projected
onto the mesh vertices. A query point is matched to its nearest neighbour in feature space.

We ask two questions about that pipeline:

1. **Does averaging the features of all viewpoints discard useful information?** The original
   pipeline kept only one mean descriptor per vertex.
2. **Does the choice of 2D backbone matter?** We compare self-supervised image-only encoders
   (DINOv2, DINOv3) with image-text contrastive encoders (SigLIP2, TIPSv2).

## What this project adds

| Contribution | What changed | Where |
|---|---|---|
| Per-view feature retention (`all_views`) | Every per-view observation of every vertex can be kept instead of a running mean. Opt-in via `feats_aggregation_mode: "all_views"`; the default mean path is unchanged. Touches the feature accumulator, the on-disk cache and pair batching. | `o3b` fork: `data/datatypes/{mesh,object,frame}.py`, `model/diff3f/` |
| Min-pairwise matching | `Crsp3DNNTask` gains `match_mode` (`mean` or `min_pairwise`) and `dist_type` (`euclidean` or `cosine`). `min_pairwise` compares a query against every retained per-view feature and keeps the closest one. | `o3b` fork: `task/crsp3d_nn/task.py` |
| Backbone integration | DINOv3, SigLIP2 and TIPSv2 added as pluggable feature extractors. | `o3b` fork: `model/{dinov3,siglip2,tipsv2}/` |
| Ablation configs | Backbone x matching-mode x distance x category sweeps. | [src/configs/ablation/](src/configs/ablation/) |
| Analysis notebooks | PCK comparison and visualisation. | [PCK_Comparison_of_0_1_and_0_05.ipynb](PCK_Comparison_of_0_1_and_0_05.ipynb), [visualisation_notebook.ipynb](visualisation_notebook.ipynb) |

### Who did what

- **Tushar Samal:** `all_views` feature aggregation, `min_pairwise` matching, SigLIP2 integration, and wiring all_views support into SigLIP2, DINOv3 and TIPSv2.
- **Prashanth Premakumar:** DINOv3 feature model and ablation config, PCK comparison notebook.
- **Smrithi V Sambrekar:** TIPSv2 feature model and ablation config, evaluation notebook.
- **Not ours:** the HouseCorr3D benchmark, the `o3b` framework and the original Diff3F pipeline are by the GenIntel lab ([GenIntel/housecorr3d](https://github.com/GenIntel/housecorr3d), [GenIntel/o3b](https://github.com/GenIntel/o3b)).

## Pipeline

```
mesh -> remesh (mc16) -> render 4 RGB views (256x256) -> frozen 2D backbone
     -> project pixel features to vertices -> [mean | keep all views]
     -> nearest-neighbour matching [mean | min-pairwise] -> PCK and distance metrics
```

All reported results use 4 views at 256x256 px per object (`mc16_vuni4_r256_*`), chosen to keep
rendering and feature extraction affordable. Whether the rankings hold at 100 views is untested.

## Results

Source: [poster](docs/poster.pdf). The poster's category table covers dinosaur, hair dryer,
teapot, toy animals and toy plane, out of 153 categories in the dataset.

| Encoder | PCK@0.05 | PCK@0.1 | Min-pairwise PCK@0.1 |
|---|---|---|---|
| DINOv2-S | 0.1436 | 0.2906 | 0.1496 |
| DINOv2-B | 0.1436 | **0.3137** | 0.1485 |
| DINOv3-B | **0.1511** | 0.2966 | **0.2759** |
| SigLIP2 | 0.1282 | 0.2654 | 0.1939 |
| TIPSv2 | 0.1425 | 0.2693 | 0.2491 |

- DINOv3-B is the strongest overall: best PCK@0.05 and by far the best under min-pairwise matching. DINOv2-B is slightly ahead on the looser mean-matching PCK@0.1.
- No encoder wins every category: DINOv2-S on dinosaur, TIPSv2 on hair dryer, SigLIP2 on teapot, DINOv2-B on toy animals and toy plane.
- Hair dryer is the easiest category (PCK@0.1 of 0.4816) and teapot the hardest by Euclidean error (0.6454).
- On one qualitative DINOv3 example, mean aggregation beat min-pairwise (37.3% vs 34.8%). Min-pairwise helps the newer encoders much more than DINOv2.

## Repository layout

```
src/housecorr3dv2/   Bench, Method and Task classes for this benchmark (cli.py, run.py entry points)
  bench/             object-pair and frame-object-pair benchmarks
  method/            DefaultMethod (identity pass-through) and a FunctionalMaps stub
src/configs/
  bench/             benchmark definitions (hc3d_*, dm_*)
  ablation/          feature_model, task (match mode x distance), category, mesh_type sweeps
third_party/o3b/     git submodule: our o3b fork (this is where the core code changes live)
tests/               feature extraction and mesh conversion tests
docs/poster.pdf      project poster
DATASETS.md          dataset notes
```

**Note on the submodule:** the model and task changes live in our `o3b` fork, on the
[`dinov3-integration`](https://github.com/TusharSamal3012/o3b/tree/dinov3-integration) branch
(that is the commit this repo pins). The fork's `main` branch is unmodified upstream.

## Setup

Requires a CUDA GPU. The setup script targets a SLURM cluster.

```bash
git clone --recurse-submodules git@github.com:TusharSamal3012/Housecorr3D-v2.git
cd Housecorr3D-v2
bash setup/setup_slurm.sh
```

If you cloned without submodules: `git submodule update --init --recursive`.

### Datasets

```bash
o3b dataset fetch -d hc3d_object
o3b dataset index -d hc3d_object
o3b dataset index -d hc3d_object_pair
o3b dataset viz -d hc3d_object
```

### Run a benchmark

```bash
# locally
o3b bench run -b hc3d_crsp3d_object_pair_nn -a category/housecorr3d_5

# per-view matching with one backbone (comma-separated ablations are combined;
# a directory name expands to every config inside it)
o3b bench run -b hc3d_crsp3d_object_pair_nn \
  -a feature_model/mc16_vuni4_r256_fdinov3b_allviews.yaml,task/crsp3d_min_pairwise_cosine.yaml

# on SLURM
o3b bench rrun -p slurm_lmbl40 -b hc3d_crsp3d_object_pair_nn -a category/housecorr3d_5
```

## Reference

Benchmarking Semantic Object Correspondence in Vision Foundation Models, arXiv:2605.31597.
