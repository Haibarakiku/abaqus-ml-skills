# Abaqus ML Skills

Claude Code agent skills for **bridging Abaqus FEA and machine-learning surrogate models**. Drop these into `.claude/skills/` and Claude can drive the full pipeline from design-space sampling → batch FEA → ML-ready training matrices.

These skills fill a gap not covered by the existing Abaqus skill packages on GitHub (which focus on single-case FEA workflows): **producing thousands of FEA samples and reshaping them into matrices that scikit-learn / PyTorch can consume directly**.

## Skills

### [`abaqus-lhs-batch-dataset`](skills/abaqus-lhs-batch-dataset/)

Generate a multi-case Abaqus FEA dataset for surrogate-model training. Latin Hypercube Sampling (or sparse-pattern / uniform random) over a parameterized design vector, one case folder per sample, batch-submit Abaqus jobs via `subprocess`, recover from crashes, and write a unified `dataset_index.csv`.

**Activates on**: "build a training set for a surrogate model", "sweep design parameters in Abaqus", "run N FEA simulations", "sample a design space".

### [`abaqus-odb-to-grid-csv`](skills/abaqus-odb-to-grid-csv/)

Convert per-case Abaqus FEA outputs into ML-ready (X, Y) wide-table CSVs. Pivots irregular FEA mesh node displacements onto a regular N×N grid via direct binning (structured mesh) or bilinear resampling, picks the final frame as the deformation target, and aggregates across many cases into `X_amplitude.csv` (design vectors) + `Y_grid_uz.csv` (flattened grid displacement).

**Activates on**: a folder of completed FEA cases that needs to become a training matrix for Ridge / MLP / Gaussian Process / PyTorch.

## Installation

### Per-project (recommended)

```bash
cd <your-project>
git clone https://github.com/<owner>/abaqus-ml-skills.git .claude/abaqus-ml-skills
ln -s .claude/abaqus-ml-skills/skills/abaqus-lhs-batch-dataset .claude/skills/
ln -s .claude/abaqus-ml-skills/skills/abaqus-odb-to-grid-csv  .claude/skills/
```

Or just copy the `skills/abaqus-*` folders into your project's `.claude/skills/`.

### User-global

```bash
git clone https://github.com/<owner>/abaqus-ml-skills.git
cp -r abaqus-ml-skills/skills/abaqus-* ~/.claude/skills/
```

## Pipeline

```
                ┌────────────────────────────┐
                │  template_dir/             │
                │   Parameters.dat           │
                │   ChannelParameters.dat    │
                │   InData.txt, ...          │
                │   solver_script.py         │
                └────────────┬───────────────┘
                             │
                             │  abaqus-lhs-batch-dataset
                             │   - sample design space (LHS / sparse / uniform)
                             │   - per-sample case dir + ForceAmplitude.dat
                             │   - subprocess.run("abaqus cae noGUI=...")
                             │   - dataset_index.csv with status + retcodes
                             ▼
                ┌────────────────────────────┐
                │  datasets/<run>/           │
                │   sample_00001/.odb        │
                │   sample_00002/.odb        │
                │   ...                      │
                │   dataset_index.csv        │
                └────────────┬───────────────┘
                             │
                             │  abaqus-odb-to-grid-csv
                             │   - read final frame uz per case
                             │   - bin onto regular N x N grid
                             │   - bilinear resample to target N
                             │   - dedupe by input
                             ▼
                ┌────────────────────────────┐
                │  aggregated/v1/            │
                │   X_amplitude.csv          │  <- ML model input
                │   Y_grid_uz.csv            │  <- ML model target
                │   sample_meta.csv          │
                │   aggregate_summary.csv    │
                └────────────────────────────┘
                             │
                             ▼
                  scikit-learn / PyTorch
                  Ridge / MLP / GP / etc.
```

## Requirements

- Abaqus 2021+ on `PATH` (verify with `abaqus --help`)
- Python 3.8+ for orchestration (the Abaqus solver script itself runs in Abaqus's bundled Python)
- `numpy` for the post-processing skill; `scipy` only if you need unstructured-mesh resampling
- Claude Code 2.x or any agent that supports the [Agent Skills](https://github.com/anthropics/skills) format

## Design-space dimension and grid size

Both skills are parameterized — the design vector dimension `D` is read from `ChannelParameters.dat`, and the target learning grid `N×N` is a CLI arg. There's nothing hardcoded about `D=16` or `N=21`. The skills work for arbitrary `(D, N)` as long as your FEA template + solver script are consistent.

## Origin

The skills were extracted from the [ShapeProgramming](https://github.com/Haibarakiku/ShapeProgramming) project's static 2D 4×4 magnetic-actuator pipeline (Ridge / MLP surrogate trained on 3000+ Abaqus samples, 16-channel design vector → 21×21 displacement field). Cleaned up and generalized for arbitrary parameterized membrane / plate / shell problems.

## License

MIT — see [LICENSE](LICENSE).

## Contributing

Issues and PRs welcome. The skills aim to stay **thin and self-contained**: pure stdlib for the LHS skill, only `numpy` (and optional `scipy`) for the post-processor. If you have a related skill (e.g. `abaqus-multichannel-actuator-loading`, `abaqus-dynamic-ramp-template`, `abaqus-surrogate-fea-validation`), open a PR.
