# BC-CGL Navier-Stokes Project (GitHub Ready)

This folder is prepared for direct upload to GitHub, including:
- source code
- paper draft
- experiment result files

## Folder structure

- `src/` : simulation and diagnostics code
- `results/` : experiment outputs included in this package
- `paper/` : LaTeX manuscript
- `theory/` : foundational theory materials (Marp slides)
- `scripts/` : helper scripts
- `docs/` : packaging notes

## Included experiment results

- `results/taylor_green/summary.json`
- `results/taylor_green/timeseries.csv`
- `results/random_iso/summary.json`
- `results/random_iso/timeseries.csv`
- `results/highRe/summary.json`
- `results/highRe/timeseries.csv`
- `results/results_tg_Re1000.csv`
- `results/results_tg_N256_Re2000.csv`
- `results/results_summary.md`

## Included theory materials

- `theory/marp/LoNalogy_marp.md`
- `theory/marp/LoNalogy_marp_rewritten.md`
- `theory/marp/LoNalogy_marp_rewritten.pdf`

## Included paper files

- `paper/paper_bc_cgl_ns.tex`
- `paper/BC-CGL_NS_Regularity_LoNalogy_v2_conditional.pdf`

## Environment

Python 3.10+ recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

Note for GPU JAX:
- install a `jax`/`jaxlib` build compatible with your CUDA driver and platform.
- see official JAX installation docs for your environment.

## Run examples

From repository root:

```bash
python src/run_experiment.py --N 64 --nu 0.01 --T 3.0 --exp tg
python src/run_enstrophy_budget.py
```

## GitHub push steps

```bash
cd NS_github_ready
git init
git add .
git commit -m "Initial commit: BC-CGL NS project with results"
git branch -M main
git remote add origin git@github.com:YOUR_NAME/YOUR_REPO.git
git push -u origin main
```

## Source provenance

Packaged from:
- `/Users/lona/Desktop/dualcomplex/NS`

Packaging date:
- 2026-02-17
