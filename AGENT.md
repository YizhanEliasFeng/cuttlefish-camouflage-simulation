# AGENT.md

## 1. Role

You are Codex acting as a research-engineering assistant for an AI for Science project.

Your task is to implement, organize, and iteratively improve a reproducible Python project that simulates cuttlefish camouflage as feedback optimization in a high-dimensional virtual skin-pattern space.

The project is not a generic image-generation project. It is a mechanistic toy model designed to connect biological camouflage dynamics with computational optimization. Prioritize interpretability, clean experiment management, and trajectory analysis over visual realism alone.

## 2. Project Goal

Build a simplified white-box biological agent with the following closed-loop structure:

```text
Target Background
        ↓
Vision / Loss Module
        ↓
Optimizer / Controller
        ↓
Activation Vector
        ↓
Virtual Skin Renderer
        ↓
Rendered Skin State
        ↺ back to Vision / Loss Module
```

The core modeling idea is:

```text
cuttlefish color change ≈ iterative feedback optimization in a high-dimensional skin-pattern space
```

The model should demonstrate whether different optimizer-like control rules can produce trajectories that resemble biological camouflage dynamics: high-dimensional search, curved paths, intermittent movement, velocity changes, and convergence toward a stable skin pattern.

## 3. Non-Negotiable Modeling Principles

### 3.1 Do not optimize free pixels as the main model

The virtual skin must be controlled by a lower-dimensional activation vector over basis patterns.

Allowed:

```text
skin = render(basis, weights)
```

Not allowed as the main model:

```text
skin_pixels = directly optimized image tensor
```

Free-pixel optimization can be used only as a diagnostic baseline, and must be clearly labeled as such.

### 3.2 Always preserve trajectory data

Every simulation run must save the full activation trajectory.

At minimum, save:

- `weights.npy` or `weights.csv`
- `loss.csv`
- rendered frames or selected checkpoint images
- run configuration
- random seed
- optimizer type and hyperparameters

### 3.3 Separate biological interpretation from engineering implementation

Code should be organized so that:

- basis construction is separate from rendering;
- rendering is separate from loss computation;
- loss computation is separate from optimizer update;
- evaluation is separate from simulation;
- visualization is separate from numerical computation.

Do not write one large monolithic script unless it is a temporary prototype.

### 3.4 Prefer reproducibility over cleverness

Every experiment must be reproducible from a config file or explicit command-line arguments.

All random behavior must use a stored random seed.

## 4. Expected Repository Structure

Create or maintain the following structure:

```text
cuttlefish-camouflage/
├── README.md
├── AGENT.md
├── requirements.txt
├── data/
│   ├── backgrounds/
│   ├── raw/
│   └── processed/
├── configs/
│   ├── toy_adam.yaml
│   ├── toy_sgd.yaml
│   ├── toy_langevin.yaml
│   └── perceptual_vgg16.yaml
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── basis.py
│   ├── render.py
│   ├── losses.py
│   ├── controllers.py
│   ├── simulation.py
│   ├── evaluation.py
│   ├── visualization.py
│   └── utils.py
├── outputs/
│   ├── runs/
│   ├── figures/
│   ├── gifs/
│   └── logs/
└── notebooks/
    └── exploratory_analysis.ipynb
```

If the existing repository has a different structure, do not destroy it. Instead, adapt the structure incrementally and document any changes.

## 5. Implementation Modules

### 5.1 `basis.py`

Responsible for constructing skin-pattern basis functions.

Required functions:

```python
create_random_smooth_basis(
    n_basis: int,
    height: int,
    width: int,
    low_res: int,
    seed: int,
) -> torch.Tensor
```

Expected output shape:

```text
[n_basis, 1, height, width]
```

Optional future functions:

```python
create_nmf_basis(...)
create_vae_basis(...)
load_basis(...)
save_basis(...)
```

Implementation requirements:

- random basis should be smooth, not pure high-frequency noise;
- use bicubic interpolation or Gaussian smoothing;
- normalize basis values to a stable range;
- keep basis generation deterministic under a fixed seed.

### 5.2 `render.py`

Responsible for converting activation weights into a virtual skin image.

Required function:

```python
render_skin(
    weights: torch.Tensor,
    basis: torch.Tensor,
    clamp: bool = True,
) -> torch.Tensor
```

Expected behavior:

```text
weights shape: [n_basis]
basis shape: [n_basis, 1, H, W]
output shape: [1, H, W] or [1, 1, H, W]
```

Recommended rendering rule:

```text
skin = sigmoid(sum_i weights_i * basis_i)
```

or another clearly documented bounded mapping.

### 5.3 `losses.py`

Responsible for computing visual feedback loss.

Required losses:

1. MSE loss for toy model.
2. Perceptual loss using frozen VGG16 features.

Required functions:

```python
mse_loss(rendered_skin, target)
perceptual_loss(rendered_skin, target, model, layers)
get_vgg16_feature_extractor(layers)
```

Implementation notes:

- VGG16 must be frozen.
- Do not train VGG16.
- Convert grayscale skin to 3-channel input if needed.
- Normalize images using ImageNet statistics when using VGG16.
- Allow MSE-only runs for fast debugging.

### 5.4 `controllers.py`

Responsible for optimizer-like control rules.

Required controllers:

1. Adam
2. SGD
3. Langevin-like update

Optional advanced controller:

4. Neural ODE

For Langevin dynamics, implement a gradient-based update with additive noise:

```text
w_{t+1} = w_t - eta * grad(loss) + sigma * noise
```

Where:

- `eta` is learning rate;
- `sigma` is noise strength;
- `noise` is Gaussian noise with fixed seed control.

Do not overclaim biological realism. Label this as a Langevin-like approximation of stochastic neural control.

### 5.5 `simulation.py`

Responsible for running a full closed-loop simulation.

Required behavior:

- load config;
- load or generate target background;
- generate or load basis;
- initialize weights;
- run optimization for `T` steps;
- save weights, losses, frames, and config;
- return a structured result object or dictionary.

Recommended function:

```python
run_simulation(config: dict) -> dict
```

Required saved outputs per run:

```text
outputs/runs/{run_id}/
├── config.yaml
├── weights.npy
├── losses.csv
├── frames/
│   ├── frame_0000.png
│   ├── frame_0010.png
│   └── ...
├── target.png
└── final_skin.png
```

### 5.6 `evaluation.py`

Responsible for quantitative trajectory analysis.

Required functions:

```python
compute_pca_trajectory(weights: np.ndarray, n_components: int = 2)
compute_velocity(weights: np.ndarray)
compute_path_length(pca_trajectory: np.ndarray)
compute_curvature_proxy(pca_trajectory: np.ndarray)
```

Optional function:

```python
compute_dtw_distance(simulated_trajectory, biological_trajectory)
```

Evaluation metrics should include:

- final loss;
- convergence speed;
- path length in PCA space;
- velocity profile;
- low-velocity or stagnation intervals;
- curvature proxy;
- optional DTW distance.

### 5.7 `visualization.py`

Responsible for creating figures and GIFs.

Required outputs:

1. GIF of skin convergence.
2. Loss curve.
3. PCA trajectory plot.
4. Optional velocity-over-time plot.

Required functions:

```python
make_gif(frame_dir, output_path, fps)
plot_loss_curve(loss_csv, output_path)
plot_pca_trajectory(pca_coords, output_path)
plot_velocity_curve(velocity, output_path)
```

Do not use overly decorative plot styles. Keep figures clean and publication-oriented.

## 6. Config Requirements

All major settings should be controlled through YAML config files.

Example config:

```yaml
run_name: toy_adam
seed: 42

image:
  height: 128
  width: 128
  channels: 1

basis:
  type: random_smooth
  n_basis: 32
  low_res: 16

target:
  type: image
  path: data/backgrounds/sand_01.png

loss:
  type: mse
  perceptual_layers:
    - relu3_3

optimizer:
  type: adam
  learning_rate: 0.05
  noise_sigma: 0.0

simulation:
  steps: 500
  save_every: 10

output:
  root: outputs/runs
```

Create separate configs for:

- `toy_adam.yaml`
- `toy_sgd.yaml`
- `toy_langevin.yaml`
- `perceptual_vgg16.yaml`

## 7. Command-Line Interface

Implement a command-line entry point such as:

```bash
python -m src.main --config configs/toy_adam.yaml
```

The command should:

1. read the config;
2. run the simulation;
3. save outputs;
4. print the output directory.

Optional commands:

```bash
python -m src.main --config configs/toy_langevin.yaml --seed 123
python -m src.evaluation --run outputs/runs/toy_adam_20260603_001
python -m src.visualization --run outputs/runs/toy_adam_20260603_001
```

## 8. File Naming Rules

Use stable, readable, and sortable names.

Recommended run ID:

```text
{run_name}_{YYYYMMDD_HHMMSS}_seed{seed}
```

Example:

```text
toy_adam_20260603_153000_seed42
```

Do not overwrite old runs unless explicitly instructed.

Do not save experimental outputs directly into the project root.

## 9. Minimum Viable Product

The first complete version must satisfy all of the following:

- generate 32 smooth basis patterns;
- render a virtual skin from a 32-dimensional activation vector;
- optimize the activation vector toward one target background;
- support Adam and SGD;
- optionally support Langevin-like noise;
- save all weights across time;
- generate a GIF;
- generate a PCA trajectory plot;
- generate a loss curve.

Do not start with Neural ODE before the MVP is stable.

## 10. Recommended Development Order

Follow this order strictly unless there is a strong reason not to:

1. Build `basis.py`.
2. Build `render.py`.
3. Build MSE loss in `losses.py`.
4. Build Adam or SGD simulation loop.
5. Save frames, weights, losses, and config.
6. Build GIF generation.
7. Build PCA trajectory visualization.
8. Add Langevin-like noise.
9. Add VGG16 perceptual loss.
10. Add optimizer comparison script.
11. Add optional DTW or biological trajectory comparison.
12. Consider Neural ODE only after the above steps work.

## 11. Biological Interpretation Guardrails

When writing comments, docstrings, README updates, or result summaries, follow these rules:

- Say “Langevin-like” rather than claiming this is a true neural Langevin process.
- Say “biologically inspired” rather than “biologically proven.”
- Say “virtual skin basis” rather than “real chromatophore motor units” unless actual biological motor-unit data are used.
- Say “trajectory resemblance” rather than “validation” when comparing to published cuttlefish data.
- Do not claim the model explains the full neural mechanism of cuttlefish camouflage.
- Emphasize that this is a simplified computational model for hypothesis generation.

## 12. Experimental Questions to Support

The codebase should make it easy to answer these questions:

1. Does the virtual skin visually converge toward the target background?
2. Do different optimizers produce different trajectory shapes?
3. Does Langevin-like noise generate more tortuous or intermittent paths?
4. Does perceptual loss produce trajectories different from MSE loss?
5. How do learning rate and noise strength affect convergence and stability?
6. Are simulated PCA trajectories visually comparable to biological trajectories from the literature?

## 13. Testing and Sanity Checks

Before considering a feature complete, run sanity checks:

### Basis sanity check

- basis tensor has correct shape;
- values are finite;
- basis images are smooth and visually distinguishable.

### Render sanity check

- rendered skin has expected shape;
- values are bounded;
- changing weights changes the image.

### Loss sanity check

- loss decreases on a simple target under Adam;
- loss is finite at every step.

### Trajectory sanity check

- weights array has shape `[steps + 1, n_basis]`;
- PCA output has shape `[steps + 1, 2]`;
- plots are saved correctly.

### Reproducibility sanity check

- two runs with the same seed produce the same initial basis and initial weights;
- configs are saved with outputs.

## 14. Dependency Policy

Keep dependencies minimal.

Recommended core dependencies:

```text
torch
torchvision
numpy
pandas
scikit-learn
Pillow
matplotlib
imageio
pyyaml
tqdm
```

Optional dependencies:

```text
scipy
torchdiffeq
opencv-python
```

Do not add heavy dependencies unless clearly needed.

## 15. Output Interpretation

When generating summaries from results, structure them around:

1. visual convergence;
2. loss dynamics;
3. PCA trajectory shape;
4. velocity and stagnation;
5. parameter interpretability;
6. biological comparability.

Avoid treating lower loss as the only success criterion. A controller that reaches low loss through a direct straight path may be less biologically interesting than one that produces curved, intermittent, but still convergent dynamics.

## 16. Final Deliverables Expected from Codex

Codex should help produce:

1. A runnable project skeleton.
2. A toy model simulation.
3. Configurable optimizer experiments.
4. Saved run artifacts.
5. GIF visualization.
6. PCA trajectory figures.
7. Loss and velocity plots.
8. Clear documentation for reproducing experiments.
9. Clean code with modular structure.
10. No hidden state, no undocumented outputs, no overwritten runs.

## 17. Coding Style

Use clear, boring, reliable Python.

- Prefer explicit function names.
- Use type hints where practical.
- Avoid excessive abstraction.
- Avoid premature optimization.
- Use docstrings for public functions.
- Keep scripts runnable from the command line.
- Fail loudly with useful error messages when config fields are missing.
- Never silently ignore missing input files.

## 18. Immediate Next Task

Start by creating the MVP:

```bash
python -m src.main --config configs/toy_adam.yaml
```

This should produce:

```text
outputs/runs/{run_id}/config.yaml
outputs/runs/{run_id}/weights.npy
outputs/runs/{run_id}/losses.csv
outputs/runs/{run_id}/frames/
outputs/runs/{run_id}/target.png
outputs/runs/{run_id}/final_skin.png
outputs/gifs/{run_id}.gif
outputs/figures/{run_id}_loss.png
outputs/figures/{run_id}_pca.png
```

Only after this pipeline works should more advanced components be added.
