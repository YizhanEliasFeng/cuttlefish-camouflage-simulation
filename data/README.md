# Data Directory

The toy configs do not require external data: they generate a target texture from the same smooth basis family so the feedback loop can be tested reproducibly.

For real or controlled backgrounds, use this layout:

```text
data/
├── backgrounds_raw/
│   ├── dtd/          # downloaded Describable Textures Dataset
│   ├── procedural/   # generated checkerboard / grating / noise controls
│   └── self_collected/
├── backgrounds_processed/
│   └── 128_gray/     # square resized grayscale PNGs for experiments
└── manifests/
    └── backgrounds.csv
```

Large raw and processed images are ignored by git. Keep source URLs, licenses, and processing choices in manifest files.

## Commands

Download DTD:

```bash
.venv/bin/python -m src.download_dtd --output-root data/backgrounds_raw/dtd
```

Generate controlled artificial backgrounds:

```bash
.venv/bin/python -m src.procedural_backgrounds \
  --output data/backgrounds_raw/procedural \
  --size 256 \
  --seed 42
```

Preprocess procedural backgrounds:

```bash
.venv/bin/python -m src.data_prep \
  --input data/backgrounds_raw/procedural \
  --output data/backgrounds_processed/128_gray \
  --size 128 \
  --source procedural \
  --license generated \
  --manifest data/manifests/procedural_128_gray.csv
```

Preprocess DTD backgrounds:

```bash
.venv/bin/python -m src.data_prep \
  --input data/backgrounds_raw/dtd/dtd/images \
  --output data/backgrounds_processed/128_gray \
  --size 128 \
  --source dtd \
  --license see_dtd_terms \
  --manifest data/manifests/dtd_128_gray.csv
```
