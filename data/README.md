# Data Directory

Place real or curated target backgrounds in `data/backgrounds/`.

The toy configs do not require external data: they generate a target texture from the same smooth basis family so the feedback loop can be tested reproducibly.

Suggested layout:

```text
data/
├── backgrounds/   # target images such as sand, gravel, coral
├── raw/           # original downloaded or extracted data
└── processed/     # resized or normalized experiment inputs
```

