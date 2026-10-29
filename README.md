# Attention animations

Manim Community scenes for an IDL attention lab. Three scenes walk through scaled dot-product attention: one query against each key, softmax into weights, then the full matrix form.

## Scenes

- **SceneA_DotProduct** — Similarity score. One query (`Q₁`) is compared with each key. The scene shows the elementwise product, sums it into a score, and highlights the most similar key.
- **SceneB_Softmax** — Similarity score → attention weights. Scores are optionally divided by √dₖ, exponentiated, then normalized so the weights sum to 1. A bar chart tracks the same values.
- **SceneC_Matrix** — Matrix similarity score. Queries and keys are stacked into `Q` and `K`, `K` is transposed, and `S = QKᵀ`. Row-wise softmax produces the attention matrix; each row is how much one query attends to every key.

## Install

```bash
pip install manim
```

Rendering needs [ffmpeg](https://ffmpeg.org/) on `PATH`. No LaTeX install is required: labels use Manim `Text` / `MarkupText` with Unicode (subscripts, √, Σ), not `Tex`.

## Render

Full 1080p, 30 fps (writes under `media/`):

```bash
SEED=31 SCALE=1 manim -r 1920,1080 --fps 30 --disable_caching attention_scenes.py SceneA_DotProduct SceneB_Softmax SceneC_Matrix
```

Quick preview (low quality, open the last file when it finishes):

```bash
manim -pql attention_scenes.py SceneA_DotProduct SceneB_Softmax SceneC_Matrix
```

Render a single scene by passing only that class name.

## Environment variables

Read at import time. Restart the render after changing them.

| Variable | Default | Meaning |
| --- | --- | --- |
| `SEED` | `31` | NumPy Generator seed for the random `Q` and `K` entries (rounded to 1 decimal). |
| `SCALE` | `1` | `1` divides scores by √dₖ before softmax. `0` skips that scale (plain softmax on raw scores). |
| `D` | `4` | Vector size (columns of `Q` and `K`; dₖ). |
| `N` | `4` | Number of tokens / keys (rows of `Q` and `K`). |

## Deck colors

Cell fills follow the lecture deck. Shade intensity encodes `|value|`; negatives use purple.

| Role | Hex |
| --- | --- |
| Q (query) | `#FFAB40` |
| K (key) | `#6AA84F` |
| V (value) | `#4285F4` |
| S (scores) | `#B5AA48` |

## Font

Labels use **Arimo**, a metric-compatible stand-in for the deck font **Arial**. Install Arimo (or Arial) on the machine that renders, or Manim falls back to another sans font and spacing can shift.
