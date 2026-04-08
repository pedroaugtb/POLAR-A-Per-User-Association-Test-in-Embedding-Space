# POLAR: A Per-User Association Test in Embedding Space

This repository contains the public implementation of POLAR, a per-user lexical association test in embedding space built on top of a lightly adapted masked language model.

The code accompanies the paper "POLAR: A Per-User Association Test in Embedding Space", accepted at ICWSM 2026. While the conference proceedings are not yet available, the public preprint is on arXiv:

- https://arxiv.org/abs/2603.15950

At a high level, POLAR:

1. Train a masked language model with one hashed user token per author.
2. Represent each author with a private deterministic token in the embedding space.
3. Score each learned user embedding against curated lexical axes.
4. Report standardized effects with permutation p-values and Benjamini-Hochberg correction.

The repository is dataset-agnostic. Private paths and local machine assumptions were removed, while the exact settings used in the paper's bot/human and hate-speech experiments are preserved in the `configs/` directory.

## Repository Layout

- `trainer.py`: generic training entrypoint
- `polar.py`: generic scoring entrypoint
- `configs/train.bot_human.paper.json`: paper training config for the bot/human experiment
- `configs/train.hate.paper.json`: paper training config for the hate-speech experiment
- `configs/polar.bot_human.paper.json`: paper scoring config for the bot/human experiment
- `configs/polar.hate.paper.json`: paper scoring config for the hate-speech experiment
- `configs/attributes.bot_human.paper.json`: attribute word sets used in the bot/human experiment
- `configs/attributes.hate.paper.json`: attribute word sets used in the hate-speech experiment
- `CITATION.cff`: citation metadata for GitHub and reference managers

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Input Formats

`trainer.py` supports two dataset formats:

1. `json_tweets`
- Use `input_json`
- Expected rows contain a user id plus a `tweets` or `user_tweets` list
- Tweet text is read from `full_text`, `extended_tweet.full_text`, or `text`

2. `txt_csv`
- Use `txt_dir` and `annotations_csv`
- The CSV must contain `file_id` and `user_id`
- Each `file_id` should map to a text file named `<file_id>.txt` inside `txt_dir`

## Usage

Train with a config:

```bash
python3 trainer.py --config configs/train.bot_human.paper.json
python3 trainer.py --config configs/train.hate.paper.json
```

Score a trained model:

```bash
python3 polar.py --config configs/polar.bot_human.paper.json
python3 polar.py --config configs/polar.hate.paper.json
```

These example configs correspond to the two experiments described in the paper:

- a balanced bot vs. human Twitter benchmark
- an extremist forum dataset used for hate-speech and ideological analysis

## Expected Outputs

Training writes:

- `model/`
- `users.csv`
- `meta.json`
- `user_embeddings.kv` when `export_kv=true`

Scoring writes:

- `per_user_scores.csv`

## Adapting To New Data

Edit a copy of one of the training configs:

- Change `input_format`
- Point the input paths to your dataset
- Choose a new `out_dir`

Edit a copy of one of the scoring configs:

- Point `out_dir` to the trained run
- Point `attributes_path` to your attribute set JSON

Attribute files must be JSON dictionaries in this shape:

```json
{
  "pair_name": {
    "pos": ["word_a", "word_b"],
    "neg": ["word_c", "word_d"]
  }
}
```

This makes it easy to define new lexical axes for other domains, communities, or research questions.

## Citation

If you use this repository, please cite the paper. Until the ICWSM 2026 proceedings are online, prefer the arXiv version:

```bibtex
@misc{bento2026polar,
  title={POLAR: A Per-User Association Test in Embedding Space},
  author={Pedro Bento and Arthur Buzelin and Arthur Chagas and Yan Aquino and Victoria Estanislau and Samira Malaquias and Pedro Robles Dutenhefner and Gisele L. Pappa and Virgilio Almeida and Wagner Meira Jr},
  year={2026},
  eprint={2603.15950},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  doi={10.48550/arXiv.2603.15950},
  url={https://arxiv.org/abs/2603.15950}
}
```

## Notes

- User identifiers are converted to hashed tokens such as `usr<sha1[:10]>`.
- The paper configs preserve the released experimental settings, but the input paths are portable placeholders instead of private machine-specific paths.
- The word lists in `configs/attributes.*.paper.json` are the paper's released lexical axes.
- `gensim` is only needed if you enable `export_kv`.
