# NuminaMath-CoT

Source: `AI-MO/NuminaMath-CoT` (HuggingFace)

859,494 competition mathematics problems with chain-of-thought solutions spanning
arithmetic, algebra, geometry, number theory, and combinatorics. Problems are drawn
from `synthetic_math`, `aops_forum`, and related sources.

## Preprocessing

`preprocess.py` downloads the train split and combines `problem` + `solution` fields
into a single `text` column:

```
Problem: {problem}

Solution: {solution}
```

This makes the reasoning chain visible to the model as continuous prose.

Output: `/mnt/d/Dev/data/numina_math_cot/train/*.parquet` (~18 shards of 50K rows each)

```bash
python -m src.data.datasets.numina_math.preprocess
```

## Provenance note

NuminaMath-CoT is synthetic/semi-synthetic data (competition problems with
model-assisted or human-written solutions). It should be treated similarly to
Cosmopedia — included as an explicitly tracked companion source, not a primary corpus.

## Prep pipeline config fields

```toml
[[datasets]]
name = "reasoning_numina_math"
path = "/mnt/d/Dev/data/numina_math_cot/train"
format = "parquet"
text_field = "text"
```
