# Phase 5: Post-Training Stack - Learning Notes

**Status**: In progress
**Started**: 2026-04-23
**Purpose**: important Phase 5 learnings only. Keep this compact; [PLAN.md](PLAN.md) remains the execution roadmap.

---

## Current Recommendation

Use the no-MoD P4 mixed-corpus recipe as the baseline for the next 32K-tokenizer comparison:

- 12L/1024H, `looped_num_blocks = 4`
- Interleaved `swa/mla/swa/mla`
- `attn_res_fused`, bf16, fused AdamW, torch compile
- Generalization filter: interval `4`, val batches `1`, damping `0.5`, preserve norm

Keep MoD separate until router/capacity tuning fixes its validation penalty.

---

## Key Decisions

| Decision | Outcome |
|---|---|
| Remove WikiText-103 from future corpora | Full Wikipedia covers it; avoid redundant exposure |
| Use OWT + FineWeb-Edu as Wave 0 web core | Apply hygiene, language filter, dedup, and repetition budgets first |
| Train 32K Unigram tokenizer | Better NL/code/math coverage; use for next stack comparison |
| Prefer looped-4 compiled shape | Similar early quality, lower memory, better throughput |
| Carry no-MoD + gen-filter forward | Best validated path so far |
| Treat MoD as separate tuning lane | Train loss matched, validation lagged on P4 slice |

---

## Compact Evidence

### Data Gates

| Probe | Result |
|---|---|
| OWT 5K docs + language filter + MinHash | 4963/5000 kept; dedup rate `0.0074` |
| OWT packing smoke | fill ratio `0.9980` |
| OWT + FineWeb-Edu repetition-budget smoke | dedup rate `0.0000`; fill ratio `0.9990` |

### Tokenizer

| Item | Result |
|---|---|
| 32K tokenizer corpus | 215K docs, ~287M tokens, 8 sources |
| Artifact | `/mnt/d/Dev/data/prepared/p5_wave1_tokenizer_32k_20260424/p5_wave1_tokenizer_32k_20260424_tokenizer.model` |
| Readout | Better NL fertility and better code/math symbol coverage vs 8K |

### Packed Training

| Arm | Val Loss | Throughput |
|---|---:|---:|
| Unpacked Flash | `5.6296` | `495k` tok/s |
| Packed SDPA fallback | `5.5286` | `267k` tok/s |
| Packed varlen Flash | `5.5286` | `460k` tok/s |

Learning: document-boundary masking improved quality on the smoke slice; varlen Flash recovered most throughput.

### Muon Proxy

| Arm | Final Val | Median Throughput |
|---|---:|---:|
| AdamW | `4.3605` | `280k` tok/s |
| Muon | `4.1062` | `214k` tok/s |
| Batched Muon | `4.1070` | `258k` tok/s |

Learning: Muon is sample-efficient but still needs larger-shape throughput-quality validation before replacing AdamW as default.

### Looped and Interleaved Shape

| Arm | Readout |
|---|---|
| 12L full physical blocks | `19.6k` tok/s, 9660 MB, best val `4.48 @300` |
| 12L looped-4 | `20.5k` tok/s, 8793 MB, best val `4.32 @300` |
| 12L looped-4 + compile/flash norm | `26.1k` tok/s, 9057 MB |
| Interleaved SWA/MLA vs pure MLA | Small validation win at equal speed/memory |

Learning: looped-4 + compiled flash-norm + interleaved SWA/MLA is the best current Phase 5 sweep shape.

### MoD Comparison

| Arm | Final Val | Readout |
|---|---:|---|
| No-MoD control | `4.890851` | baseline |
| No-MoD + gen-filter | `4.887799` | best 600-step arm |
| MoD control | `4.978868` | validation penalty despite matched train loss |
| MoD + gen-filter | `4.951251` | recovered ~31% of penalty, still behind no-MoD |

Learning: MoD needs router/capacity work before promotion.

### Generalization Filter

| Run | Result |
|---|---|
| 4L no-renorm damping | Slowed train and validation |
| 4L norm-preserving filter | Preserved curve while damping ~36% of probed elements |
| 12L 600-step no-MoD + filter | Slightly beat control at matched train loss/throughput |
| 12L 2000-step no-MoD + filter | Final val `4.056977`, ppl `57.80`, monotonic eval descent |

2000-step validation curve:

| Step | Val Loss |
|---:|---:|
| 200 | `5.673934` |
| 400 | `5.124903` |
| 600 | `4.837539` |
| 800 | `4.641930` |
| 1000 | `4.473479` |
| 1200 | `4.308452` |
| 1400 | `4.201900` |
| 1600 | `4.122359` |
| 1800 | `4.080792` |
| 2000 | `4.056977` |

Learning: norm preservation is essential. The filter appears to focus update direction without reducing overall learning speed.

---

## Next Work

1. Run the no-MoD + gen-filter baseline on the 32K tokenizer stack.
2. Complete selected Wave 1 production preps with the 32K tokenizer.
3. Add prompt templates / `ChatFormatter` for multi-turn inference.
4. Keep MoD router/capacity tuning separate from the main baseline.
5. Start SFT and grounding data products after the tokenizer/data baseline is stable.
