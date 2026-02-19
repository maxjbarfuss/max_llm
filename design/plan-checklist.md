# Max LLM Session Checklist

Purpose: lightweight execution tracker for humans and AI agents.

Session rule: a session is all work since the last commit.

How to use:
1. Read this file at session start.
2. Work from **Next Steps**.
3. Keep temporary notes in **Current Session Scratch Pad**.
4. Add a short entry to **Running Session Log** before committing.

---

## Roadmap (High-Level)

- [x] Foundation setup (repo, docs baseline, config system)
- [ ] Core model blocks (embeddings, RoPE, MLA, transformer forward pass)
- [ ] Advanced model blocks (MoE, GRU, precision scheduler)
- [ ] Training system (data pipeline, distributed training, checkpointing, monitoring)
- [ ] Optimization and validation (compile, checkpointing strategy, smoke/perf tests)
- [ ] Training runs and evaluation (pre-train, fine-tune, post-train)

---

## Next Steps

### Immediate
1. Build `src/models/embeddings.py` with tests first (shape, scaling, weight tying).
2. Build `src/models/position.py` (RoPE) with correctness tests.
3. Build `src/models/attention/mla.py` (Q full, latent KV, cache behavior) with tests.
4. Run tokenizer benchmark slice (GPT-2 BPE vs Unigram) and record recommendation in `design/plan.md`.

### Soon
5. Implement `src/models/transformer.py` with Pre-LN residual structure.
6. Integrate basic forward pass (embeddings -> attention -> projection) with shape tests.

### Blocked
- None.

---

## Current Session Scratch Pad

- Session date: 2026-02-19
- Since commit: post-`48d2ff2`
- Active focus: Documentation cleanup, config refactoring, engineering principles elevation, Apache-2.0 licensing
- Decisions made: Config TOML-first + split per-class modules; docs summary-first style; simplified session tracker; Apache-2.0 for license
- Open questions: None
- Blockers: None
- Files touched: ~20 files (config system, design docs, governance docs, license, pyproject.toml)
- Before commit checklist:
  - [x] Tests updated and passing (static analysis 0 errors)
  - [x] Docs updated (README, CONTRIBUTING, philosophy, all crosspages)
  - [x] Add brief log entry below

---

## Running Session Log (Brief)

| Date | Since Commit | Summary |
|---|---|---|
| 2026-02-19 | post-`48d2ff2` | Config externalization (TOML split modules), docs/cross-ref normalization, principles elevation (Big-O/SOLID/reproducibility), Apache-2.0 license. Production-ready for model blocks. |
| 2026-02-18 | repo init | Initial project scaffolding, config/test baseline, and first design docs. |

---

## Canonical References

- `design/plan.md` (architecture + implementation plan)
- `design/philosophy.md` (engineering standards)
- `CONTRIBUTING.md` (workflow and validation gates)
- `config/*.toml` (authoritative runtime values)
