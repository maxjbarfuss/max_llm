# Looped Attention Plan

> **Transitory document:** delete this file once the looped-attention work described here has been implemented, validated, and reflected in the permanent docs/config reference.

Purpose: preserve the Phase 5 looped-attention notes in a durable project document so future agents do not have to infer the plan from ephemeral configs or chat history.

## Goal

Looped attention means executing a smaller set of physical transformer blocks across a deeper logical stack. The immediate target is memory and parameter efficiency for Phase 5 shape sweeps: keep the logical depth and existing attention-residual behavior, but reuse block weights on a fixed schedule so large-context MLA + block-attn probes can fit more comfortably.

This is not a replacement for sequence packing, chunked loss, xIELU recompute, FFN chunking, or activation checkpointing. It is the next architecture-level lever after those memory slices.

## Current State

- `model.share_layer_weights = true` already reuses one physical block for every logical layer.
- KV-cache hardening already treats shared-weight models as `num_layers` logical cache slots, so each logical layer has its own K/V history even if the weights are shared.
- Standard, `full_attn`, and `block_attn` residual paths already select `self.blocks[0]` when `share_layer_weights` is enabled.
- The current boolean is too blunt for serious training: one physical block across the entire depth cuts capacity aggressively and is documented as unsuitable for real runs.

## Proposed Shape

Introduce a generalized looped execution mode that decouples logical depth from physical block count.

- Keep `num_layers` as the logical depth used by residual routing, KV caches, schedulers, and reporting.
- Add a physical block count field, tentatively `looped_num_blocks`, disabled by default.
- When disabled, keep the existing one-block-per-layer behavior exactly.
- When enabled, construct `looped_num_blocks` physical `TransformerBlock`s and map logical layer `i` to physical block `i % looped_num_blocks`.
- Preserve `share_layer_weights` as a backward-compatible alias for the extreme case `looped_num_blocks = 1`, but prefer the new explicit field in future configs.
- Keep per-logical-layer KV-cache slots. Shared physical weights must not imply shared cache state.
- Keep the first slice free of extra per-depth adapters or loop embeddings. Add those only if smoke runs show repeated-block collapse.

## Implementation Slices

1. Config and construction
   - Add `ModelConfig.looped_num_blocks: int | None = None` with validation: positive when set, `<= num_layers`, and only meaningful when `num_layers > 0`.
   - Update `LearningModel` block construction so the physical block count is `1` for `share_layer_weights`, `looped_num_blocks` when set, otherwise `num_layers`.
   - Add a helper like `_block_for_layer(layer_idx)` so standard, `full_attn`, `block_attn`, and `make_kv_cache()` all share the same mapping.

2. Correctness tests
   - Default config still creates `num_layers` physical blocks.
   - `looped_num_blocks=1` matches `share_layer_weights=True` when initialized from the same state.
   - `looped_num_blocks=K` creates exactly K physical blocks and executes `num_layers` logical passes.
   - KV-cache creation returns `num_layers` logical caches, not K physical caches.
   - Standard, `full_attn`, and `block_attn` residual paths all run with looped physical blocks.

3. Training and checkpointing compatibility
   - Confirm activation checkpointing modes still operate on logical layer indices.
   - Confirm DDP/FSDP wrapping sees only the physical parameters and does not require `find_unused_parameters=True` for normal looped configs.
   - Confirm checkpoints load both default and looped configs cleanly; document that changing looped physical count is not checkpoint-compatible without an explicit remap.

4. Smoke experiment
   - Start from the existing Phase 5 memory-envelope shape: 12 logical layers, 1024 hidden, MLA, xIELU, RoPE-500K, `res_type="block_attn"`, `attn_res_num_blocks=6`, Flash attention.
   - Compare baseline 12 physical blocks against looped variants such as 6 and 4 physical blocks.
   - Use a short packed-data smoke first, then a one-step memory envelope probe.
   - Record parameters, peak memory, tokens/sec, and early val-loss behavior in `outputs/ephemeral/`, then promote the result summary into `docs/PLAN.md`.

## Acceptance Gates

- Existing non-looped behavior is unchanged and covered by tests.
- `share_layer_weights=True` remains backward-compatible.
- Looped models use distinct logical KV caches and pass cached-generation equivalence tests where the attention type supports KV cache.
- A looped smoke run does not show immediate loss collapse versus the same logical-depth baseline.
- The final Phase 5 recommendation includes whether looped execution is a training default, an experimental memory probe, or a dead end.

## Open Questions

- Whether repeated physical blocks need small per-logical-depth adapters, norm scales, or loop embeddings to preserve capacity.
- Whether looped execution should be cyclic (`i % K`) or staged (`floor(i / repeat_count)`) for better optimization.
- Whether Muon should manage shared physical matrices differently from ordinary repeated-use parameters.
- Whether block-attn residual queries should stay per logical sublayer, as they do today, or share by physical block in a lower-parameter variant.