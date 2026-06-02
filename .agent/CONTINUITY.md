[PLANS]
- 2026-06-02T05:08:54-04:00 [USER] Build `math-reasoning-agent` with source/logs/env in `/home/pham156/math-reasoning-agent` and large models/data/cache/outputs in `/scratch/gilbreth/pham156/math-reasoning-agent`.

[DECISIONS]
- 2026-06-02T05:08:54-04:00 [CODE] Real pipeline mode must fail clearly if Qwen files are missing; only `--mock_llm` may use deterministic mock generation.
- 2026-06-02T05:08:54-04:00 [CODE] A requested `/home/pham156/math-reasoning-agent/models/...` model path is remapped to the scratch model directory when the home path is empty, preserving the large-file storage rule.

[PROGRESS]
- 2026-06-02T05:08:54-04:00 [CODE] Implemented real Qwen2.5-Math-7B-Instruct generation path using `tokenizer.apply_chat_template` when available, `device_map="auto"`, and CUDA bf16 when supported.
- 2026-06-02T05:08:54-04:00 [CODE] Added `scripts/run_real_qwen_smoke.slurm` for one-GPU Gilbreth smoke testing with logs in home and outputs in scratch.
- 2026-06-02T05:10:00-04:00 [TOOL] Submitted real Qwen smoke Slurm job `10940001` via `scripts/run_real_qwen_smoke.slurm`; initial state is `PENDING` on `a100-40gb` with reason `AssocGrpGRES`.
- 2026-06-02T05:11:00-04:00 [USER] User provided preferred Gilbreth directives: account `ai-forge`, partition `a100-80gb`, QoS `standby`, one GPU, 64G memory, 4-hour limit.
- 2026-06-02T05:11:00-04:00 [TOOL] Canceled job `10940001`, updated `scripts/run_real_qwen_smoke.slurm`, and submitted corrected job `10940015`; it is `PENDING` on `a100-80gb` under `ai-forge/standby` with reason `Priority`.
- 2026-06-02T08:45:00-04:00 [CODE] Patched `scripts/download_models.sh` to prefer the modern `hf download` CLI and remove deprecated/incompatible flags.
- 2026-06-02T08:45:00-04:00 [TOOL] Downloaded all configured models to scratch and submitted real Qwen smoke job `10941299` on `ai-forge/a100-80gb/standby`; initial state is `PENDING` with reason `Priority`.
- 2026-06-02T10:28:40-04:00 [CODE] Fixed Qwen chat-template handling for Transformers 5 by normalizing `BatchEncoding` outputs into tensor dicts before `model.generate`.
- 2026-06-02T10:31:00-04:00 [CODE] Tightened verifier JSON prompt and added a conservative fallback for unparseable verifier prose that clearly says the solution is valid.

[DISCOVERIES]
- 2026-06-02T05:08:54-04:00 [TOOL] `/scratch/gilbreth/pham156/math-reasoning-agent/models` currently has no Qwen model files, so real Qwen smoke cannot complete until `bash scripts/download_models.sh` downloads weights to scratch.
- 2026-06-02T05:10:00-04:00 [TOOL] Job `10940001` will likely fail after starting unless Qwen weights are downloaded to `/scratch/gilbreth/pham156/math-reasoning-agent/models/Qwen2.5-Math-7B-Instruct` before execution.
- 2026-06-02T08:45:00-04:00 [TOOL] Supersedes earlier missing-model note: scratch now contains `Qwen2.5-Math-7B-Instruct` (15G), `Qwen2.5-Math-7B` (15G), `deepseek-math-7b-instruct` (13G), and `all-MiniLM-L6-v2` (873M).

[OUTCOMES]
- 2026-06-02T05:08:54-04:00 [TOOL] Mock LLM plus real MATH retrieval smoke passed and wrote `/scratch/gilbreth/pham156/math-reasoning-agent/outputs/smoke_mock_after_qwen_patch.json`; real model command exits status 2 with a clear scratch-path missing-model error.
- 2026-06-02T08:36:06-04:00 [TOOL] Corrected real Qwen smoke job `10940015` failed after 23 seconds with exit code `2:0`; logs show expected missing model files at `/scratch/gilbreth/pham156/math-reasoning-agent/models/Qwen2.5-Math-7B-Instruct`, and no `smoke_real_qwen.json` was created.
- 2026-06-02T10:32:01-04:00 [TOOL] Final real Qwen smoke job `10941319` completed successfully in 56 seconds and wrote `/scratch/gilbreth/pham156/math-reasoning-agent/outputs/smoke_real_qwen.json`; solver used real Qwen bf16 on CUDA, MATH retrieval returned 3 examples, verifier output is structured `valid` with score `8.0`.
