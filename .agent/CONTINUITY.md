[PLANS]
- 2026-06-02T05:08:54-04:00 [USER] Build `math-reasoning-agent` with source/logs/env in `/home/pham156/math-reasoning-agent` and large models/data/cache/outputs in `/scratch/gilbreth/pham156/math-reasoning-agent`.

[DECISIONS]
- 2026-06-02T05:08:54-04:00 [CODE] Real pipeline mode must fail clearly if Qwen files are missing; only `--mock_llm` may use deterministic mock generation.
- 2026-06-02T05:08:54-04:00 [CODE] A requested `/home/pham156/math-reasoning-agent/models/...` model path is remapped to the scratch model directory when the home path is empty, preserving the large-file storage rule.

[PROGRESS]
- 2026-06-02T05:08:54-04:00 [CODE] Implemented real Qwen2.5-Math-7B-Instruct generation path using `tokenizer.apply_chat_template` when available, `device_map="auto"`, and CUDA bf16 when supported.
- 2026-06-02T05:08:54-04:00 [CODE] Added `scripts/run_real_qwen_smoke.slurm` for one-GPU Gilbreth smoke testing with logs in home and outputs in scratch.
- 2026-06-02T05:10:00-04:00 [TOOL] Submitted real Qwen smoke Slurm job `10940001` via `scripts/run_real_qwen_smoke.slurm`; initial state is `PENDING` on `a100-40gb` with reason `AssocGrpGRES`.

[DISCOVERIES]
- 2026-06-02T05:08:54-04:00 [TOOL] `/scratch/gilbreth/pham156/math-reasoning-agent/models` currently has no Qwen model files, so real Qwen smoke cannot complete until `bash scripts/download_models.sh` downloads weights to scratch.
- 2026-06-02T05:10:00-04:00 [TOOL] Job `10940001` will likely fail after starting unless Qwen weights are downloaded to `/scratch/gilbreth/pham156/math-reasoning-agent/models/Qwen2.5-Math-7B-Instruct` before execution.

[OUTCOMES]
- 2026-06-02T05:08:54-04:00 [TOOL] Mock LLM plus real MATH retrieval smoke passed and wrote `/scratch/gilbreth/pham156/math-reasoning-agent/outputs/smoke_mock_after_qwen_patch.json`; real model command exits status 2 with a clear scratch-path missing-model error.
