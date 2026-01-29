# Telemetry Audit

Offline checker for Sudo telemetry (.jsonl) runs.

## What it does
- Collects one or many `.jsonl` files (or a directory)
- Reconstructs policy call windows using `LLM_CALLPOLICY_BEGIN/END_*`
- Extracts GRID_CONTEXT from `LLM_PROMPT_DUMP` (messages_pretty_json)
- Reads finalized toolplan from `LLM_TOOLPLAN_FINALIZED`
- Emits:
  - `audit.json` (machine-checkable)
  - `audit.md` (human-friendly)

## Run
```powershell
python tools\telemetry_audit\telemetry_audit.py --in runs\my_run_folder --out debug\telemetry_audit\my_run