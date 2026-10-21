# Selected SWE tasks

The task directories contain the original user prompts and task metadata. `manifest.json` records the source captures, hashes, repository commits and original token counts. Reference answers remain in `replay/swebench_sources/` and are not passed to the agent.

See [manual section 12](../../agentic_workflow_manual.md#12-swe-bench--swe-bench-proclaude-code--vllm) for installation, measured execution, regression tests and result retrieval. Dataset test patches are in `../swebench_evaluation/` and are applied only to separate evaluation checkouts.
