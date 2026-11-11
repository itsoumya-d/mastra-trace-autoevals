# Mastra Trace & AutoEvals Workbench

TypeScript benchmark harness exploring agent observability, deterministic `MemoryProcessor` context engineering, normalized multi-scorer evaluation, failure-cluster patch synthesis, and shadow rollout gating.

## Architecture

- **Memory Primitives (`src/memory.ts`)**: `PriorityRetentionProcessor`, `ToolCallFilterProcessor`, and `TokenLimiterProcessor` pipeline that preserves pinned constraints (IDs, policies, deadlines) under tight token budgets.
- **Execution Tracing (`src/tracing.ts`)**: OpenTelemetry-style `TraceRecorder` capturing agent loop spans (`tool_call`, `reasoning`, `memory_recall`, `eval_score`), step latency, and token utilization.
- **Scorers (`src/scorers.ts`)**: Composable `[0..1]` evaluators (`ToolTrajectoryScorer`, `MemoryRecallFidelityScorer`, `SchemaAndGroundingScorer`, `LatencySloScorer`) and deterministic failure classifier.
- **AutoEvals & Shadow Rollouts (`src/autoevals.ts`)**: Diagnoses recurring trace failure modes into concrete config diffs and validates candidate rollouts against baseline datasets before config promotion.

## Run Locally

```bash
npm test
npm run bench
```
