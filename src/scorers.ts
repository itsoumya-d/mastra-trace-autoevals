import type { AgentTrace } from "./tracing.ts";

export interface EvalCase {
  id: string;
  category: string;
  expectedKeys?: string[];
  latencyBudgetMs: number;
  groundTruthSubstrings: string[];
}

export interface ScoreResult {
  scorer: string;
  score: number; // 0..1 normalized
  passed: boolean;
  reason: string;
}

export interface TraceEvalReport {
  traceId: string;
  taskId: string;
  configVersion: string;
  overallScore: number;
  passed: boolean;
  scores: ScoreResult[];
  failureMode: "none" | "wrong_tool_args" | "memory_eviction" | "schema_drift" | "latency_breach";
}

export interface TraceScorer {
  readonly name: string;
  evaluate(trace: AgentTrace, evalCase: EvalCase): ScoreResult;
}

export class ToolTrajectoryScorer implements TraceScorer {
  readonly name = "ToolTrajectoryScorer";

  evaluate(trace: AgentTrace): ScoreResult {
    const failedTools = trace.spans.filter((s) => s.kind === "tool_call" && s.status !== "ok");
    if (trace.expectedTools.length === 0) {
      return {
        scorer: this.name,
        score: failedTools.length === 0 ? 1 : 0.4,
        passed: failedTools.length === 0,
        reason: failedTools.length === 0 ? "no tool failures" : `${failedTools.length} tool error spans`,
      };
    }
    const calledSet = new Set(trace.calledTools);
    const hits = trace.expectedTools.filter((t) => calledSet.has(t)).length;
    const raw = hits / trace.expectedTools.length;
    const penalty = failedTools.length > 0 ? Math.min(0.4, failedTools.length * 0.2) : 0;
    const score = Math.max(0, Number((raw - penalty).toFixed(3)));
    return {
      scorer: this.name,
      score,
      passed: score >= 0.8 && failedTools.length === 0,
      reason: `matched ${hits}/${trace.expectedTools.length} expected tools (${failedTools.length} errored)`,
    };
  }
}

export class MemoryRecallFidelityScorer implements TraceScorer {
  readonly name = "MemoryRecallFidelityScorer";

  evaluate(trace: AgentTrace): ScoreResult {
    if (trace.requiredMemoryFacts.length === 0) {
      return { scorer: this.name, score: 1, passed: true, reason: "no pinned facts required" };
    }
    const lower = trace.recalledContextText.toLowerCase();
    const hits = trace.requiredMemoryFacts.filter((f) => lower.includes(f.toLowerCase()));
    const score = Number((hits.length / trace.requiredMemoryFacts.length).toFixed(3));
    return {
      scorer: this.name,
      score,
      passed: score === 1,
      reason: `recalled ${hits.length}/${trace.requiredMemoryFacts.length} required memory facts`,
    };
  }
}

export class SchemaAndGroundingScorer implements TraceScorer {
  readonly name = "SchemaAndGroundingScorer";

  evaluate(trace: AgentTrace, evalCase: EvalCase): ScoreResult {
    const lower = trace.finalOutput.toLowerCase();
    let jsonKeysOk = true;
    if (evalCase.expectedKeys && evalCase.expectedKeys.length > 0) {
      try {
        const parsed = JSON.parse(trace.finalOutput) as Record<string, unknown>;
        jsonKeysOk = evalCase.expectedKeys.every((k) => Object.prototype.hasOwnProperty.call(parsed, k));
      } catch {
        jsonKeysOk = false;
      }
    }
    const gtHits = evalCase.groundTruthSubstrings.filter((s) => lower.includes(s.toLowerCase())).length;
    const gtRatio = evalCase.groundTruthSubstrings.length
      ? gtHits / evalCase.groundTruthSubstrings.length
      : 1;
    const score = Number(((jsonKeysOk ? 0.5 : 0) + gtRatio * 0.5).toFixed(3));
    return {
      scorer: this.name,
      score,
      passed: jsonKeysOk && gtRatio >= 0.8,
      reason: `jsonKeysOk=${jsonKeysOk}, groundTruth=${gtHits}/${evalCase.groundTruthSubstrings.length}`,
    };
  }
}

export class LatencySloScorer implements TraceScorer {
  readonly name = "LatencySloScorer";

  evaluate(trace: AgentTrace, evalCase: EvalCase): ScoreResult {
    if (trace.totalDurationMs <= evalCase.latencyBudgetMs) {
      return {
        scorer: this.name,
        score: 1,
        passed: true,
        reason: `${trace.totalDurationMs}ms <= ${evalCase.latencyBudgetMs}ms budget`,
      };
    }
    const ratio = evalCase.latencyBudgetMs / Math.max(1, trace.totalDurationMs);
    const score = Number(Math.max(0.1, ratio).toFixed(3));
    return {
      scorer: this.name,
      score,
      passed: false,
      reason: `${trace.totalDurationMs}ms exceeded ${evalCase.latencyBudgetMs}ms budget`,
    };
  }
}

export function classifyFailureMode(scores: ScoreResult[]): TraceEvalReport["failureMode"] {
  const byName = new Map(scores.map((s) => [s.scorer, s]));
  if (!byName.get("MemoryRecallFidelityScorer")?.passed) return "memory_eviction";
  if (!byName.get("ToolTrajectoryScorer")?.passed) return "wrong_tool_args";
  if (!byName.get("SchemaAndGroundingScorer")?.passed) return "schema_drift";
  if (!byName.get("LatencySloScorer")?.passed) return "latency_breach";
  return "none";
}

export function scoreTrace(
  trace: AgentTrace,
  evalCase: EvalCase,
  scorers: TraceScorer[] = [
    new ToolTrajectoryScorer(),
    new MemoryRecallFidelityScorer(),
    new SchemaAndGroundingScorer(),
    new LatencySloScorer(),
  ],
): TraceEvalReport {
  const scores = scorers.map((s) => s.evaluate(trace, evalCase));
  const overallScore = Number((scores.reduce((sum, s) => sum + s.score, 0) / scores.length).toFixed(3));
  const failureMode = classifyFailureMode(scores);
  return {
    traceId: trace.traceId,
    taskId: trace.taskId,
    configVersion: trace.configVersion,
    overallScore,
    passed: scores.every((s) => s.passed),
    scores,
    failureMode,
  };
}
