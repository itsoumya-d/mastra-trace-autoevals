import type { EvalCase, TraceEvalReport } from "./scorers.ts";
import { scoreTrace } from "./scorers.ts";
import type { AgentTrace } from "./tracing.ts";

export interface AgentConfigProfile {
  version: string;
  memoryMaxTokens: number;
  enablePriorityRetention: boolean;
  keepLastNTools: number;
  toolValidationMode: "loose" | "strict_zod";
  enforceJsonOutput: boolean;
}

export interface ConfigSuggestion {
  failureMode: TraceEvalReport["failureMode"];
  affectedCount: number;
  rationale: string;
  patch: Partial<AgentConfigProfile>;
}

export interface ShadowRolloutComparison {
  baselineVersion: string;
  candidateVersion: string;
  baselinePassRate: number;
  candidatePassRate: number;
  baselineAvgScore: number;
  candidateAvgScore: number;
  tokenDeltaAvg: number;
  promote: boolean;
  reason: string;
}

export function analyzeTraceFailures(
  reports: TraceEvalReport[],
  currentConfig: AgentConfigProfile,
): { suggestions: ConfigSuggestion[]; proposedConfig: AgentConfigProfile } {
  const counts = new Map<TraceEvalReport["failureMode"], number>();
  const scorerMap: Record<string, TraceEvalReport["failureMode"]> = {
    MemoryRecallFidelityScorer: "memory_eviction",
    ToolTrajectoryScorer: "wrong_tool_args",
    SchemaAndGroundingScorer: "schema_drift",
    LatencySloScorer: "latency_breach",
  };
  for (const r of reports) {
    let countedAny = false;
    for (const s of r.scores) {
      if (!s.passed && scorerMap[s.scorer]) {
        const mode = scorerMap[s.scorer];
        counts.set(mode, (counts.get(mode) ?? 0) + 1);
        countedAny = true;
      }
    }
    if (!countedAny && r.failureMode !== "none") {
      counts.set(r.failureMode, (counts.get(r.failureMode) ?? 0) + 1);
    }
  }

  const suggestions: ConfigSuggestion[] = [];
  const proposed: AgentConfigProfile = {
    ...currentConfig,
    version: `${currentConfig.version}-tuned`,
  };

  const memFailures = counts.get("memory_eviction") ?? 0;
  if (memFailures > 0) {
    suggestions.push({
      failureMode: "memory_eviction",
      affectedCount: memFailures,
      rationale: "Inject PriorityRetentionProcessor + ToolCallFilter truncation before TokenLimiter so critical IDs survive context pruning.",
      patch: { enablePriorityRetention: true, keepLastNTools: 1 },
    });
    proposed.enablePriorityRetention = true;
    proposed.keepLastNTools = Math.min(currentConfig.keepLastNTools, 1);
  }

  const toolFailures = counts.get("wrong_tool_args") ?? 0;
  if (toolFailures > 0) {
    suggestions.push({
      failureMode: "wrong_tool_args",
      affectedCount: toolFailures,
      rationale: "Enable strict Zod schema coercion + typed parameter hints in tool definitions.",
      patch: { toolValidationMode: "strict_zod" },
    });
    proposed.toolValidationMode = "strict_zod";
  }

  const schemaFailures = counts.get("schema_drift") ?? 0;
  if (schemaFailures > 0) {
    suggestions.push({
      failureMode: "schema_drift",
      affectedCount: schemaFailures,
      rationale: "Enable structured JSON output mode with schema retry fallback.",
      patch: { enforceJsonOutput: true },
    });
    proposed.enforceJsonOutput = true;
  }

  const latencyFailures = counts.get("latency_breach") ?? 0;
  if (latencyFailures > 0) {
    const trimmed = Math.max(1200, Math.round(currentConfig.memoryMaxTokens * 0.75));
    suggestions.push({
      failureMode: "latency_breach",
      affectedCount: latencyFailures,
      rationale: "Tighten context window budget and cap historical tool payloads to reduce prefill latency.",
      patch: { memoryMaxTokens: trimmed },
    });
    proposed.memoryMaxTokens = trimmed;
  }

  return { suggestions, proposedConfig: proposed };
}

export function compareShadowRollout(
  baselineTraces: AgentTrace[],
  candidateTraces: AgentTrace[],
  evalCases: EvalCase[],
): ShadowRolloutComparison {
  const caseById = new Map(evalCases.map((c) => [c.id, c]));
  const evalPairs = (traces: AgentTrace[]) =>
    traces.map((t) => {
      const c = caseById.get(t.taskId);
      if (!c) throw new Error(`Missing eval case for ${t.taskId}`);
      return { trace: t, report: scoreTrace(t, c) };
    });

  const base = evalPairs(baselineTraces);
  const cand = evalPairs(candidateTraces);
  const avg = (xs: number[]) => (xs.length ? Number((xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(3)) : 0);

  const baselinePassRate = avg(base.map((x) => (x.report.passed ? 1 : 0)));
  const candidatePassRate = avg(cand.map((x) => (x.report.passed ? 1 : 0)));
  const baselineAvgScore = avg(base.map((x) => x.report.overallScore));
  const candidateAvgScore = avg(cand.map((x) => x.report.overallScore));
  const baseTok = avg(base.map((x) => x.trace.totalTokens));
  const candTok = avg(cand.map((x) => x.trace.totalTokens));

  const promote = candidatePassRate > baselinePassRate && candidateAvgScore >= baselineAvgScore;
  const reason = promote
    ? `Candidate lifted pass rate from ${baselinePassRate} to ${candidatePassRate} and avg score from ${baselineAvgScore} to ${candidateAvgScore}.`
    : `Blocked promotion: candidate pass rate (${candidatePassRate}) did not beat baseline (${baselinePassRate}).`;

  return {
    baselineVersion: baselineTraces[0]?.configVersion ?? "baseline",
    candidateVersion: candidateTraces[0]?.configVersion ?? "candidate",
    baselinePassRate,
    candidatePassRate,
    baselineAvgScore,
    candidateAvgScore,
    tokenDeltaAvg: Number((candTok - baseTok).toFixed(1)),
    promote,
    reason,
  };
}
