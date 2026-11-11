import {
  PriorityRetentionProcessor,
  TokenLimiterProcessor,
  ToolCallFilterProcessor,
  runMemoryPipeline,
  type AgentMessage,
} from "./memory.ts";
import { TraceRecorder } from "./tracing.ts";
import { scoreTrace, type EvalCase } from "./scorers.ts";
import { analyzeTraceFailures, compareShadowRollout, type AgentConfigProfile } from "./autoevals.ts";

const messages: AgentMessage[] = [
  { id: "sys", role: "system", content: "You are a support triage agent." },
  { id: "u1", role: "user", content: "Investigate account_id=acct_9012 billing hold under policy=eu_psd2." },
  { id: "t1", role: "tool", toolName: "fetchLogs", content: "L".repeat(2400) },
  { id: "u2", role: "user", content: "Summarize why account_id=acct_9012 is held and draft next step." },
];

const baselineMem = runMemoryPipeline(messages, [new TokenLimiterProcessor()], 160);
const tunedMem = runMemoryPipeline(
  messages,
  [
    new PriorityRetentionProcessor(),
    new ToolCallFilterProcessor({ keepLastNTools: 1, maxToolChars: 120 }),
    new TokenLimiterProcessor(),
  ],
  160,
);

const evalCase: EvalCase = {
  id: "case_billing_hold",
  category: "memory_retention",
  expectedKeys: ["status", "account_id", "action"],
  latencyBudgetMs: 900,
  groundTruthSubstrings: ["acct_9012", "eu_psd2"],
};

const recBase = new TraceRecorder("tr-base", "triageAgent", "v1", "case_billing_hold");
recBase.recordSpan({ kind: "tool_call", name: "fetchLogs", durationMs: 320, inputTokens: 40, outputTokens: 600, status: "ok", attributes: {} });
const baseTrace = recBase.finalize({
  finalOutput: "Held due to generic rule",
  expectedTools: ["fetchLogs"],
  requiredMemoryFacts: ["acct_9012", "eu_psd2"],
  recalledContextText: baselineMem.messages.map((m) => m.content).join(" "),
});

const recCand = new TraceRecorder("tr-cand", "triageAgent", "v1-tuned", "case_billing_hold");
recCand.recordSpan({ kind: "tool_call", name: "fetchLogs", durationMs: 280, inputTokens: 40, outputTokens: 90, status: "ok", attributes: {} });
const candTrace = recCand.finalize({
  finalOutput: JSON.stringify({ status: "held", account_id: "acct_9012", action: "verify under eu_psd2" }),
  expectedTools: ["fetchLogs"],
  requiredMemoryFacts: ["acct_9012", "eu_psd2"],
  recalledContextText: tunedMem.messages.map((m) => m.content).join(" "),
});

const baselineConfig: AgentConfigProfile = {
  version: "v1",
  memoryMaxTokens: 160,
  enablePriorityRetention: false,
  keepLastNTools: 3,
  toolValidationMode: "loose",
  enforceJsonOutput: false,
};

const baseReport = scoreTrace(baseTrace, evalCase);
const analysis = analyzeTraceFailures([baseReport], baselineConfig);
const rollout = compareShadowRollout([baseTrace], [candTrace], [evalCase]);

console.log(JSON.stringify({ baselineMemTokens: baselineMem.tokensAfter, tunedMemTokens: tunedMem.tokensAfter, analysis, rollout }, null, 2));
