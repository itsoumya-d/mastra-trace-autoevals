import assert from "node:assert/strict";
import test from "node:test";
import {
  PriorityRetentionProcessor,
  TokenLimiterProcessor,
  ToolCallFilterProcessor,
  analyzeTraceFailures,
  compareShadowRollout,
  runMemoryPipeline,
  scoreTrace,
  type AgentConfigProfile,
  type AgentMessage,
  type EvalCase,
  TraceRecorder,
} from "../src/index.ts";

test("MemoryProcessor pipeline preserves pinned facts while evicting noisy tool blobs", () => {
  const history: AgentMessage[] = [
    { id: "s1", role: "system", content: "Follow escalation SLAs strictly." },
    { id: "u1", role: "user", content: "Check account_id=acct_442 under deadline=24h." },
    { id: "t1", role: "tool", toolName: "rawLogDump", content: "X".repeat(3200) },
    { id: "t2", role: "tool", toolName: "getStatus", content: "status=review_required; reason=kyc_mismatch" },
    { id: "u2", role: "user", content: "Explain the next action concisely." },
  ];

  const naive = runMemoryPipeline(history, [new TokenLimiterProcessor()], 140);
  assert.equal(
    naive.messages.some((m) => m.id === "u1"),
    false,
    "Naive token limiter evicts earlier user constraints when oversized tool output fills budget",
  );

  const smart = runMemoryPipeline(
    history,
    [
      new PriorityRetentionProcessor(),
      new ToolCallFilterProcessor({ keepLastNTools: 1, maxToolChars: 80 }),
      new TokenLimiterProcessor(),
    ],
    140,
  );

  assert.ok(smart.tokensAfter <= 140);
  const combined = smart.messages.map((m) => m.content).join(" ");
  assert.ok(combined.includes("acct_442"));
  assert.ok(combined.includes("deadline=24h"));
});

test("TraceIntelligence proposes targeted config patches and validates shadow rollout promotion", () => {
  const evalCase: EvalCase = {
    id: "case_refund_guard",
    category: "tool_and_memory",
    expectedKeys: ["decision", "account_id", "policy"],
    latencyBudgetMs: 1000,
    groundTruthSubstrings: ["acct_442", "policy_refund"],
  };

  const baseRec = new TraceRecorder("t-base-1", "refundAgent", "v1.0", "case_refund_guard");
  baseRec.recordSpan({
    kind: "tool_call",
    name: "issueRefund",
    durationMs: 1200,
    inputTokens: 150,
    outputTokens: 600,
    status: "error",
    errorCode: "ZOD_VALIDATION_ERROR",
    attributes: {},
  });
  const baseTrace = baseRec.finalize({
    finalOutput: "Unable to refund account",
    expectedTools: ["issueRefund"],
    requiredMemoryFacts: ["acct_442", "policy_refund"],
    recalledContextText: "Generic conversation without facts",
  });

  const candRec = new TraceRecorder("t-cand-1", "refundAgent", "v1.0-tuned", "case_refund_guard");
  candRec.recordSpan({
    kind: "tool_call",
    name: "issueRefund",
    durationMs: 420,
    inputTokens: 90,
    outputTokens: 110,
    status: "ok",
    attributes: { validated: true },
  });
  const candTrace = candRec.finalize({
    finalOutput: JSON.stringify({
      decision: "approved",
      account_id: "acct_442",
      policy: "policy_refund",
    }),
    expectedTools: ["issueRefund"],
    requiredMemoryFacts: ["acct_442", "policy_refund"],
    recalledContextText: "Pinned account_id=acct_442 under policy_refund",
  });

  const baseConfig: AgentConfigProfile = {
    version: "v1.0",
    memoryMaxTokens: 2000,
    enablePriorityRetention: false,
    keepLastNTools: 4,
    toolValidationMode: "loose",
    enforceJsonOutput: false,
  };

  const baseReport = scoreTrace(baseTrace, evalCase);
  assert.equal(baseReport.passed, false);
  assert.equal(baseReport.failureMode, "memory_eviction");

  const { suggestions, proposedConfig } = analyzeTraceFailures([baseReport], baseConfig);
  assert.ok(suggestions.some((s) => s.failureMode === "memory_eviction"));
  assert.ok(suggestions.some((s) => s.failureMode === "wrong_tool_args"));
  assert.equal(proposedConfig.enablePriorityRetention, true);
  assert.equal(proposedConfig.toolValidationMode, "strict_zod");

  const comparison = compareShadowRollout([baseTrace], [candTrace], [evalCase]);
  assert.equal(comparison.promote, true);
  assert.equal(comparison.baselinePassRate, 0);
  assert.equal(comparison.candidatePassRate, 1);
  assert.ok(comparison.tokenDeltaAvg < 0);
});
