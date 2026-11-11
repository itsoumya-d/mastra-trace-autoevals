export type SpanKind = "agent_run" | "reasoning" | "tool_call" | "memory_recall" | "eval_score";

export type SpanStatus = "ok" | "error" | "timeout";

export interface TraceSpan {
  spanId: string;
  parentSpanId?: string;
  traceId: string;
  kind: SpanKind;
  name: string;
  durationMs: number;
  inputTokens: number;
  outputTokens: number;
  status: SpanStatus;
  attributes: Record<string, string | number | boolean>;
  errorCode?: string;
}

export interface AgentTrace {
  traceId: string;
  agentName: string;
  configVersion: string;
  taskId: string;
  finalOutput: string;
  expectedTools: string[];
  calledTools: string[];
  requiredMemoryFacts: string[];
  recalledContextText: string;
  spans: TraceSpan[];
  totalDurationMs: number;
  totalTokens: number;
}

export class TraceRecorder {
  private readonly spans: TraceSpan[] = [];
  private seq = 0;
  readonly traceId: string;
  readonly agentName: string;
  readonly configVersion: string;
  readonly taskId: string;

  constructor(
    traceId: string,
    agentName: string,
    configVersion: string,
    taskId: string,
  ) {
    this.traceId = traceId;
    this.agentName = agentName;
    this.configVersion = configVersion;
    this.taskId = taskId;
  }

  recordSpan(input: Omit<TraceSpan, "spanId" | "traceId">): TraceSpan {
    const span: TraceSpan = {
      ...input,
      spanId: `${this.traceId}-sp-${++this.seq}`,
      traceId: this.traceId,
    };
    this.spans.push(span);
    return span;
  }

  finalize(params: {
    finalOutput: string;
    expectedTools: string[];
    recalledContextText: string;
    requiredMemoryFacts: string[];
  }): AgentTrace {
    const calledTools = this.spans
      .filter((s) => s.kind === "tool_call" && s.status === "ok")
      .map((s) => s.name);

    return {
      traceId: this.traceId,
      agentName: this.agentName,
      configVersion: this.configVersion,
      taskId: this.taskId,
      finalOutput: params.finalOutput,
      expectedTools: params.expectedTools,
      calledTools,
      requiredMemoryFacts: params.requiredMemoryFacts,
      recalledContextText: params.recalledContextText,
      spans: [...this.spans],
      totalDurationMs: this.spans.reduce((s, x) => s + x.durationMs, 0),
      totalTokens: this.spans.reduce((s, x) => s + x.inputTokens + x.outputTokens, 0),
    };
  }
}
