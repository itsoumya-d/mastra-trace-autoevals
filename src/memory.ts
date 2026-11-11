export type Role = "system" | "user" | "assistant" | "tool";

export interface AgentMessage {
  id: string;
  role: Role;
  content: string;
  toolName?: string;
  pinned?: boolean;
  tokenEstimate?: number;
}

export interface ProcessorContext {
  maxTokens: number;
  droppedIds: string[];
  notes: string[];
}

export interface MemoryProcessor {
  readonly name: string;
  process(messages: AgentMessage[], ctx: ProcessorContext): AgentMessage[];
}

export function estimateTokens(msg: AgentMessage): number {
  if (typeof msg.tokenEstimate === "number") return msg.tokenEstimate;
  return Math.max(4, Math.ceil(msg.content.length / 4));
}

export class ToolCallFilterProcessor implements MemoryProcessor {
  readonly name = "ToolCallFilterProcessor";
  private readonly keepLastNTools: number;
  private readonly maxToolChars: number;

  constructor(options?: { keepLastNTools?: number; maxToolChars?: number }) {
    this.keepLastNTools = options?.keepLastNTools ?? 2;
    this.maxToolChars = options?.maxToolChars ?? 360;
  }

  process(messages: AgentMessage[], ctx: ProcessorContext): AgentMessage[] {
    const toolIndices: number[] = [];
    messages.forEach((m, idx) => {
      if (m.role === "tool") toolIndices.push(idx);
    });
    const cutoff = Math.max(0, toolIndices.length - this.keepLastNTools);
    const evictSet = new Set(toolIndices.slice(0, cutoff));

    const out: AgentMessage[] = [];
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (evictSet.has(i) && !m.pinned) {
        ctx.droppedIds.push(m.id);
        ctx.notes.push(`evicted verbose tool output ${m.id}`);
        continue;
      }
      if (m.role === "tool" && m.content.length > this.maxToolChars) {
        out.push({
          ...m,
          content: `${m.content.slice(0, this.maxToolChars)}...[truncated ${m.content.length - this.maxToolChars} chars]`,
          tokenEstimate: undefined,
        });
        ctx.notes.push(`truncated recent tool output ${m.id}`);
        continue;
      }
      out.push(m);
    }
    return out;
  }
}

export class PriorityRetentionProcessor implements MemoryProcessor {
  readonly name = "PriorityRetentionProcessor";
  private readonly preserveFactsRegex: RegExp;

  constructor(preserveFactsRegex = /\b(account_id|policy|constraint|deadline|api_key_ref|order_id)\b/i) {
    this.preserveFactsRegex = preserveFactsRegex;
  }

  process(messages: AgentMessage[], _ctx: ProcessorContext): AgentMessage[] {
    return messages.map((m) =>
      m.role === "system" || this.preserveFactsRegex.test(m.content) ? { ...m, pinned: true } : m,
    );
  }
}

export class TokenLimiterProcessor implements MemoryProcessor {
  readonly name = "TokenLimiterProcessor";

  process(messages: AgentMessage[], ctx: ProcessorContext): AgentMessage[] {
    let total = messages.reduce((sum, m) => sum + estimateTokens(m), 0);
    if (total <= ctx.maxTokens) return messages;

    const kept = [...messages];
    // Drop oldest non-pinned non-system messages while keeping the most recent user turn intact.
    let i = 0;
    while (total > ctx.maxTokens && i < kept.length - 1) {
      const candidate = kept[i];
      if (candidate.role === "system" || candidate.pinned) {
        i++;
        continue;
      }
      total -= estimateTokens(candidate);
      ctx.droppedIds.push(candidate.id);
      ctx.notes.push(`token limiter dropped ${candidate.id}`);
      kept.splice(i, 1);
    }
    return kept;
  }
}

export interface MemoryPipelineResult {
  messages: AgentMessage[];
  tokensBefore: number;
  tokensAfter: number;
  droppedIds: string[];
  notes: string[];
}

export function runMemoryPipeline(
  messages: AgentMessage[],
  processors: MemoryProcessor[],
  maxTokens: number,
): MemoryPipelineResult {
  const tokensBefore = messages.reduce((s, m) => s + estimateTokens(m), 0);
  const ctx: ProcessorContext = { maxTokens, droppedIds: [], notes: [] };
  let current = [...messages];
  for (const proc of processors) {
    current = proc.process(current, ctx);
  }
  const tokensAfter = current.reduce((s, m) => s + estimateTokens(m), 0);
  return {
    messages: current,
    tokensBefore,
    tokensAfter,
    droppedIds: ctx.droppedIds,
    notes: ctx.notes,
  };
}
