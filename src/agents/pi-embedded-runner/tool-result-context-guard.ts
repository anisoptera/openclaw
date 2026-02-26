import type { AgentMessage } from "@mariozechner/pi-agent-core";

// Fallback chars-per-token ratios used when no empirical data is available (first API call
// or providers that report zero usage). Real content is typically 3-4 chars/token.
const CHARS_PER_TOKEN_ESTIMATE = 4;
// Keep a conservative input budget to absorb tokenizer variance and provider framing overhead.
const CONTEXT_INPUT_HEADROOM_RATIO = 0.9;
const SINGLE_TOOL_RESULT_CONTEXT_SHARE = 0.5;
// Minimum total savings (as fraction of context budget in tokens) required to justify a
// compaction pass. Set high (15%) because each compaction busts the prompt cache — we only
// want to pay that cost when the savings are substantial.
const MIN_COMPACTION_SAVINGS_RATIO = 0.15;
// Fallback for tool results; real tool outputs (code, JSON, text) are typically ~3 chars/token.
const TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE = 3;
const IMAGE_CHAR_ESTIMATE = 8_000;

export const CONTEXT_LIMIT_TRUNCATION_NOTICE = "[truncated: output exceeded context limit]";
const CONTEXT_LIMIT_TRUNCATION_SUFFIX = `\n${CONTEXT_LIMIT_TRUNCATION_NOTICE}`;

export const PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER =
  "[compacted: tool output removed to free context]";

type GuardableTransformContext = (
  messages: AgentMessage[],
  signal: AbortSignal,
) => AgentMessage[] | Promise<AgentMessage[]>;

type GuardableAgent = object;

type GuardableAgentRecord = {
  transformContext?: GuardableTransformContext;
};

function isTextBlock(block: unknown): block is { type: "text"; text: string } {
  return !!block && typeof block === "object" && (block as { type?: unknown }).type === "text";
}

function isImageBlock(block: unknown): boolean {
  return !!block && typeof block === "object" && (block as { type?: unknown }).type === "image";
}

function estimateUnknownChars(value: unknown): number {
  if (typeof value === "string") {
    return value.length;
  }
  if (value === undefined) {
    return 0;
  }
  try {
    const serialized = JSON.stringify(value);
    return typeof serialized === "string" ? serialized.length : 0;
  } catch {
    return 256;
  }
}

function isToolResultMessage(msg: AgentMessage): boolean {
  const role = (msg as { role?: unknown }).role;
  const type = (msg as { type?: unknown }).type;
  return role === "toolResult" || role === "tool" || type === "toolResult";
}

function getToolResultContent(msg: AgentMessage): unknown[] {
  if (!isToolResultMessage(msg)) {
    return [];
  }
  const content = (msg as { content?: unknown }).content;
  if (typeof content === "string") {
    return [{ type: "text", text: content }];
  }
  return Array.isArray(content) ? content : [];
}

function getToolResultText(msg: AgentMessage): string {
  const content = getToolResultContent(msg);
  const chunks: string[] = [];
  for (const block of content) {
    if (isTextBlock(block)) {
      chunks.push(block.text);
    }
  }
  return chunks.join("\n");
}

/** Count raw characters in a message (no weighting). */
function rawMessageChars(msg: AgentMessage): number {
  if (!msg || typeof msg !== "object") {
    return 0;
  }

  if (msg.role === "user") {
    const content = msg.content;
    if (typeof content === "string") {
      return content.length;
    }
    let chars = 0;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (isTextBlock(block)) {
          chars += block.text.length;
        } else if (isImageBlock(block)) {
          chars += IMAGE_CHAR_ESTIMATE;
        } else {
          chars += estimateUnknownChars(block);
        }
      }
    }
    return chars;
  }

  if (msg.role === "assistant") {
    let chars = 0;
    const content = (msg as { content?: unknown }).content;
    if (Array.isArray(content)) {
      for (const block of content) {
        if (!block || typeof block !== "object") {
          continue;
        }
        const typed = block as {
          type?: unknown;
          text?: unknown;
          thinking?: unknown;
          arguments?: unknown;
        };
        if (typed.type === "text" && typeof typed.text === "string") {
          chars += typed.text.length;
        } else if (typed.type === "thinking" && typeof typed.thinking === "string") {
          chars += typed.thinking.length;
        } else if (typed.type === "toolCall") {
          try {
            chars += JSON.stringify(typed.arguments ?? {}).length;
          } catch {
            chars += 128;
          }
        } else {
          chars += estimateUnknownChars(block);
        }
      }
    }
    return chars;
  }

  if (isToolResultMessage(msg)) {
    let chars = 0;
    const content = getToolResultContent(msg);
    for (const block of content) {
      if (isTextBlock(block)) {
        chars += block.text.length;
      } else if (isImageBlock(block)) {
        chars += IMAGE_CHAR_ESTIMATE;
      } else {
        chars += estimateUnknownChars(block);
      }
    }
    const details = (msg as { details?: unknown }).details;
    chars += estimateUnknownChars(details);
    return chars;
  }

  return 256;
}

// ── Empirical chars/token calibration ──────────────────────────────────────────
// Derive the actual chars-per-token ratio from the last assistant message's
// usage.input (exact prompt token count). Falls back to hardcoded constants
// when no usage data exists (first call, or providers that report zero usage).

type TokenCalibration = {
  /** Empirical or fallback chars-per-token ratio for non-tool-result messages. */
  charsPerToken: number;
  /**
   * Chars-per-token ratio for tool result messages. Equal to charsPerToken when empirical
   * data is available (the ratio already reflects the content mix); falls back to the
   * lower TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE constant on the first call or when providers
   * report no usage (code/JSON/file listings tokenize at ~3 chars/token, not 4).
   */
  toolResultCharsPerToken: number;
};

function extractAssistantInputTokens(msg: AgentMessage): number | undefined {
  if ((msg as { role?: unknown }).role !== "assistant") {
    return undefined;
  }
  const usage = (msg as { usage?: unknown }).usage;
  if (!usage || typeof usage !== "object") {
    return undefined;
  }
  const input = (usage as { input?: unknown }).input;
  if (typeof input !== "number" || !Number.isFinite(input) || input <= 0) {
    return undefined;
  }
  return input;
}

/**
 * Calibrate chars/token from real API usage data.
 * Scans messages for the last assistant message with usage.input, sums raw chars
 * for all messages up to that point, and derives the empirical ratio.
 */
function calibrateCharsPerToken(messages: AgentMessage[]): TokenCalibration {
  // Find the last assistant message with valid usage.input
  let lastAssistantIdx = -1;
  let lastAssistantInputTokens = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const tokens = extractAssistantInputTokens(messages[i]);
    if (tokens !== undefined) {
      lastAssistantIdx = i;
      lastAssistantInputTokens = tokens;
      break;
    }
  }

  if (lastAssistantIdx < 0) {
    return {
      charsPerToken: CHARS_PER_TOKEN_ESTIMATE,
      toolResultCharsPerToken: TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE,
    };
  }

  // Sum raw chars for all messages BEFORE the assistant message (i < lastAssistantIdx).
  // usage.input is the input token count — it does not include the assistant's own
  // output tokens. Including the assistant's text chars in the numerator would inflate
  // the ratio, making token estimates lower and biasing the guard toward not triggering.
  // usage.input also includes the system prompt, which we can't measure here — so the
  // ratio implicitly absorbs that overhead as a slight undercount of chars/token,
  // making context estimates slightly larger (conservative direction).
  let totalChars = 0;
  for (let i = 0; i < lastAssistantIdx; i++) {
    totalChars += rawMessageChars(messages[i]);
  }

  if (totalChars <= 0) {
    return {
      charsPerToken: CHARS_PER_TOKEN_ESTIMATE,
      toolResultCharsPerToken: TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE,
    };
  }

  const empirical = totalChars / lastAssistantInputTokens;
  // Clamp to a sane range to avoid absurd ratios from degenerate cases
  // (e.g., tiny conversations with large system prompts).
  const clamped = Math.max(1.5, Math.min(empirical, 8));
  // When empirical data is available, use one ratio for everything — it already
  // reflects the actual content mix (code, JSON, prose) in this conversation.
  return { charsPerToken: clamped, toolResultCharsPerToken: clamped };
}

/** Estimate total context tokens from raw char counts + calibrated ratios. */
function estimateContextTokens(messages: AgentMessage[], cal: TokenCalibration): number {
  let total = 0;
  for (const msg of messages) {
    total += estimateMessageTokens(msg, cal);
  }
  return total;
}

/** Estimate a single message's token cost using the appropriate ratio for its type. */
function estimateMessageTokens(msg: AgentMessage, cal: TokenCalibration): number {
  const ratio = isToolResultMessage(msg) ? cal.toolResultCharsPerToken : cal.charsPerToken;
  return Math.ceil(rawMessageChars(msg) / ratio);
}

function truncateTextToBudget(text: string, maxChars: number): string {
  if (text.length <= maxChars) {
    return text;
  }

  if (maxChars <= 0) {
    return CONTEXT_LIMIT_TRUNCATION_NOTICE;
  }

  const bodyBudget = Math.max(0, maxChars - CONTEXT_LIMIT_TRUNCATION_SUFFIX.length);
  if (bodyBudget <= 0) {
    return CONTEXT_LIMIT_TRUNCATION_NOTICE;
  }

  let cutPoint = bodyBudget;
  const newline = text.lastIndexOf("\n", bodyBudget);
  if (newline > bodyBudget * 0.7) {
    cutPoint = newline;
  }

  return text.slice(0, cutPoint) + CONTEXT_LIMIT_TRUNCATION_SUFFIX;
}

function replaceToolResultText(msg: AgentMessage, text: string): AgentMessage {
  const content = (msg as { content?: unknown }).content;
  const replacementContent =
    typeof content === "string" || content === undefined ? text : [{ type: "text", text }];

  const sourceRecord = msg as unknown as Record<string, unknown>;
  const { details: _details, ...rest } = sourceRecord;
  return {
    ...rest,
    content: replacementContent,
  } as AgentMessage;
}

function truncateToolResultToTokens(
  msg: AgentMessage,
  maxTokens: number,
  cal: TokenCalibration,
): AgentMessage {
  if (!isToolResultMessage(msg)) {
    return msg;
  }

  const estimatedTokens = estimateMessageTokens(msg, cal);
  if (estimatedTokens <= maxTokens) {
    return msg;
  }

  const rawText = getToolResultText(msg);
  if (!rawText) {
    return replaceToolResultText(msg, CONTEXT_LIMIT_TRUNCATION_NOTICE);
  }

  // Truncate raw text to fit the token budget (convert back to chars using tool-result ratio).
  const maxChars = Math.floor(maxTokens * cal.toolResultCharsPerToken);
  const truncatedText = truncateTextToBudget(rawText, maxChars);
  return replaceToolResultText(msg, truncatedText);
}

function compactExistingToolResultsInPlace(params: {
  messages: AgentMessage[];
  tokensNeeded: number;
  contextBudgetTokens: number;
  recentToolResultsToPreserve: number;
  cal: TokenCalibration;
}): number {
  const { messages, tokensNeeded, contextBudgetTokens, recentToolResultsToPreserve, cal } = params;
  if (tokensNeeded <= 0) {
    return 0;
  }

  // Placeholder is plain text, so use the general (non-tool-result) ratio.
  const placeholderTokens = Math.ceil(
    PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER.length / cal.charsPerToken,
  );
  // Per-result minimum: savings must be at least 2× the placeholder token cost.
  const minPerResultSavings = placeholderTokens * 2;

  // Identify the most recent N tool-result indices to protect from compaction.
  // These are results the model hasn't processed yet (or just processed this turn)
  // and must remain intact so the model can act on them.
  const toolResultIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (isToolResultMessage(messages[i])) {
      toolResultIndices.push(i);
    }
  }
  const protectedStart = Math.max(0, toolResultIndices.length - recentToolResultsToPreserve);
  const protectedIndices = new Set(toolResultIndices.slice(protectedStart));

  // Pre-scan: sum potential savings across eligible (non-protected) results.
  // The pass-level gate ensures we only bust the cache when the total benefit justifies it.
  let totalEligibleSavings = 0;
  for (let i = 0; i < messages.length; i++) {
    if (protectedIndices.has(i)) {
      continue;
    }
    const msg = messages[i];
    if (!isToolResultMessage(msg)) {
      continue;
    }
    const beforeTokens = estimateMessageTokens(msg, cal);
    if (beforeTokens <= placeholderTokens) {
      continue;
    }
    const potentialSavings = beforeTokens - placeholderTokens;
    if (potentialSavings >= minPerResultSavings) {
      totalEligibleSavings += potentialSavings;
    }
  }

  // Pass-level gate: skip if total savings don't justify the cache bust.
  const minPassSavings = Math.floor(contextBudgetTokens * MIN_COMPACTION_SAVINGS_RATIO);
  if (totalEligibleSavings < minPassSavings) {
    return 0;
  }

  // Compact oldest-first, skipping protected and individually-trivial results.
  let reduced = 0;
  for (let i = 0; i < messages.length; i++) {
    if (protectedIndices.has(i)) {
      continue;
    }

    const msg = messages[i];
    if (!isToolResultMessage(msg)) {
      continue;
    }

    const beforeTokens = estimateMessageTokens(msg, cal);
    if (beforeTokens <= placeholderTokens) {
      continue;
    }

    const potentialSavings = beforeTokens - placeholderTokens;
    if (potentialSavings < minPerResultSavings) {
      continue;
    }

    const compacted = replaceToolResultText(msg, PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    applyMessageMutationInPlace(msg, compacted);
    const afterTokens = estimateMessageTokens(msg, cal);
    if (afterTokens >= beforeTokens) {
      continue;
    }

    reduced += beforeTokens - afterTokens;
    if (reduced >= tokensNeeded) {
      break;
    }
  }

  return reduced;
}

function applyMessageMutationInPlace(target: AgentMessage, source: AgentMessage): void {
  if (target === source) {
    return;
  }

  const targetRecord = target as unknown as Record<string, unknown>;
  const sourceRecord = source as unknown as Record<string, unknown>;
  for (const key of Object.keys(targetRecord)) {
    if (!(key in sourceRecord)) {
      delete targetRecord[key];
    }
  }
  Object.assign(targetRecord, sourceRecord);
}

function enforceToolResultContextBudgetInPlace(params: {
  messages: AgentMessage[];
  contextBudgetTokens: number;
  maxSingleToolResultTokens: number;
  recentToolResultsToPreserve: number;
}): void {
  const { messages, contextBudgetTokens, maxSingleToolResultTokens, recentToolResultsToPreserve } =
    params;

  // Calibrate chars/token from real usage data when available.
  const cal = calibrateCharsPerToken(messages);

  // Ensure each tool result has an upper bound before considering total context usage.
  for (const message of messages) {
    if (!isToolResultMessage(message)) {
      continue;
    }
    const truncated = truncateToolResultToTokens(message, maxSingleToolResultTokens, cal);
    applyMessageMutationInPlace(message, truncated);
  }

  const currentTokens = estimateContextTokens(messages, cal);
  if (currentTokens <= contextBudgetTokens) {
    return;
  }

  // Hysteresis: if we're going to bust the prompt cache, free at least minPassSavings tokens
  // (15% of budget) — not just the bare overshoot. This prevents immediate re-triggering
  // (thrashing) by ensuring each compaction pass creates meaningful headroom.
  const overshoot = currentTokens - contextBudgetTokens;
  const minPassSavings = Math.floor(contextBudgetTokens * MIN_COMPACTION_SAVINGS_RATIO);
  const tokensNeeded = Math.max(overshoot, minPassSavings);

  // Compact oldest tool outputs first until the context is back under budget.
  compactExistingToolResultsInPlace({
    messages,
    tokensNeeded,
    contextBudgetTokens,
    recentToolResultsToPreserve,
    cal,
  });
}

export function installToolResultContextGuard(params: {
  agent: GuardableAgent;
  contextWindowTokens: number;
  /** Number of most-recent tool results to leave untouched during preemptive compaction.
   *  Protects results the model hasn't had a chance to act on yet.
   *  Defaults to 3 (covers parallel tool calls from a single assistant turn).
   */
  recentToolResultsToPreserve?: number;
}): () => void {
  const contextWindowTokens = Math.max(1, Math.floor(params.contextWindowTokens));
  const recentToolResultsToPreserve = params.recentToolResultsToPreserve ?? 3;
  // Budget and limits are now in tokens (not chars).
  const contextBudgetTokens = Math.max(
    256,
    Math.floor(contextWindowTokens * CONTEXT_INPUT_HEADROOM_RATIO),
  );
  const maxSingleToolResultTokens = Math.max(
    256,
    Math.floor(contextWindowTokens * SINGLE_TOOL_RESULT_CONTEXT_SHARE),
  );

  // Agent.transformContext is private in pi-coding-agent, so access it via a
  // narrow runtime view to keep callsites type-safe while preserving behavior.
  const mutableAgent = params.agent as GuardableAgentRecord;
  const originalTransformContext = mutableAgent.transformContext;

  mutableAgent.transformContext = (async (messages: AgentMessage[], signal: AbortSignal) => {
    const transformed = originalTransformContext
      ? await originalTransformContext.call(mutableAgent, messages, signal)
      : messages;

    const contextMessages = Array.isArray(transformed) ? transformed : messages;
    enforceToolResultContextBudgetInPlace({
      messages: contextMessages,
      contextBudgetTokens,
      maxSingleToolResultTokens,
      recentToolResultsToPreserve,
    });

    return contextMessages;
  }) as GuardableTransformContext;

  return () => {
    mutableAgent.transformContext = originalTransformContext;
  };
}

export const __testing = {
  calibrateCharsPerToken,
  estimateContextTokens,
  estimateMessageTokens,
  rawMessageChars,
} as const;
