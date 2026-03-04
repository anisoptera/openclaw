import type { AgentMessage } from "@mariozechner/pi-agent-core";
import {
  CHARS_PER_TOKEN_ESTIMATE,
  TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE,
  type MessageCharEstimateCache,
  createMessageCharEstimateCache,
  estimateContextChars,
  estimateMessageCharsCached,
  getToolResultText,
  invalidateMessageCharsCacheEntry,
  isToolResultMessage,
} from "./tool-result-char-estimator.js";

// Keep a conservative input budget to absorb tokenizer variance and provider framing overhead.
const CONTEXT_INPUT_HEADROOM_RATIO = 0.8;
const SINGLE_TOOL_RESULT_CONTEXT_SHARE = 0.5;
// Minimum total eligible savings (as fraction of context budget) to justify a compaction pass.
// Skipping passes with marginal gains avoids cache perturbations for little benefit; when we do
// compact, we free meaningful headroom.
const MIN_COMPACTION_SAVINGS_RATIO = 0.2;

// Sanity bounds for token attribution: if the implied chars-per-token ratio for a tool
// result falls outside this range, the attribution is likely distorted (e.g. by a prior
// compaction removing content) and we fall back to the heuristic constant.
const MIN_CHARS_PER_TOKEN = TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE / 2; // 1
const MAX_CHARS_PER_TOKEN = CHARS_PER_TOKEN_ESTIMATE * 2; // 8

export type ToolResultTokenCache = WeakMap<AgentMessage, number>;

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

function truncateToolResultToChars(
  msg: AgentMessage,
  maxChars: number,
  cache: MessageCharEstimateCache,
): AgentMessage {
  if (!isToolResultMessage(msg)) {
    return msg;
  }

  const estimatedChars = estimateMessageCharsCached(msg, cache);
  if (estimatedChars <= maxChars) {
    return msg;
  }

  const rawText = getToolResultText(msg);
  if (!rawText) {
    return replaceToolResultText(msg, CONTEXT_LIMIT_TRUNCATION_NOTICE);
  }

  const truncatedText = truncateTextToBudget(rawText, maxChars);
  return replaceToolResultText(msg, truncatedText);
}

function compactExistingToolResultsInPlace(params: {
  messages: AgentMessage[];
  charsNeeded: number;
  contextBudgetChars: number;
  recentToolResultsToPreserve: number;
  cache: MessageCharEstimateCache;
}): number {
  const { messages, charsNeeded, contextBudgetChars, recentToolResultsToPreserve, cache } = params;
  if (charsNeeded <= 0) {
    return 0;
  }

  const placeholderChars = PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER.length;
  // Per-result minimum: savings must be at least 2× the placeholder size.
  // Results smaller than this are not worth the per-entry cache perturbation.
  const minPerResultSavings = placeholderChars * 2;

  // Identify the most recent N tool-result indices to protect from compaction.
  // These are results the model hasn't had a chance to act on yet (covers parallel
  // tool call batches from a single assistant turn).
  const toolResultIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    if (isToolResultMessage(messages[i])) {
      toolResultIndices.push(i);
    }
  }
  const protectedStart = Math.max(0, toolResultIndices.length - recentToolResultsToPreserve);
  const protectedIndices = new Set(toolResultIndices.slice(protectedStart));

  // Pre-scan: sum potential savings across eligible (non-protected) results.
  // The pass-level gate ensures we only bust the cache when total benefit justifies it.
  let totalEligibleSavings = 0;
  for (let i = 0; i < messages.length; i++) {
    if (protectedIndices.has(i)) {
      continue;
    }
    const msg = messages[i];
    if (!isToolResultMessage(msg)) {
      continue;
    }
    const before = estimateMessageCharsCached(msg, cache);
    if (before <= placeholderChars) {
      continue;
    }
    const potentialSavings = before - placeholderChars;
    if (potentialSavings >= minPerResultSavings) {
      totalEligibleSavings += potentialSavings;
    }
  }

  // Pass-level gate: skip if total savings don't justify the cache perturbation.
  const minPassSavings = Math.floor(contextBudgetChars * MIN_COMPACTION_SAVINGS_RATIO);
  if (totalEligibleSavings < minPassSavings) {
    return 0;
  }

  let reduced = 0;
  for (let i = 0; i < messages.length; i++) {
    if (protectedIndices.has(i)) {
      continue;
    }
    const msg = messages[i];
    if (!isToolResultMessage(msg)) {
      continue;
    }

    const before = estimateMessageCharsCached(msg, cache);
    if (before <= placeholderChars) {
      continue;
    }

    const potentialSavings = before - placeholderChars;
    if (potentialSavings < minPerResultSavings) {
      continue;
    }

    const compacted = replaceToolResultText(msg, PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    applyMessageMutationInPlace(msg, compacted, cache);
    const after = estimateMessageCharsCached(msg, cache);
    if (after >= before) {
      continue;
    }

    reduced += before - after;
    if (reduced >= charsNeeded) {
      break;
    }
  }

  return reduced;
}

function applyMessageMutationInPlace(
  target: AgentMessage,
  source: AgentMessage,
  cache?: MessageCharEstimateCache,
): void {
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
  if (cache) {
    invalidateMessageCharsCacheEntry(cache, target);
  }
}

// ---------------------------------------------------------------------------
// Token attribution from API usage data
// ---------------------------------------------------------------------------

type AssistantUsage = { input: number; output: number };

/** Extract usage.input/output from the last assistant message, if available. */
function getLastAssistantUsage(messages: AgentMessage[]): AssistantUsage | undefined {
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role !== "assistant") {
      continue;
    }
    const usage = (msg as { usage?: { input?: number; output?: number } }).usage;
    if (typeof usage?.input === "number") {
      return { input: usage.input, output: typeof usage.output === "number" ? usage.output : 0 };
    }
  }
  return undefined;
}

/**
 * Attribute real token costs to tool results by diffing consecutive assistant usage.
 *
 * Between assistant N and assistant N+1:
 *   delta = N+1.usage.input - N.usage.input
 *   toolResultTokens ≈ delta - N.usage.output  (the assistant response tokens)
 *
 * With a single tool result, attribution is exact. With multiple, we distribute
 * proportionally by char length. A sanity clamp discards wild ratios (e.g. after
 * compaction removed content between turns).
 */
function attributeToolResultTokens(
  messages: AgentMessage[],
  tokenCache: ToolResultTokenCache,
  charCache: MessageCharEstimateCache,
): void {
  // Collect assistant indices with usage data.
  const assistantIndices: number[] = [];
  for (let i = 0; i < messages.length; i++) {
    const msg = messages[i];
    if (msg.role !== "assistant") {
      continue;
    }
    const usage = (msg as { usage?: { input?: number } }).usage;
    if (typeof usage?.input === "number") {
      assistantIndices.push(i);
    }
  }
  if (assistantIndices.length < 2) {
    return;
  }

  // Walk consecutive pairs and attribute tokens to tool results between them.
  for (let p = 0; p < assistantIndices.length - 1; p++) {
    const olderIdx = assistantIndices[p];
    const newerIdx = assistantIndices[p + 1];
    const older = messages[olderIdx] as { usage: { input: number; output?: number } };
    const newer = messages[newerIdx] as { usage: { input: number } };

    const delta = newer.usage.input - older.usage.input;
    const assistantOutputTokens = typeof older.usage.output === "number" ? older.usage.output : 0;
    const toolTokenBudget = delta - assistantOutputTokens;

    // Collect tool results between the two assistants that aren't already cached.
    const uncached: { index: number; chars: number }[] = [];
    let totalUncachedChars = 0;
    for (let i = olderIdx + 1; i < newerIdx; i++) {
      if (!isToolResultMessage(messages[i])) {
        continue;
      }
      if (tokenCache.has(messages[i])) {
        continue;
      }
      const chars = estimateMessageCharsCached(messages[i], charCache);
      uncached.push({ index: i, chars });
      totalUncachedChars += chars;
    }
    if (uncached.length === 0 || totalUncachedChars === 0) {
      continue;
    }

    // Sanity check: is the implied overall ratio reasonable?
    const impliedCharsPerToken = totalUncachedChars / Math.max(1, toolTokenBudget);
    if (
      toolTokenBudget <= 0 ||
      impliedCharsPerToken < MIN_CHARS_PER_TOKEN ||
      impliedCharsPerToken > MAX_CHARS_PER_TOKEN
    ) {
      // Anomalous — fall back to heuristic for these results.
      for (const { index, chars } of uncached) {
        tokenCache.set(messages[index], Math.ceil(chars / TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE));
      }
      continue;
    }

    // Distribute tokens proportionally by char size.
    for (const { index, chars } of uncached) {
      const fraction = chars / totalUncachedChars;
      const tokens = Math.ceil(toolTokenBudget * fraction);
      tokenCache.set(messages[index], tokens);
    }
  }
}

function enforceToolResultContextBudgetInPlace(params: {
  messages: AgentMessage[];
  contextBudgetChars: number;
  contextWindowTokens: number;
  maxSingleToolResultChars: number;
  recentToolResultsToPreserve: number;
  tokenCache: ToolResultTokenCache;
}): void {
  const {
    messages,
    contextBudgetChars,
    contextWindowTokens,
    maxSingleToolResultChars,
    recentToolResultsToPreserve,
    tokenCache,
  } = params;
  const charCache = createMessageCharEstimateCache();

  // Ensure each tool result has an upper bound before considering total context usage.
  for (const message of messages) {
    if (!isToolResultMessage(message)) {
      continue;
    }
    const truncated = truncateToolResultToChars(message, maxSingleToolResultChars, charCache);
    applyMessageMutationInPlace(message, truncated, charCache);
  }

  // Attribute real token costs to tool results from API usage deltas.
  attributeToolResultTokens(messages, tokenCache, charCache);

  // Determine overshoot: prefer real token data from the last API call.
  let overshootChars: number;
  const lastUsage = getLastAssistantUsage(messages);

  if (lastUsage) {
    // Token-based: use real input token count from the last API call, plus an estimate
    // for new messages added since (tool results the model hasn't seen yet).
    const tokenBudget = Math.floor(contextWindowTokens * CONTEXT_INPUT_HEADROOM_RATIO);
    const lastAssistantIdx = findLastAssistantIndex(messages);

    let newContentTokenEstimate = 0;
    if (lastAssistantIdx >= 0) {
      for (let i = lastAssistantIdx + 1; i < messages.length; i++) {
        // Use real token count from cache if available, else fall back to char estimate.
        const cached = tokenCache.get(messages[i]);
        if (cached !== undefined) {
          newContentTokenEstimate += cached;
        } else {
          const chars = estimateMessageCharsCached(messages[i], charCache);
          newContentTokenEstimate += Math.ceil(chars / CHARS_PER_TOKEN_ESTIMATE);
        }
      }
    }

    const estimatedCurrentTokens = lastUsage.input + newContentTokenEstimate;
    if (estimatedCurrentTokens <= tokenBudget) {
      return; // Under budget — no compaction needed.
    }

    const overshootTokens = estimatedCurrentTokens - tokenBudget;
    overshootChars = overshootTokens * CHARS_PER_TOKEN_ESTIMATE;
  } else {
    // First turn (no assistant messages yet): fall back to char-based estimation.
    const currentChars = estimateContextChars(messages, charCache);
    if (currentChars <= contextBudgetChars) {
      return;
    }
    overshootChars = currentChars - contextBudgetChars;
  }

  const minPassSavings = Math.floor(contextBudgetChars * MIN_COMPACTION_SAVINGS_RATIO);
  // Hysteresis: free at least minPassSavings per pass to avoid thrashing on near-threshold contexts.
  const charsNeeded = Math.max(overshootChars, minPassSavings);

  compactExistingToolResultsInPlace({
    messages,
    charsNeeded,
    contextBudgetChars,
    recentToolResultsToPreserve,
    cache: charCache,
  });
}

/** Find the index of the last assistant message, or -1. */
function findLastAssistantIndex(messages: AgentMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      return i;
    }
  }
  return -1;
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
  const contextBudgetChars = Math.max(
    1_024,
    Math.floor(contextWindowTokens * CHARS_PER_TOKEN_ESTIMATE * CONTEXT_INPUT_HEADROOM_RATIO),
  );
  const maxSingleToolResultChars = Math.max(
    1_024,
    Math.floor(
      contextWindowTokens * TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE * SINGLE_TOOL_RESULT_CONTEXT_SHARE,
    ),
  );

  // Persistent token cache: survives across transformContext calls so attributed
  // token costs accumulate over the agent session's lifetime.
  const tokenCache: ToolResultTokenCache = new WeakMap();

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
      contextBudgetChars,
      contextWindowTokens,
      maxSingleToolResultChars,
      recentToolResultsToPreserve,
      tokenCache,
    });

    return contextMessages;
  }) as GuardableTransformContext;

  return () => {
    mutableAgent.transformContext = originalTransformContext;
  };
}
