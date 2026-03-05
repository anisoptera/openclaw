import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { stripReasoningTagsFromText } from "../../shared/text/reasoning-tags.js";

type AssistantContentBlock = Extract<AgentMessage, { role: "assistant" }>["content"][number];
type AssistantMessage = Extract<AgentMessage, { role: "assistant" }>;

export function isAssistantMessageWithContent(message: AgentMessage): message is AssistantMessage {
  return (
    !!message &&
    typeof message === "object" &&
    message.role === "assistant" &&
    Array.isArray(message.content)
  );
}

/**
 * Strip `<think>`/`<thinking>`/`<thought>` tags from text blocks in a content array.
 * Non-text and null/non-object blocks are passed through unchanged.
 * Text blocks that become empty after stripping are omitted (callers add a synthetic
 * fallback block when needed to preserve turn structure).
 * Returns `{ content, changed }` where `content` is a new array only when `changed` is true.
 */
function stripTagsFromContent(blocks: AssistantContentBlock[]): {
  content: AssistantContentBlock[];
  changed: boolean;
} {
  const next: AssistantContentBlock[] = [];
  let changed = false;
  for (const block of blocks) {
    if (!block || typeof block !== "object") {
      next.push(block);
      continue;
    }
    const typed = block as { type?: unknown; text?: unknown };
    if (typed.type === "text" && typeof typed.text === "string") {
      const stripped = stripReasoningTagsFromText(typed.text);
      if (stripped !== typed.text) {
        changed = true;
        if (stripped) {
          next.push({ ...block, text: stripped } as AssistantContentBlock);
        }
        // Empty after stripping — omit; caller adds synthetic block if needed.
        continue;
      }
    }
    next.push(block);
  }
  return { content: next, changed };
}

/**
 * Strip thinking blocks from assistant messages in completed chunks.
 *
 * With `chunkSize = N`, assistant turns are grouped into chunks of N. When a new
 * chunk boundary is crossed, all thinking blocks from every chunk before the
 * current one are removed. This limits KV cache invalidation to one bust per N
 * turns rather than every turn (useful for llama.cpp checkpoint-heavy providers).
 *
 * Example (chunkSize=5):
 *   - T=1..5:  active chunk [0,5)   → nothing stripped
 *   - T=6..10: active chunk [5,10)  → strip turns [0,5)
 *   - T=11..15: active chunk [10,15) → strip turns [0,10)
 *
 * Returns the original array reference when nothing was changed.
 */
export function dropStaleThinkingBlocks(
  messages: AgentMessage[],
  chunkSize: number,
): AgentMessage[] {
  if (chunkSize <= 0) {
    return messages;
  }

  // Count total assistant turns.
  let assistantTurnCount = 0;
  for (const msg of messages) {
    if (isAssistantMessageWithContent(msg)) {
      assistantTurnCount++;
    }
  }

  // Chunk-based cutoff: strip turns with index < activeChunkStart.
  // At T=N: completedChunks=1, activeChunkStart=0 → early return, nothing stripped yet.
  // At T=2N: completedChunks=2, activeChunkStart=N → strip first chunk [0,N).
  const completedChunks = Math.floor(assistantTurnCount / chunkSize);
  if (completedChunks === 0) {
    return messages;
  }
  const activeChunkStart = completedChunks * chunkSize - chunkSize;
  if (activeChunkStart <= 0) {
    return messages;
  }

  let touched = false;
  let assistantIdx = 0;
  const out: AgentMessage[] = [];

  for (const msg of messages) {
    if (!isAssistantMessageWithContent(msg)) {
      out.push(msg);
      continue;
    }

    const idx = assistantIdx++;
    if (idx >= activeChunkStart) {
      // Within the active (current) chunk — keep as-is.
      out.push(msg);
      continue;
    }

    // Stale turn: drop thinking blocks and strip residual <think> tags from text.
    // Some providers (e.g. llama.cpp) emit thinking via reasoning_content AND echo the
    // same content in <think> tags in the text block. promoteThinkingTagsToBlocks() skips
    // promotion when a structured block exists, leaving the raw tags. Strip both.
    const withoutThinking = msg.content.filter(
      (b) => !(b && typeof b === "object" && (b as { type?: unknown }).type === "thinking"),
    );
    const thinkingDropped = withoutThinking.length < msg.content.length;
    const { content: stripped, changed: tagsStripped } = stripTagsFromContent(withoutThinking);

    if (!thinkingDropped && !tagsStripped) {
      out.push(msg);
      continue;
    }

    touched = true;
    const content =
      stripped.length > 0 ? stripped : [{ type: "text", text: "" } as AssistantContentBlock];
    out.push({ ...msg, content });
  }

  return touched ? out : messages;
}

/**
 * Strip redundant `<think>` tags from text blocks in assistant messages that
 * already carry a structured `{type:"thinking"}` block.
 *
 * Some providers (e.g. llama.cpp with Qwen) emit thinking content in two forms
 * simultaneously: a structured block (from the `reasoning_content` field) AND the
 * same content wrapped in `<think>...</think>` tags inside the text block.
 * `promoteThinkingTagsToBlocks()` skips promotion when a structured block exists,
 * leaving the raw tags in the text. This function removes those tags from ALL
 * assistant turns — both stale and active — so the redundant tokens are never
 * sent to the provider.
 *
 * Returns the original array reference when nothing was changed.
 */
export function stripRedundantThinkingTags(messages: AgentMessage[]): AgentMessage[] {
  let touched = false;
  const out: AgentMessage[] = [];

  for (const msg of messages) {
    if (!isAssistantMessageWithContent(msg)) {
      out.push(msg);
      continue;
    }

    // Only process messages that have a structured thinking block; otherwise
    // the <think> tags ARE the thinking and must not be stripped.
    const hasThinkingBlock = msg.content.some(
      (b) => b && typeof b === "object" && (b as { type?: unknown }).type === "thinking",
    );
    if (!hasThinkingBlock) {
      out.push(msg);
      continue;
    }

    const { content, changed } = stripTagsFromContent(msg.content);
    if (!changed) {
      out.push(msg);
      continue;
    }

    touched = true;
    const nextContent =
      content.length > 0 ? content : [{ type: "text", text: "" } as AssistantContentBlock];
    out.push({ ...msg, content: nextContent });
  }

  return touched ? out : messages;
}

/**
 * Strip all `type: "thinking"` content blocks from assistant messages.
 *
 * If an assistant message becomes empty after stripping, it is replaced with
 * a synthetic `{ type: "text", text: "" }` block to preserve turn structure
 * (some providers require strict user/assistant alternation).
 *
 * Returns the original array reference when nothing was changed (callers can
 * use reference equality to skip downstream work).
 */
export function dropThinkingBlocks(messages: AgentMessage[]): AgentMessage[] {
  let touched = false;
  const out: AgentMessage[] = [];
  for (const msg of messages) {
    if (!isAssistantMessageWithContent(msg)) {
      out.push(msg);
      continue;
    }
    const nextContent: AssistantContentBlock[] = [];
    let changed = false;
    for (const block of msg.content) {
      if (block && typeof block === "object" && (block as { type?: unknown }).type === "thinking") {
        touched = true;
        changed = true;
        continue;
      }
      nextContent.push(block);
    }
    if (!changed) {
      out.push(msg);
      continue;
    }
    // Preserve the assistant turn even if all blocks were thinking-only.
    const content =
      nextContent.length > 0 ? nextContent : [{ type: "text", text: "" } as AssistantContentBlock];
    out.push({ ...msg, content });
  }
  return touched ? out : messages;
}
