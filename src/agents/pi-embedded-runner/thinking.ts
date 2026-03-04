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

  // Count total assistant turns and build index → position map.
  let assistantTurnCount = 0;
  for (const msg of messages) {
    if (isAssistantMessageWithContent(msg)) {
      assistantTurnCount++;
    }
  }

  // Chunk-based cutoff: strip turns with index < activeChunkStart.
  // activeChunkStart = floor(T/N)*N - N (0 when T < N, so nothing stripped).
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

    // Strip thinking blocks and any residual <think> tags from this stale turn.
    // Some providers (e.g. llama.cpp) emit thinking via a reasoning_content field AND
    // echo the same content inside <think>...</think> tags in the text block. When
    // promoteThinkingTagsToBlocks() sees an existing structured thinking block it skips
    // tag promotion, leaving the raw tags in the text. Strip both here.
    const nextContent: AssistantContentBlock[] = [];
    let changed = false;
    for (const block of msg.content) {
      if (!block || typeof block !== "object") {
        nextContent.push(block);
        continue;
      }
      const typed = block as { type?: unknown; text?: unknown };
      if (typed.type === "thinking") {
        touched = true;
        changed = true;
        continue;
      }
      if (typed.type === "text" && typeof typed.text === "string") {
        const stripped = stripReasoningTagsFromText(typed.text);
        if (stripped !== typed.text) {
          touched = true;
          changed = true;
          if (stripped) {
            nextContent.push({ ...block, text: stripped } as AssistantContentBlock);
          }
          // Empty after stripping — omit; synthetic block added below if needed.
          continue;
        }
      }
      nextContent.push(block);
    }

    if (!changed) {
      out.push(msg);
      continue;
    }

    const content =
      nextContent.length > 0 ? nextContent : [{ type: "text", text: "" } as AssistantContentBlock];
    out.push({ ...msg, content });
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
