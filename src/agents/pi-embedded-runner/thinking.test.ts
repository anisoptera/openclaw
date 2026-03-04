import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import { castAgentMessage } from "../test-helpers/agent-message-fixtures.js";
import {
  dropStaleThinkingBlocks,
  dropThinkingBlocks,
  isAssistantMessageWithContent,
  stripRedundantThinkingTags,
} from "./thinking.js";

describe("isAssistantMessageWithContent", () => {
  it("accepts assistant messages with array content and rejects others", () => {
    const assistant = castAgentMessage({
      role: "assistant",
      content: [{ type: "text", text: "ok" }],
    });
    const user = castAgentMessage({ role: "user", content: "hi" });
    const malformed = castAgentMessage({ role: "assistant", content: "not-array" });

    expect(isAssistantMessageWithContent(assistant)).toBe(true);
    expect(isAssistantMessageWithContent(user)).toBe(false);
    expect(isAssistantMessageWithContent(malformed)).toBe(false);
  });
});

describe("dropThinkingBlocks", () => {
  it("returns the original reference when no thinking blocks are present", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({ role: "user", content: "hello" }),
      castAgentMessage({ role: "assistant", content: [{ type: "text", text: "world" }] }),
    ];

    const result = dropThinkingBlocks(messages);
    expect(result).toBe(messages);
  });

  it("drops thinking blocks while preserving non-thinking assistant content", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        content: [
          { type: "thinking", thinking: "internal" },
          { type: "text", text: "final" },
        ],
      }),
    ];

    const result = dropThinkingBlocks(messages);
    const assistant = result[0] as Extract<AgentMessage, { role: "assistant" }>;
    expect(result).not.toBe(messages);
    expect(assistant.content).toEqual([{ type: "text", text: "final" }]);
  });

  it("keeps assistant turn structure when all content blocks were thinking", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({
        role: "assistant",
        content: [{ type: "thinking", thinking: "internal-only" }],
      }),
    ];

    const result = dropThinkingBlocks(messages);
    const assistant = result[0] as Extract<AgentMessage, { role: "assistant" }>;
    expect(assistant.content).toEqual([{ type: "text", text: "" }]);
  });
});

describe("dropStaleThinkingBlocks", () => {
  function makeMessages(turns: Array<"thinking" | "text" | "both" | "none">): AgentMessage[] {
    const msgs: AgentMessage[] = [];
    for (const kind of turns) {
      msgs.push(castAgentMessage({ role: "user", content: "prompt" }));
      if (kind === "thinking") {
        msgs.push(
          castAgentMessage({
            role: "assistant",
            content: [{ type: "thinking", thinking: "t" }],
          }),
        );
      } else if (kind === "text") {
        msgs.push(
          castAgentMessage({ role: "assistant", content: [{ type: "text", text: "reply" }] }),
        );
      } else if (kind === "both") {
        msgs.push(
          castAgentMessage({
            role: "assistant",
            content: [
              { type: "thinking", thinking: "t" },
              { type: "text", text: "reply" },
            ],
          }),
        );
      } else {
        // "none" — assistant with no thinking blocks
        msgs.push(
          castAgentMessage({ role: "assistant", content: [{ type: "text", text: "no-think" }] }),
        );
      }
    }
    return msgs;
  }

  it("returns original reference when there are no thinking blocks", () => {
    const messages = makeMessages(["text", "text", "text"]);
    expect(dropStaleThinkingBlocks(messages, 2)).toBe(messages);
  });

  it("returns original reference when chunkSize is 0 or negative", () => {
    const messages = makeMessages(["thinking", "thinking", "thinking"]);
    expect(dropStaleThinkingBlocks(messages, 0)).toBe(messages);
    expect(dropStaleThinkingBlocks(messages, -1)).toBe(messages);
  });

  it("returns original reference when total turns < chunkSize (first chunk incomplete)", () => {
    // 4 turns, chunkSize=5 → completedChunks=0 → nothing stripped
    const messages = makeMessages(["thinking", "thinking", "thinking", "thinking"]);
    expect(dropStaleThinkingBlocks(messages, 5)).toBe(messages);
  });

  it("returns original reference exactly at chunkSize boundary (activeChunkStart=0)", () => {
    // 5 turns, chunkSize=5 → completedChunks=1, activeChunkStart=0 → nothing stripped
    const messages = makeMessages(["thinking", "thinking", "thinking", "thinking", "thinking"]);
    expect(dropStaleThinkingBlocks(messages, 5)).toBe(messages);
  });

  it("returns original reference mid-second-chunk before boundary (T=6, chunkSize=5)", () => {
    // T=6: completedChunks=1, activeChunkStart=0 → no stripping until second chunk completes at T=10
    const messages = makeMessages([
      "thinking",
      "thinking",
      "thinking",
      "thinking",
      "thinking",
      "thinking",
    ]);
    expect(dropStaleThinkingBlocks(messages, 5)).toBe(messages);
  });

  it("strips thinking from first chunk at second chunk boundary (T=10, chunkSize=5)", () => {
    // T=10: completedChunks=2, activeChunkStart=5 → strip [0,5), keep [5,10)
    const turns: Array<"thinking"> = Array(10).fill("thinking");
    const messages = makeMessages(turns);
    const result = dropStaleThinkingBlocks(messages, 5);
    expect(result).not.toBe(messages);
    const assistants = result.filter((m) => m.role === "assistant");
    expect(assistants).toHaveLength(10);
    for (let i = 0; i < 5; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        false,
      );
    }
    for (let i = 5; i < 10; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        true,
      );
    }
  });

  it("preserves non-thinking content when stripping stale turns", () => {
    // 10 turns (chunkSize=5): strip turns 0-4, keep 5-9.
    const messages = makeMessages([
      "both",
      "both",
      "both",
      "both",
      "both",
      "both",
      "both",
      "both",
      "both",
      "both",
    ]);
    const result = dropStaleThinkingBlocks(messages, 5);
    const assistants = result.filter((m) => m.role === "assistant");
    // Stale turns should keep the text block, drop thinking.
    for (let i = 0; i < 5; i++) {
      expect(assistants[i].content).toEqual([{ type: "text", text: "reply" }]);
    }
    // Active turns should keep both blocks.
    for (let i = 5; i < 10; i++) {
      expect(assistants[i].content).toEqual([
        { type: "thinking", thinking: "t" },
        { type: "text", text: "reply" },
      ]);
    }
  });

  it("adds synthetic text block when stale turn had only a thinking block", () => {
    // T=10, chunkSize=5: first 5 are thinking-only (stale), turns 5-9 are text.
    const messages = makeMessages([
      "thinking",
      "thinking",
      "thinking",
      "thinking",
      "thinking",
      "text",
      "text",
      "text",
      "text",
      "text",
    ]);
    const result = dropStaleThinkingBlocks(messages, 5);
    const assistants = result.filter((m) => m.role === "assistant");
    for (let i = 0; i < 5; i++) {
      expect(assistants[i].content).toEqual([{ type: "text", text: "" }]);
    }
  });

  it("strips only first chunk mid-second-chunk (T=11, chunkSize=5)", () => {
    // T=11, chunkSize=5: activeChunkStart=5, strip [0,5), keep [5,11)
    // No additional bust until T=15.
    const turns: Array<"thinking"> = Array(11).fill("thinking");
    const messages = makeMessages(turns);
    const result = dropStaleThinkingBlocks(messages, 5);
    const assistants = result.filter((m) => m.role === "assistant");
    for (let i = 0; i < 5; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        false,
      );
    }
    for (let i = 5; i < 11; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        true,
      );
    }
  });

  it("strips residual <think> tags from text blocks in stale turns", () => {
    // llama.cpp emits thinking via reasoning_content AND echoes tags in text.
    // promoteThinkingTagsToBlocks() skips when a structured block already exists,
    // leaving raw tags in the text block. Both must be removed on stale turns.
    const msgs: AgentMessage[] = [];
    for (let i = 0; i < 10; i++) {
      msgs.push(castAgentMessage({ role: "user", content: "prompt" }));
      msgs.push(
        castAgentMessage({
          role: "assistant",
          content: [
            { type: "thinking", thinking: "t" },
            { type: "text", text: "<think>\nt\n</think>\n\nreply" },
          ],
        }),
      );
    }
    const result = dropStaleThinkingBlocks(msgs, 5);
    const assistants = result.filter((m) => m.role === "assistant");
    // Stale turns: thinking block gone, tags stripped from text, only reply remains.
    for (let i = 0; i < 5; i++) {
      expect(assistants[i].content).toEqual([{ type: "text", text: "reply" }]);
    }
    // Active turns: untouched.
    for (let i = 5; i < 10; i++) {
      expect(assistants[i].content).toEqual([
        { type: "thinking", thinking: "t" },
        { type: "text", text: "<think>\nt\n</think>\n\nreply" },
      ]);
    }
  });

  it("strips two chunks at the third chunk boundary (T=15, chunkSize=5)", () => {
    // T=15, chunkSize=5: activeChunkStart=10, strip [0,10), keep [10,15)
    const turns: Array<"thinking"> = Array(15).fill("thinking");
    const messages = makeMessages(turns);
    const result = dropStaleThinkingBlocks(messages, 5);
    const assistants = result.filter((m) => m.role === "assistant");
    for (let i = 0; i < 10; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        false,
      );
    }
    for (let i = 10; i < 15; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        true,
      );
    }
  });
});

describe("stripRedundantThinkingTags", () => {
  it("returns original reference when no messages have a structured thinking block", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({ role: "user", content: "hi" }),
      castAgentMessage({
        role: "assistant",
        content: [{ type: "text", text: "<think>reasoning</think>\n\nreply" }],
      }),
    ];
    // No structured thinking block present — tags are the only thinking, must not strip.
    expect(stripRedundantThinkingTags(messages)).toBe(messages);
  });

  it("returns original reference when no text blocks have think tags", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({ role: "user", content: "hi" }),
      castAgentMessage({
        role: "assistant",
        content: [
          { type: "thinking", thinking: "t" },
          { type: "text", text: "reply" },
        ],
      }),
    ];
    expect(stripRedundantThinkingTags(messages)).toBe(messages);
  });

  it("strips tags from text blocks when a structured thinking block co-exists", () => {
    const messages: AgentMessage[] = [
      castAgentMessage({ role: "user", content: "hi" }),
      castAgentMessage({
        role: "assistant",
        content: [
          { type: "thinking", thinking: "t" },
          { type: "text", text: "<think>\nt\n</think>\n\nreply" },
        ],
      }),
    ];
    const result = stripRedundantThinkingTags(messages);
    expect(result).not.toBe(messages);
    const assistant = result.find((m) => m.role === "assistant");
    expect((assistant as Extract<AgentMessage, { role: "assistant" }>).content).toEqual([
      { type: "thinking", thinking: "t" },
      { type: "text", text: "reply" },
    ]);
  });

  it("processes all turns, not just stale ones", () => {
    // Three assistant turns all with redundant tags — all should be cleaned.
    const make = (i: number) => [
      castAgentMessage({ role: "user", content: `q${i}` }),
      castAgentMessage({
        role: "assistant",
        content: [
          { type: "thinking", thinking: `t${i}` },
          { type: "text", text: `<think>t${i}</think>\n\nreply${i}` },
        ],
      }),
    ];
    const messages: AgentMessage[] = [...make(1), ...make(2), ...make(3)];
    const result = stripRedundantThinkingTags(messages);
    const assistants = result.filter((m) => m.role === "assistant");
    for (let i = 0; i < 3; i++) {
      expect(assistants[i].content.some((b) => (b as { type?: string }).type === "thinking")).toBe(
        true,
      );
      const textBlock = assistants[i].content.find(
        (b) => (b as { type?: string }).type === "text",
      ) as { type: string; text: string } | undefined;
      expect(textBlock?.text).toBe(`reply${i + 1}`);
    }
  });
});
