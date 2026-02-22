import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
  CONTEXT_LIMIT_TRUNCATION_NOTICE,
  PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
  installToolResultContextGuard,
} from "./tool-result-context-guard.js";

function makeUser(text: string): AgentMessage {
  return {
    role: "user",
    content: text,
    timestamp: Date.now(),
  } as unknown as AgentMessage;
}

function makeToolResult(id: string, text: string): AgentMessage {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "read",
    content: [{ type: "text", text }],
    isError: false,
    timestamp: Date.now(),
  } as unknown as AgentMessage;
}

function makeLegacyToolResult(id: string, text: string): AgentMessage {
  return {
    role: "tool",
    tool_call_id: id,
    tool_name: "read",
    content: text,
  } as unknown as AgentMessage;
}

function makeToolResultWithDetails(id: string, text: string, detailText: string): AgentMessage {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: "read",
    content: [{ type: "text", text }],
    details: {
      truncation: {
        truncated: true,
        outputLines: 100,
        content: detailText,
      },
    },
    isError: false,
    timestamp: Date.now(),
  } as unknown as AgentMessage;
}

function getToolResultText(msg: AgentMessage): string {
  const content = (msg as { content?: unknown }).content;
  if (!Array.isArray(content)) {
    return "";
  }
  const block = content.find(
    (entry) => entry && typeof entry === "object" && (entry as { type?: string }).type === "text",
  ) as { text?: string } | undefined;
  return typeof block?.text === "string" ? block.text : "";
}

function makeGuardableAgent(
  transformContext?: (
    messages: AgentMessage[],
    signal: AbortSignal,
  ) => AgentMessage[] | Promise<AgentMessage[]>,
) {
  return { transformContext };
}

function makeTwoToolResultOverflowContext(): AgentMessage[] {
  return [
    makeUser("u".repeat(2_000)),
    makeToolResult("call_old", "x".repeat(1_000)),
    makeToolResult("call_new", "y".repeat(1_000)),
  ];
}

async function applyGuardToContext(
  agent: { transformContext?: (messages: AgentMessage[], signal: AbortSignal) => unknown },
  contextForNextCall: AgentMessage[],
) {
  installToolResultContextGuard({
    agent,
    contextWindowTokens: 1_000,
    recentToolResultsToPreserve: 0,
  });
  return await agent.transformContext?.(contextForNextCall, new AbortController().signal);
}

function expectCompactedToolResultsWithoutContextNotice(
  contextForNextCall: AgentMessage[],
  oldIndex: number,
  newIndex: number,
) {
  const oldResultText = getToolResultText(contextForNextCall[oldIndex]);
  const newResultText = getToolResultText(contextForNextCall[newIndex]);
  expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  expect(newResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  expect(newResultText).not.toContain(CONTEXT_LIMIT_TRUNCATION_NOTICE);
}

describe("installToolResultContextGuard", () => {
  it("compacts oldest-first when total context overflows, even if each result fits individually", async () => {
    const agent = makeGuardableAgent();
    const contextForNextCall = makeTwoToolResultOverflowContext();
    const transformed = await applyGuardToContext(agent, contextForNextCall);

    expect(transformed).toBe(contextForNextCall);
    expectCompactedToolResultsWithoutContextNotice(contextForNextCall, 1, 2);
  });

  it("keeps compacting oldest-first until context is back under budget", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_200)),
      makeToolResult("call_1", "a".repeat(800)),
      makeToolResult("call_2", "b".repeat(800)),
      makeToolResult("call_3", "c".repeat(800)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const first = getToolResultText(contextForNextCall[1]);
    const second = getToolResultText(contextForNextCall[2]);
    const third = getToolResultText(contextForNextCall[3]);

    expect(first).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(second).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(third).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("survives repeated large tool results by compacting older outputs before later turns", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 100_000,
    });

    const contextForNextCall: AgentMessage[] = [makeUser("stress")];
    for (let i = 1; i <= 4; i++) {
      contextForNextCall.push(makeToolResult(`call_${i}`, String(i).repeat(95_000)));
      await agent.transformContext?.(contextForNextCall, new AbortController().signal);
    }

    const toolResultTexts = contextForNextCall
      .filter((msg) => msg.role === "toolResult")
      .map((msg) => getToolResultText(msg as AgentMessage));

    expect(toolResultTexts[0]).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(toolResultTexts[3]?.length).toBe(95_000);
    expect(toolResultTexts.join("\n")).not.toContain(CONTEXT_LIMIT_TRUNCATION_NOTICE);
  });

  it("truncates an individually oversized tool result with a context-limit notice", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
    });

    const contextForNextCall = [makeToolResult("call_big", "z".repeat(5_000))];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const newResultText = getToolResultText(contextForNextCall[0]);
    expect(newResultText.length).toBeLessThan(5_000);
    expect(newResultText).toContain(CONTEXT_LIMIT_TRUNCATION_NOTICE);
  });

  it("keeps compacting oldest-first until overflow clears, including the newest tool result when needed", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_600)),
      makeToolResult("call_old", "x".repeat(700)),
      makeToolResult("call_new", "y".repeat(1_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);
    expectCompactedToolResultsWithoutContextNotice(contextForNextCall, 1, 2);
  });

  it("wraps an existing transformContext and guards the transformed output", async () => {
    const agent = makeGuardableAgent((messages) => {
      return messages.map(
        (msg) =>
          ({
            ...(msg as unknown as Record<string, unknown>),
          }) as unknown as AgentMessage,
      );
    });
    const contextForNextCall = makeTwoToolResultOverflowContext();
    const transformed = await applyGuardToContext(agent, contextForNextCall);

    expect(transformed).not.toBe(contextForNextCall);
    const transformedMessages = transformed as AgentMessage[];
    const oldResultText = getToolResultText(transformedMessages[1]);
    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("handles legacy role=tool string outputs when enforcing context budget", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_000)),
      makeLegacyToolResult("call_old", "x".repeat(1_000)),
      makeLegacyToolResult("call_new", "y".repeat(1_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResultText = (contextForNextCall[1] as { content?: unknown }).content;
    const newResultText = (contextForNextCall[2] as { content?: unknown }).content;

    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(newResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("skips compacting tool results below minimum savings threshold", async () => {
    // contextWindowTokens: 1_000 => contextBudgetChars = 1000 * 4 * 0.90 = 3600
    // placeholder = 47 chars; minPerResultSavings = 47 * 2 = 94
    // Small tool result: 50 chars text => weighted = max(50, ceil(50*2)) = 100
    // potentialSavings = 100 - 47 = 53 < 94 => skipped per-result
    // totalEligibleSavings = 0 < minPassSavings => pass skipped entirely
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
    });

    // Make context over budget with a large user message, but small tool results
    // that don't meet the per-result savings threshold (53 < 94).
    const smallText = "s".repeat(50);
    const contextForNextCall = [
      makeUser("u".repeat(3_700)), // push total over 3600 budget
      makeToolResult("call_small_1", smallText),
      makeToolResult("call_small_2", smallText),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const first = getToolResultText(contextForNextCall[1]);
    const second = getToolResultText(contextForNextCall[2]);

    expect(first).toBe(smallText);
    expect(second).toBe(smallText);
  });

  it("compacts tool results that meet minimum savings threshold", async () => {
    // contextWindowTokens: 1_000 => contextBudgetChars = 3600, minSavingsChars = 72
    // Tool result with 500 chars: weighted = max(500, ceil(500*2)) = 1000
    // potentialSavings = 1000 - 94 = 906 >= 72 => should be compacted
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const largeText = "L".repeat(500);
    const contextForNextCall = [
      makeUser("u".repeat(2_000)),
      makeToolResult("call_large", largeText),
      makeToolResult("call_large_2", largeText),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const first = getToolResultText(contextForNextCall[1]);

    expect(first).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("drops oversized read-tool details payloads when compacting tool results", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_200)),
      makeToolResultWithDetails("call_old", "x".repeat(900), "d".repeat(8_000)),
      makeToolResultWithDetails("call_new", "y".repeat(900), "d".repeat(8_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResult = contextForNextCall[1] as unknown as {
      details?: unknown;
    };
    const newResult = contextForNextCall[2] as unknown as {
      details?: unknown;
    };
    const oldResultText = getToolResultText(contextForNextCall[1]);
    const newResultText = getToolResultText(contextForNextCall[2]);

    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(newResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(oldResult.details).toBeUndefined();
    expect(newResult.details).toBeUndefined();
  });

  it("preserves the most recent N tool results from preemptive compaction", async () => {
    // With recentToolResultsToPreserve: 3 (default), the last 3 tool results are never
    // replaced with placeholders — even when the context is over budget.
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      // default recentToolResultsToPreserve: 3
    });

    // 4 tool results: oldest should be compacted, last 3 should be preserved.
    const contextForNextCall = [
      makeUser("u".repeat(500)),
      makeToolResult("call_1", "a".repeat(800)), // oldest — eligible to compact
      makeToolResult("call_2", "b".repeat(800)), // protected (within last 3)
      makeToolResult("call_3", "c".repeat(800)), // protected
      makeToolResult("call_4", "d".repeat(800)), // protected
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    expect(getToolResultText(contextForNextCall[1])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[2])).toBe("b".repeat(800));
    expect(getToolResultText(contextForNextCall[3])).toBe("c".repeat(800));
    expect(getToolResultText(contextForNextCall[4])).toBe("d".repeat(800));
  });

  it("respects a custom recentToolResultsToPreserve value", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 1,
    });

    // With preserve=1, only the very last result is safe; older ones can be compacted.
    const contextForNextCall = [
      makeUser("u".repeat(500)),
      makeToolResult("call_old", "x".repeat(800)),
      makeToolResult("call_mid", "y".repeat(800)),
      makeToolResult("call_new", "z".repeat(800)), // protected
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    expect(getToolResultText(contextForNextCall[1])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[3])).toBe("z".repeat(800));
  });
});
