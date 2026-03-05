import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import { castAgentMessage } from "../test-helpers/agent-message-fixtures.js";
import {
  CONTEXT_LIMIT_TRUNCATION_NOTICE,
  PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
  installToolResultContextGuard,
} from "./tool-result-context-guard.js";

function makeUser(text: string): AgentMessage {
  return castAgentMessage({
    role: "user",
    content: text,
    timestamp: Date.now(),
  });
}

function makeToolResult(id: string, text: string): AgentMessage {
  return castAgentMessage({
    role: "toolResult",
    toolCallId: id,
    toolName: "read",
    content: [{ type: "text", text }],
    isError: false,
    timestamp: Date.now(),
  });
}

function makeLegacyToolResult(id: string, text: string): AgentMessage {
  return castAgentMessage({
    role: "tool",
    tool_call_id: id,
    tool_name: "read",
    content: text,
  });
}

function makeToolResultWithDetails(id: string, text: string, detailText: string): AgentMessage {
  return castAgentMessage({
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
  });
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

function makeAssistantWithUsage(input: number, output: number = 0): AgentMessage {
  return castAgentMessage({
    role: "assistant",
    content: [{ type: "text", text: "ok" }],
    usage: { input, output },
    timestamp: Date.now(),
  });
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
  // budget=3200 (1000 tokens × 4 c/t × 0.8). user: 2200; tools: 1000 each → total 4200 > 3200.
  // overshoot 1000 > 952 (savings per result) → both compacted.
  return [
    makeUser("u".repeat(2_200)),
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

    // user: 2800 chars; 3 tool results: 800 chars each → total 5200 > 3200 budget.
    // overshoot 2000; each saves 752; need all 3 to clear 2000 (752×2=1504 < 2000 < 752×3=2256).
    const contextForNextCall = [
      makeUser("u".repeat(2_800)),
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

    // user: 2600; call_old: 700 (saves 652); call_new: 1000 (saves 952). Total 4300 > 3200.
    // overshoot 1100; call_old saves 652 < 1100, then call_new pushes total to 1604 ≥ 1100.
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
      return messages.map((msg) =>
        castAgentMessage({
          ...(msg as unknown as Record<string, unknown>),
        }),
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

    // user: 3000; tools: 1000 each → total 5000 > 3200. overshoot 1800 > 952 → both compacted.
    const contextForNextCall = [
      makeUser("u".repeat(3_000)),
      makeLegacyToolResult("call_old", "x".repeat(1_000)),
      makeLegacyToolResult("call_new", "y".repeat(1_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResultText = (contextForNextCall[1] as { content?: unknown }).content;
    const newResultText = (contextForNextCall[2] as { content?: unknown }).content;

    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(newResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("drops oversized read-tool details payloads when compacting tool results", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    // Char estimator counts content text only (not details field). user: 2800; tools: 900 text each.
    // Total 4600 > 3200. overshoot 1400 > 852 → both compacted; replaceToolResultText strips details.
    const contextForNextCall = [
      makeUser("u".repeat(2_800)),
      makeToolResultWithDetails("call_old", "x".repeat(900), "d".repeat(8_000)),
      makeToolResultWithDetails("call_new", "y".repeat(900), "d".repeat(8_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResult = contextForNextCall[1] as {
      details?: unknown;
    };
    const newResult = contextForNextCall[2] as {
      details?: unknown;
    };
    const oldResultText = getToolResultText(contextForNextCall[1]);
    const newResultText = getToolResultText(contextForNextCall[2]);

    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(newResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(oldResult.details).toBeUndefined();
    expect(newResult.details).toBeUndefined();
  });

  it("skips compacting tool results below the per-result savings threshold", async () => {
    // placeholder: 48 chars; minPerResultSavings = 48 * 2 = 96 chars.
    // 50-char tool result: savings = 50 - 48 = 2 < 96 → skipped per-result.
    // totalEligibleSavings = 0 < minPassSavings (640) → entire pass skipped.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({ agent, contextWindowTokens: 1_000 });

    const contextForNextCall = [
      makeUser("u".repeat(3_700)), // total 3800 > 3200: overflow, but tool results too small to compact
      makeToolResult("call_1", "s".repeat(50)),
      makeToolResult("call_2", "s".repeat(50)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);
    expect(getToolResultText(contextForNextCall[1])).toBe("s".repeat(50));
    expect(getToolResultText(contextForNextCall[2])).toBe("s".repeat(50));
  });

  it("preserves the most recent N tool results from preemptive compaction (default 3)", async () => {
    // 4 tool results; last 3 protected by default. Only the oldest is eligible.
    // user: 500; tools: 800 each → total 3700 > 3200. overshoot 500 < hysteresis floor 640.
    // charsNeeded = 640 (hysteresis); oldest saves 752 ≥ 640 → compacted. last 3 untouched.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({ agent, contextWindowTokens: 1_000 });

    const contextForNextCall = [
      makeUser("u".repeat(500)),
      makeToolResult("call_1", "a".repeat(800)),
      makeToolResult("call_2", "b".repeat(800)),
      makeToolResult("call_3", "c".repeat(800)),
      makeToolResult("call_4", "d".repeat(800)),
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
    // preserve=1: only call_new protected; call_old and call_mid are eligible.
    // user: 2600; tools: 800 each → total 5000 > 3200. overshoot 1800.
    // call_old saves 752, call_mid saves 752; eligible exhausted at 1504 → both compacted.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 1,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_600)),
      makeToolResult("call_old", "x".repeat(800)),
      makeToolResult("call_mid", "y".repeat(800)),
      makeToolResult("call_new", "z".repeat(800)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    expect(getToolResultText(contextForNextCall[1])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[2])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[3])).toBe("z".repeat(800));
  });

  it("skips compaction when real token usage (usage.input) is under the token budget", async () => {
    // Char-based: user(200) + tool_1(1600 weighted) + tool_2(1600 weighted) = 3400 > 3200 → would compact.
    // Token-based: usage.input=500 + 0 new tokens = 500 ≤ 800 (tokenBudget) → skip.
    // This shows that accurate token data avoids false-positive compaction.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({ agent, contextWindowTokens: 1_000 });

    const messages = [
      makeUser("u".repeat(200)),
      makeToolResult("call_1", "x".repeat(800)),
      makeToolResult("call_2", "y".repeat(800)),
      makeAssistantWithUsage(500, 100), // 500 real input tokens, well under 800 budget
    ];

    await agent.transformContext?.(messages, new AbortController().signal);

    // Token-based check reports no overflow → tool results left untouched.
    expect(getToolResultText(messages[1])).toBe("x".repeat(800));
    expect(getToolResultText(messages[2])).toBe("y".repeat(800));
  });

  it("compacts oldest-first when real token usage exceeds the token budget", async () => {
    // usage.input=900 > tokenBudget=800 → overshoot 100 tokens → 400 chars overshoot.
    // hysteresis floor = 640 chars → charsNeeded=640.
    // call_old (800 chars, 1600 weighted): savings=1552 ≥ 640 → compacted.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const messages = [
      makeUser("u".repeat(100)),
      makeToolResult("call_old", "x".repeat(800)),
      makeAssistantWithUsage(900, 50), // 900 real tokens > 800 budget
    ];

    await agent.transformContext?.(messages, new AbortController().signal);

    expect(getToolResultText(messages[1])).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("handles anomalous usage deltas (e.g. negative) without crashing, falling back to heuristic", async () => {
    // Negative delta (common after prior compaction): toolTokenBudget < 0 → anomalous.
    // Attribution falls back to heuristic constant. Guard must not crash.
    // With lastUsage.input=90 ≤ tokenBudget=800, no compaction should happen.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const messages = [
      makeAssistantWithUsage(100, 50), // older: input=100, output=50
      makeToolResult("call_mid", "x".repeat(800)),
      makeAssistantWithUsage(90, 20), // newer: input=90 < 100 → delta=-10, toolTokenBudget=-60 (anomalous)
    ];

    // Must not throw; tool result between two assistants is already counted in lastUsage.input.
    await agent.transformContext?.(messages, new AbortController().signal);

    // Token-based budget check: estimatedCurrentTokens = 90 ≤ 800 → no compaction.
    expect(getToolResultText(messages[1])).toBe("x".repeat(800));
  });

  it("falls back to char-based budget estimation when no prior assistant usage data exists", async () => {
    // No assistant messages → char-based fallback. user(3000) + tool(1600 weighted) = 4600 > 3200 → compact.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const messages = [makeUser("u".repeat(3_000)), makeToolResult("call_1", "a".repeat(800))];

    await agent.transformContext?.(messages, new AbortController().signal);

    expect(getToolResultText(messages[1])).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("custom contextInputHeadroomRatio changes compaction trigger point", async () => {
    // With default headroom (0.8):
    //   budget = 1000 * 4 * 0.8 = 3200 chars. total = 2200+1000 = 3200 → no overshoot → no compaction.
    // With tighter headroom (0.5):
    //   budget = 1000 * 4 * 0.5 = 2000 chars. overshoot = 1200 → compaction runs.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
      contextInputHeadroomRatio: 0.5,
    });

    const messages = [makeUser("u".repeat(2_200)), makeToolResult("call_1", "x".repeat(1_000))];

    await agent.transformContext?.(messages, new AbortController().signal);

    // Under default ratios this would not be compacted, but tight headroom forces compaction.
    expect(getToolResultText(messages[1])).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("frees at least minPassSavings per compaction pass even when overshoot is smaller", async () => {
    // overshoot 100 < hysteresis floor 640 → charsNeeded = 640 (hysteresis).
    // 200-char tool results each save 152 chars.
    // Without hysteresis: need=100, compact 1 (152≥100, stop).
    // With hysteresis:    need=640, compact all 3 (152×3=456, exhausts eligible) → more headroom freed.
    const agent = makeGuardableAgent();
    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_700)), // total 2700+600=3300, overshoot=100
      makeToolResult("call_1", "a".repeat(200)),
      makeToolResult("call_2", "b".repeat(200)),
      makeToolResult("call_3", "c".repeat(200)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    // All three compacted because hysteresis raises charsNeeded to 640.
    expect(getToolResultText(contextForNextCall[1])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[2])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    expect(getToolResultText(contextForNextCall[3])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
  });
});
