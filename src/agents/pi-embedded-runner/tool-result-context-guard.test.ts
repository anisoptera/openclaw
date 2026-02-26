import type { AgentMessage } from "@mariozechner/pi-agent-core";
import { describe, expect, it } from "vitest";
import {
  CONTEXT_LIMIT_TRUNCATION_NOTICE,
  PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
  __testing,
  installToolResultContextGuard,
} from "./tool-result-context-guard.js";

const { calibrateCharsPerToken } = __testing;

function makeUser(text: string): AgentMessage {
  return {
    role: "user",
    content: text,
    timestamp: Date.now(),
  } as unknown as AgentMessage;
}

function makeAssistant(text: string, usage?: { input: number }): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    usage: usage ?? { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0 },
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

// With contextWindowTokens: 1_000 and fallback ratios (4 c/t general, 3 c/t tool results):
//   contextBudgetTokens = 1000 * 0.9 = 900 tokens
//   maxSingleToolResultTokens = 1000 * 0.5 = 500 tokens
//   minPassSavings = 900 * 0.15 = 135 tokens
function makeTwoToolResultOverflowContext(): AgentMessage[] {
  // user: 2600 chars = 650 tokens (at 4 c/t)
  // tool1: 1000 chars = ceil(1000/3) = 334 tokens; tool2: same
  // total: 650 + 334 + 334 = 1318 > 900 budget, overshoot 418
  // tokensNeeded = max(418, 135) = 418
  // Each compaction saves 334-12=322 tokens — both needed to cover 418 (322<418, then 644≥418).
  return [
    makeUser("u".repeat(2_600)),
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

    // user: 2800 chars = 700 tokens (at 4 c/t); 3 tool results: 800 chars = ceil(800/3)=267 tokens each
    // total: 700 + 267*3 = 1501 > 900 budget, overshoot 601
    // tokensNeeded = max(601, 135) = 601
    // Each compaction saves 267-12=255 tokens — need all 3 (255<601, 510<601, 765≥601).
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

    // 5000 chars = ceil(5000/3) = 1667 tokens > maxSingleToolResultTokens (500)
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

    // user: 2600 chars = 650 tokens; tool1: 700 chars = ceil(700/3)=234 tokens; tool2: 1000 = 334
    // total: 650+234+334 = 1218 > 900 budget, overshoot 318
    // tokensNeeded = max(318, 135) = 318; tool1 saves 222, tool2 saves 322; 222<318 then 544≥318
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

    // user: 2000 chars = ceil(2000/4)=500 tokens; tool1: 1000 = ceil(1000/3)=334 tokens; tool2: same
    // total: 500+334+334 = 1168 > 900. Overshoot: 268. tokensNeeded=max(268,135)=268.
    // tool1 savings = 322 ≥ 268 → oldest compacted, sufficient.
    const contextForNextCall = [
      makeUser("u".repeat(2_000)),
      makeLegacyToolResult("call_old", "x".repeat(1_000)),
      makeLegacyToolResult("call_new", "y".repeat(1_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResultText = (contextForNextCall[1] as { content?: unknown }).content;

    // Oldest legacy tool result gets compacted.
    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
  });

  it("skips compacting tool results below minimum savings threshold", async () => {
    // contextWindowTokens: 1_000 => contextBudgetTokens = 900
    // placeholder ≈ ceil(48/4)=12 tokens; minPerResultSavings = 12 * 2 = 24 tokens
    // Small tool result: 50 chars at toolResultCharsPerToken=3 → ceil(50/3)=17 tokens
    // potentialSavings = 17 - 12 = 5 < 24 → skipped per-result
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
    });

    // Push total over 900 budget with a large user message, but small tool results
    // that don't meet the per-result savings threshold.
    const smallText = "s".repeat(50);
    const contextForNextCall = [
      makeUser("u".repeat(3_700)), // 925 tokens
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
    // contextWindowTokens: 1_000 => contextBudgetTokens = 900, minPassSavings = 135
    // Tool result with 500 chars = ceil(500/3)=167 tokens; placeholder ≈ 12 tokens
    // perResultSavings = 155 >= 24 => eligible
    // Two results: totalEligibleSavings = 310 >= 135 => passes gate
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const largeText = "L".repeat(500);
    // user: 2800 chars = 700 tokens; tool1: ceil(500/3)=167 tokens; tool2: same
    // total: 700+167+167 = 1034 > 900 budget, overshoot 134, tokensNeeded = max(134,135) = 135
    const contextForNextCall = [
      makeUser("u".repeat(2_800)),
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

    // Per-result truncation: each tool result has 900 chars text + ~8060 chars details (JSON).
    // Total raw = 8960 chars → ceil(8960/3)=2987 tokens > maxSingle 500 → truncate.
    // maxChars = floor(500*3)=1500. Text is 900 chars ≤ 1500, so text unchanged but details dropped.
    // After truncation: each tool = ceil(900/3)=300 tokens.
    // user: 2200 chars = ceil(2200/4)=550 tokens. Total: 550+300+300 = 1150 > 900 budget.
    // overshoot=250, tokensNeeded=max(250,135)=250. savings=300-12=288≥250 → compact oldest only.
    // The newer result's details are stripped by per-result truncation but its text stays.
    const contextForNextCall = [
      makeUser("u".repeat(2_200)),
      makeToolResultWithDetails("call_old", "x".repeat(900), "d".repeat(8_000)),
      makeToolResultWithDetails("call_new", "y".repeat(900), "d".repeat(8_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    const oldResult = contextForNextCall[1] as unknown as { details?: unknown };
    const newResult = contextForNextCall[2] as unknown as { details?: unknown };
    const oldResultText = getToolResultText(contextForNextCall[1]);

    // Oldest result is compacted (placeholder replaces text, details dropped).
    expect(oldResultText).toBe(PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER);
    expect(oldResult.details).toBeUndefined();
    // Newer result: details stripped by per-result truncation, but text content preserved.
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
    // user: 500 chars = ceil(500/4)=125 tokens; 4 tool results: 800 chars = ceil(800/3)=267 tokens each
    // total: 125 + 267*4 = 1193 > 900 budget, overshoot 293, tokensNeeded = max(293,135)=293
    // Only call_1 eligible (last 3 protected). savings=255≥293 not met — compact call_1 (255 freed,
    // still 38 below tokensNeeded but no more eligible). Context stays slightly over budget.
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
    // user: 1600 chars = ceil(1600/4)=400 tokens; 3 tool results: 800 chars = ceil(800/3)=267 tokens each
    // total: 400 + 267*3 = 1201 > 900 budget, overshoot 301, tokensNeeded = max(301,135)=301
    // eligible: call_old(255) + call_mid(255) = 510 ≥ 135 ✓; compact both (255<301, then 510≥301)
    const contextForNextCall = [
      makeUser("u".repeat(1_600)),
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

describe("calibrateCharsPerToken", () => {
  it("falls back to CHARS_PER_TOKEN_ESTIMATE when no assistant messages exist", () => {
    const messages = [makeUser("hello"), makeToolResult("t1", "result")];
    const cal = calibrateCharsPerToken(messages);
    // Fallback: general ratio 4, tool-result ratio 3 (code/JSON tokenize denser).
    expect(cal.charsPerToken).toBe(4); // CHARS_PER_TOKEN_ESTIMATE
    expect(cal.toolResultCharsPerToken).toBe(3); // TOOL_RESULT_CHARS_PER_TOKEN_ESTIMATE
  });

  it("falls back when assistant has no usage data", () => {
    const messages = [makeUser("hello"), makeAssistant("response")];
    const cal = calibrateCharsPerToken(messages);
    expect(cal.charsPerToken).toBe(4);
    expect(cal.toolResultCharsPerToken).toBe(3);
  });

  it("derives empirical ratio from messages before the assistant (excludes assistant output)", () => {
    // Only messages BEFORE the assistant (index < lastAssistantIdx) are counted.
    // user: 400 chars; assistant output is NOT included in usage.input, so excluded.
    // usage.input = 200 tokens → ratio = 400/200 = 2.0 chars/token
    // When empirical data is available, both ratios equal the empirical value.
    const messages = [makeUser("x".repeat(400)), makeAssistant("y".repeat(200), { input: 200 })];
    const cal = calibrateCharsPerToken(messages);
    expect(cal.charsPerToken).toBe(2);
    expect(cal.toolResultCharsPerToken).toBe(2);
  });

  it("uses the last assistant message's usage for calibration", () => {
    // Chars before last assistant (indices 0,1,2): user(300) + assistant(200) + tool(400) = 900
    // usage.input on last assistant = 300 → ratio = 900/300 = 3.0
    const messages = [
      makeUser("x".repeat(300)),
      makeAssistant("y".repeat(200), { input: 100 }),
      makeToolResult("t1", "z".repeat(400)),
      makeAssistant("w".repeat(200), { input: 300 }),
    ];
    const cal = calibrateCharsPerToken(messages);
    expect(cal.charsPerToken).toBe(3);
    expect(cal.toolResultCharsPerToken).toBe(3);
  });

  it("clamps ratio to sane range", () => {
    // Absurdly high ratio: 10000 chars / 100 tokens = 100 → clamped to 8
    const messages = [makeUser("x".repeat(10000)), makeAssistant("y", { input: 100 })];
    const cal = calibrateCharsPerToken(messages);
    expect(cal.charsPerToken).toBe(8);
    expect(cal.toolResultCharsPerToken).toBe(8);

    // Absurdly low ratio: 10 chars / 1000 tokens = 0.01 → clamped to 1.5
    const messages2 = [makeUser("x".repeat(10)), makeAssistant("y", { input: 1000 })];
    const cal2 = calibrateCharsPerToken(messages2);
    expect(cal2.charsPerToken).toBe(1.5);
    expect(cal2.toolResultCharsPerToken).toBe(1.5);
  });
});

describe("empirical calibration integration", () => {
  it("uses empirical ratio for budget decisions when usage data is available", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    // Empirical calibration: chars before assistant (user 400) / usage.input 200 = 2.0 ratio.
    // At ratio 2.0, tool result 800 chars = 400 tokens (< maxSingleToolResultTokens 500 → no truncation).
    // Total: user 400/2=200 + assistant 200/2=100 + tool 800/2=400 = 700 < budget 900
    // → should NOT compact
    const contextForNextCall = [
      makeUser("x".repeat(400)),
      makeAssistant("y".repeat(200), { input: 200 }),
      makeToolResult("call_1", "z".repeat(800)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);
    expect(getToolResultText(contextForNextCall[2])).toBe("z".repeat(800));
  });

  it("compacts when empirical ratio shows context is over budget", async () => {
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    // Empirical calibration: chars before assistant (user 400) / usage.input 200 = 2.0 ratio.
    // At ratio 2.0 (both charsPerToken and toolResultCharsPerToken equal empirical):
    //   user: 400/2 = 200 tokens; assistant: 200/2 = 100 tokens
    //   tool1: 1000/2 = 500 tokens (exactly maxSingle — no truncation)
    //   tool2: 1000/2 = 500 tokens → total 1300 > 900 budget
    // overshoot=400, minPassSavings=135, tokensNeeded=400
    // placeholder ≈ ceil(48/2)=24; perResult savings=476≥48 ✓; total eligible 952≥135 ✓
    // compact tool1: saves 476≥400 → stop
    const contextForNextCall = [
      makeUser("x".repeat(400)),
      makeAssistant("y".repeat(200), { input: 200 }),
      makeToolResult("call_1", "a".repeat(1_000)),
      makeToolResult("call_2", "b".repeat(1_000)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);
    // Oldest tool result should be compacted
    expect(getToolResultText(contextForNextCall[2])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
  });
});

describe("hysteresis: minimum tokens freed per compaction pass", () => {
  it("frees at least minPassSavings tokens even when overshoot is smaller", async () => {
    // contextWindowTokens: 1_000 → budget 900, minPassSavings = floor(900 * 0.15) = 135
    // user: 2000 chars = 500 tokens; tool1: 800 chars = 267 tokens; tool2: 800 chars = 267 tokens
    // total: 1034 > 900. overshoot = 134 < minPassSavings 135 → tokensNeeded = 135
    // Each tool savings = 267-12 = 255 ≥ 135 → compact tool1, save 255 ≥ 135, stop.
    // Without hysteresis, tool1 would have been compacted for just 134 tokens (busting cache
    // for almost nothing). With hysteresis, we free 255 tokens — meaningful headroom.
    const agent = makeGuardableAgent();

    installToolResultContextGuard({
      agent,
      contextWindowTokens: 1_000,
      recentToolResultsToPreserve: 0,
    });

    const contextForNextCall = [
      makeUser("u".repeat(2_000)),
      makeToolResult("call_1", "a".repeat(800)),
      makeToolResult("call_2", "b".repeat(800)),
    ];

    await agent.transformContext?.(contextForNextCall, new AbortController().signal);

    // Oldest result is compacted (255 freed ≥ 135 = minPassSavings).
    expect(getToolResultText(contextForNextCall[1])).toBe(
      PREEMPTIVE_TOOL_RESULT_COMPACTION_PLACEHOLDER,
    );
    // Newer result is still intact (compacting tool1 already met the minPassSavings threshold).
    expect(getToolResultText(contextForNextCall[2])).toBe("b".repeat(800));
  });
});
