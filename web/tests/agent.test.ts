/** The agent loop's token stream and limits (web/src/llm/agent.ts) against a scripted backend. */
import { describe, expect, it } from "vitest";

import { Episode, makeEnvConfig } from "../src/env";
import { IM_END_ID, runAgent, userTurn, visibleText, type LlmBackend } from "../src/llm/agent";
import type { StopReason } from "../src/llm/engine";

const IM_END = "<|im_end|>";

/** One id per character; IM_END_ID stands for <|im_end|>. */
class ScriptedBackend implements LlmBackend {
  /** Every token fed to the model, in order: appended context and generated tokens. */
  stream: number[] = [];
  calls: { append: number[]; maxTokens: number }[] = [];

  constructor(private readonly turns: { text: string; reason: StopReason }[]) {}

  async reset(): Promise<void> {
    this.stream = [];
  }

  async encode(text: string): Promise<number[]> {
    const ids: number[] = [];
    for (let i = 0; i < text.length; i++) {
      if (text.startsWith(IM_END, i)) {
        ids.push(IM_END_ID);
        i += IM_END.length - 1;
      } else {
        ids.push(text.charCodeAt(i));
      }
    }
    return ids;
  }

  async decode(ids: number[]): Promise<string> {
    return ids.map((id) => (id === IM_END_ID ? IM_END : String.fromCharCode(id))).join("");
  }

  async generate(append: number[], maxTokens: number) {
    this.calls.push({ append, maxTokens });
    const turn = this.turns.shift();
    if (!turn) throw new Error("no scripted turn left");
    let ids = await this.encode(turn.text);
    if (turn.reason === "eos") ids.push(IM_END_ID);
    ids = ids.slice(0, maxTokens);
    this.stream.push(...append, ...ids);
    return { ids, reason: turn.reason };
  }
}

const config = makeEnvConfig({ activeParams: [] });

describe("visibleText", () => {
  it("keeps the text after the last </think> without special tokens", () => {
    expect(visibleText("a</think>b</think>\n<action>done</action><|im_end|>")).toBe("\n<action>done</action>");
    expect(visibleText("<action>done</action><|im_end|>")).toBe("<action>done</action>");
  });
});

describe("runAgent", () => {
  it("reproduces the training token stream", async () => {
    const episode = new Episode(1, config);
    const ask = `<action>ask(Subject=${episode.language.nouns[0]}, SubjectNumber=singular, Verb=${episode.language.intransitive[0]}, Tense=present)</action>`;
    const backend = new ScriptedBackend([
      { text: ask, reason: "eos" },
      { text: "<action>done</action>", reason: "eos" },
      { text: '<answer id="1">x</answer>', reason: "eos" },
    ]);
    const reference = new Episode(1, config);
    const result = await runAgent(backend, episode);

    const reply1 = reference.step(ask, false);
    const reply2 = reference.step("<action>done</action>", false);
    const expected = await backend.encode(
      userTurn(reference.buildInitialPrompt()) +
        ask + IM_END + "\n" + userTurn(reply1) +
        "<action>done</action>" + IM_END + "\n" + userTurn(reply2) +
        '<answer id="1">x</answer>' + IM_END,
    );
    expect(backend.stream).toEqual(expected);
    expect(backend.calls.map((c) => c.maxTokens)).toEqual([512, 512, 2048]);
    expect(result.end).toBe("test");
    expect(result.turns.map((t) => t.reply)).toEqual([reply1, reply2, undefined]);
    expect(result.score).toEqual(reference.score('<answer id="1">x</answer>', false));
  });

  it("treats a turn cut off at the limit as a format error and adds no newline", async () => {
    const episode = new Episode(3, config);
    const backend = new ScriptedBackend([
      { text: "x".repeat(600), reason: "length" },
      { text: "<action>done</action>", reason: "eos" },
      { text: "", reason: "eos" },
    ]);
    const result = await runAgent(backend, episode, {}, {
      limits: { turnMaxTokens: 512, testMaxTokens: 2048, responseLength: 6144 },
    });
    expect(result.turns[0]!.truncated).toBe(true);
    expect(result.turns[0]!.reply!.startsWith("Error: format error.")).toBe(true);
    // The context after the cut-off turn starts with <|im_start|>, not with a newline.
    expect(backend.calls[1]!.append[0]).toBe("<".charCodeAt(0));
    expect(episode.numInvalid).toBe(1);
  });

  it("ends the episode when the response length runs out", async () => {
    const episode = new Episode(4, config);
    const backend = new ScriptedBackend(Array.from({ length: 10 }, () => ({ text: "y".repeat(100), reason: "length" as const })));
    const result = await runAgent(backend, episode, {}, {
      limits: { turnMaxTokens: 100, testMaxTokens: 100, responseLength: 250 },
    });
    expect(result.end).toBe("length");
    // Turn 1 generates 100 tokens; its reply context brings the total under 250, so turn 2 may
    // generate only what is left (verl's room = response_length - len(response_mask)), and the
    // context after it no longer fits.
    const room = 250 - 100 - backend.calls[1]!.append.length;
    expect(backend.calls.map((c) => c.maxTokens)).toEqual([100, room]);
    expect(result.turns).toHaveLength(2);
    expect(result.score).toMatchObject({ reward: 0, numInvalid: 2, testTruncated: 1 });
  });
});
