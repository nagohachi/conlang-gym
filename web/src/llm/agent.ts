/**
 * The agent loop of conlang_gym/agent_loop.py (non-thinking mode), on top of any LLM backend.
 *
 * The token stream matches what the model saw in training (verl's Continuous Token builder for
 * Qwen3.5 with enable_thinking=False):
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 *   {generated turn, ending with <|im_end|> unless cut off}
 *   \n (only after <|im_end|>)
 *   <|im_start|>user\n{informant reply}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 *   ...
 * Investigation turns are cut off at turnMaxTokens (a format error), the test answer at
 * testMaxTokens (scores 0), and the episode at responseLength tokens after the first prompt.
 */
import type { StopReason } from "./engine";

export const IM_END_ID = 248046;
export const ENDOFTEXT_ID = 248044;
export const STOP_IDS: ReadonlySet<number> = new Set([IM_END_ID, ENDOFTEXT_ID]);

/** Limits of the published model's run (agent_loop.yaml, scripts/train_qwen3_5_2b.sh nothink). */
export const TRAINING_LIMITS = { turnMaxTokens: 512, testMaxTokens: 2048, responseLength: 6144 };
export type Limits = typeof TRAINING_LIMITS;

const SPECIAL_TOKEN_RE = /<\|[^|>]*\|>/g;

export interface LlmBackend {
  /** Forget the conversation. */
  reset(): Promise<void>;
  encode(text: string): Promise<number[]>;
  decode(ids: number[]): Promise<string>;
  /**
   * Feed `append` after everything fed so far, then greedily generate up to maxTokens tokens or
   * until a stop token. onText gets the decoded text of the tokens generated so far.
   */
  generate(
    append: number[],
    maxTokens: number,
    onText?: (text: string, numTokens: number) => void,
  ): Promise<{ ids: number[]; reason: StopReason }>;
}

/** What the loop needs from an Episode (web/src/env). */
export interface AgentEpisode<Score> {
  readonly inTestPhase: boolean;
  buildInitialPrompt(): string;
  step(text: string, truncated: boolean): string;
  score(text: string, truncated: boolean): Score;
}

export interface Turn {
  index: number;
  inTest: boolean;
  /** Decoded text of the turn's tokens, special tokens included. */
  rawText: string;
  /** The text the environment reads: special tokens removed, after the last </think>. */
  text: string;
  numTokens: number;
  truncated: boolean;
  /** The informant's reply (investigation turns only). */
  reply?: string;
  durationMs: number;
}

export interface AgentCallbacks {
  onTurnStart?(index: number, inTest: boolean): void;
  onTurnText?(index: number, text: string, numTokens: number): void;
  onTurnEnd?(turn: Turn): void;
}

export interface AgentResult<Score> {
  turns: Turn[];
  score: Score;
  /** "test" if the test answer was scored, "length" if the response length ran out, or "aborted". */
  end: "test" | "length" | "aborted";
}

export function userTurn(content: string): string {
  return `<|im_start|>user\n${content.trim()}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;
}

/** Python: _SPECIAL_TOKEN_RE.sub("", raw_text.rsplit("</think>", 1)[-1]). */
export function visibleText(rawText: string): string {
  const end = rawText.lastIndexOf("</think>");
  const afterThinking = end < 0 ? rawText : rawText.slice(end + "</think>".length);
  return afterThinking.replace(SPECIAL_TOKEN_RE, "");
}

export async function runAgent<Score>(
  llm: LlmBackend,
  episode: AgentEpisode<Score>,
  callbacks: AgentCallbacks = {},
  options: { limits?: Limits; signal?: AbortSignal } = {},
): Promise<AgentResult<Score>> {
  const limits = options.limits ?? TRAINING_LIMITS;
  const newlineIds = await llm.encode("\n");
  await llm.reset();
  let append = await llm.encode(userTurn(episode.buildInitialPrompt()));
  let responseTokens = 0;
  const turns: Turn[] = [];
  let score: Score | null = null;
  let end: AgentResult<Score>["end"] = "length";

  for (let index = 0; ; index++) {
    if (options.signal?.aborted) {
      end = "aborted";
      break;
    }
    const inTest = episode.inTestPhase;
    const answerMax = inTest ? limits.testMaxTokens : limits.turnMaxTokens;
    const maxTokens = Math.min(answerMax, limits.responseLength - responseTokens);
    if (maxTokens <= 0) break;
    callbacks.onTurnStart?.(index, inTest);
    const started = performance.now();
    const { ids, reason } = await llm.generate(append, maxTokens, (text, n) =>
      callbacks.onTurnText?.(index, text, n),
    );
    if (reason === "aborted") {
      end = "aborted";
      break;
    }
    responseTokens += ids.length;
    const truncated = reason === "length";
    const rawText = await llm.decode(ids);
    const text = visibleText(rawText);
    const turn: Turn = {
      index,
      inTest,
      rawText,
      text,
      numTokens: ids.length,
      truncated,
      durationMs: performance.now() - started,
    };
    if (inTest) {
      score = episode.score(text, truncated);
      turns.push(turn);
      callbacks.onTurnEnd?.(turn);
      end = "test";
      break;
    }
    turn.reply = episode.step(text, truncated);
    turns.push(turn);
    callbacks.onTurnEnd?.(turn);
    append = [
      ...(ids[ids.length - 1] === IM_END_ID ? newlineIds : []),
      ...(await llm.encode(userTurn(turn.reply))),
    ];
    // verl stops the episode when the context would fill the response length.
    if (responseTokens + append.length >= limits.responseLength) break;
    responseTokens += append.length;
  }
  if (score === null) score = episode.score("", true);
  return { turns, score, end };
}
