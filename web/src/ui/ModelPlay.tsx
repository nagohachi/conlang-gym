import { useEffect, useMemo, useRef, useState } from "preact/hooks";

import { type EnvConfig, Episode } from "../env";
import { runAgent, type Turn } from "../llm/agent";
import { getClient, loadModel, useModelState } from "./model-store";
import { type Grade, Results } from "./Results";
import { SurveySheet } from "./SurveySheet";

interface LiveTurn {
  index: number;
  inTest: boolean;
  text: string;
  numTokens: number;
  startedAt: number;
}

interface RunState {
  episode: Episode;
  turns: Turn[];
  live: LiveTurn | null;
  status: "running" | "done" | "stopped" | "error";
  error?: string;
  grade?: Grade;
  end?: string;
}

const SPECIAL_TOKEN_RE = /<\|[^|>]*\|>/g;

export function ModelPlay({
  seed,
  config,
  onTryYourself,
}: {
  seed: number;
  config: EnvConfig;
  onTryYourself: () => void;
}) {
  const model = useModelState();
  const preview = useMemo(() => new Episode(seed, config), [seed, config]);
  const [run, setRun] = useState<RunState | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  // Leaving the episode (e.g. starting another) stops its generation.
  useEffect(
    () => () => {
      const controller = abortRef.current;
      if (controller && !controller.signal.aborted) {
        controller.abort();
        getClient().abort();
      }
    },
    [],
  );

  const start = async () => {
    const episode = new Episode(seed, config);
    const controller = new AbortController();
    abortRef.current = controller;
    setRun({ episode, turns: [], live: null, status: "running" });
    try {
      const result = await runAgent(
        getClient(),
        episode,
        {
          onTurnStart: (index, inTest) =>
            setRun((r) => r && { ...r, live: { index, inTest, text: "", numTokens: 0, startedAt: performance.now() } }),
          onTurnText: (index, text, numTokens) =>
            setRun((r) => r && r.live?.index === index ? { ...r, live: { ...r.live, text, numTokens } } : r),
          onTurnEnd: (turn) => setRun((r) => r && { ...r, live: null, turns: [...r.turns, turn] }),
        },
        { signal: controller.signal },
      );
      const testTurn = result.turns.find((turn) => turn.inTest);
      const grade = episode.grade(testTurn?.text ?? "", testTurn ? testTurn.truncated : true);
      setRun((r) => r && { ...r, live: null, status: result.end === "aborted" ? "stopped" : "done", grade, end: result.end });
    } catch (error) {
      setRun((r) => r && { ...r, live: null, status: "error", error: error instanceof Error ? error.message : String(error) });
    } finally {
      // Only a run in progress has a controller, so that leaving afterwards aborts nothing.
      if (abortRef.current === controller) abortRef.current = null;
    }
  };

  const stop = () => {
    abortRef.current?.abort();
    getClient().abort();
  };

  const episode = run?.episode ?? preview;
  const totalTokens = run ? run.turns.reduce((n, t) => n + t.numTokens, 0) + (run.live?.numTokens ?? 0) : 0;
  const totalMs = run
    ? run.turns.reduce((n, t) => n + t.durationMs, 0) + (run.live ? performance.now() - run.live.startedAt : 0)
    : 0;

  return (
    <div class="play">
      <aside class="play-side">
        <SurveySheet episode={episode} player="model" />
      </aside>
      <div class="play-main">
        <section class="card" aria-labelledby="model-title">
          <div class="card-head">
            <h2 id="model-title">The model's investigation</h2>
            {run && (
              <span class="budget meta" aria-live="polite">
                <span>
                  {episode.inTestPhase ? "test" : `${episode.budgetLeft} question${episode.budgetLeft === 1 ? "" : "s"} left`}
                </span>
                {totalMs > 0 && <span>{totalTokens} tokens</span>}
                {totalMs > 0 && <span>{((totalTokens / totalMs) * 1000).toFixed(1)} tok/s</span>}
              </span>
            )}
          </div>
          {model.status !== "ready" ? (
            <ModelLoader />
          ) : !run ? (
            <div class="row">
              <button class="primary" onClick={start}>
                Run the model
              </button>
              <p class="hint inline">Greedy decoding, same prompts and limits as in training.</p>
            </div>
          ) : null}
          {run && (
            <ol class="transcript">
              {run.turns.map((turn) => (
                <TurnView key={turn.index} turn={turn} />
              ))}
              {run.live && (
                <li class="turn live">
                  <div class="turn-head meta">
                    <span>{run.live.inTest ? "Test answer" : `Turn ${run.live.index + 1}`}</span>
                    <span>{run.live.numTokens} tokens</span>
                  </div>
                  <pre class="model-text">
                    {run.live.text.replace(SPECIAL_TOKEN_RE, "")}
                    <span class="cursor" aria-hidden="true" />
                  </pre>
                </li>
              )}
            </ol>
          )}
          {run?.status === "running" && (
            <button class="secondary" onClick={stop}>
              Stop
            </button>
          )}
          {run?.status === "stopped" && <p class="hint">Stopped.</p>}
          {run?.status === "error" && <p class="error-text">{run.error}</p>}
          {run?.end === "length" && (
            <p class="warn-text">The model used up its token budget before finishing the test.</p>
          )}
        </section>

        {run?.status === "done" && run.grade && (
          <Results episode={run.episode} grade={run.grade} who="The model">
            <button class="primary" onClick={onTryYourself}>
              Try this language yourself
            </button>
            <button class="secondary" onClick={start}>
              Run again
            </button>
          </Results>
        )}
      </div>
    </div>
  );
}

function TurnView({ turn }: { turn: Turn }) {
  return (
    <li class={`turn${turn.truncated ? " truncated" : ""}`}>
      <div class="turn-head meta">
        <span>{turn.inTest ? "Test answer" : `Turn ${turn.index + 1}`}</span>
        <span>{turn.numTokens} tokens</span>
        <span>{(turn.durationMs / 1000).toFixed(1)} s</span>
        {turn.truncated && <span>cut off at the token limit</span>}
      </div>
      <pre class="model-text">{turn.text}</pre>
      {turn.reply !== undefined && <EnvReply reply={turn.reply} />}
    </li>
  );
}

const TEST_PROMPT_START = "The investigation is over.";

/**
 * The environment's reply (Episode.step): "Informant: …" or an error, then either the questions left
 * or, when the investigation ends, the test prompt; after done, only the test prompt.
 */
function EnvReply({ reply }: { reply: string }) {
  const [first, ...others] = reply.split("\n");
  const informant = first!.startsWith("Informant: ") ? first!.slice("Informant: ".length) : null;
  const error = first!.startsWith("Error: ") ? first! : null;
  const rest = (informant !== null || error !== null ? others.join("\n") : reply).trim();
  const testStart = rest.indexOf(TEST_PROMPT_START);
  const note = (testStart < 0 ? rest : rest.slice(0, testStart)).trim();
  const testPrompt = testStart < 0 ? "" : rest.slice(testStart);
  return (
    <div class="env-reply">
      {informant !== null && (
        <p>
          <span class="who">Informant</span>
          <span class="conlang said">{informant}</span>
        </p>
      )}
      {error && <p class="error-text">{error}</p>}
      {(note || testPrompt) && (
        <p class="env-note">{note || "The model ends the investigation."}</p>
      )}
      {testPrompt && (
        <details class="test-prompt">
          <summary>Test prompt</summary>
          <pre class="env-text">{testPrompt}</pre>
        </details>
      )}
    </div>
  );
}

function ModelLoader() {
  const model = useModelState();
  if (model.status === "unchecked") return <p class="hint">Checking WebGPU support…</p>;
  if (model.status === "unsupported") {
    return (
      <div class="notice">
        <p>
          <strong>This browser can't run the model.</strong> {model.message}
        </p>
        <p class="hint">You can still play the episode yourself.</p>
      </div>
    );
  }
  if (model.status === "loading") {
    const progress = model.progress;
    const fraction = progress && progress.total ? progress.loaded / progress.total : 0;
    return (
      <div class="loader">
        <progress value={progress?.phase === "init" ? undefined : fraction} max={1} />
        <p class="hint">
          {progress?.phase === "init"
            ? "Preparing the model for your GPU…"
            : progress
              ? `Downloading ${(progress.loaded / 1e6).toFixed(0)} / ${(progress.total / 1e6).toFixed(0)} MB`
              : "Starting…"}
        </p>
      </div>
    );
  }
  return (
    <div class="loader">
      <button class="primary" onClick={() => void loadModel()}>
        Load the model (1.4 GB)
      </button>
      <p class="hint">
        Downloaded once, then cached by your browser.
      </p>
      {model.status === "error" && <p class="error-text">Loading failed: {model.message}</p>}
    </div>
  );
}
