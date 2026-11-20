import { useEffect, useRef, useState } from "preact/hooks";

import { Episode, type EnvConfig } from "../env";
import { HumanPlay } from "./HumanPlay";
import { ModelPlay } from "./ModelPlay";
import { MODEL_REPO } from "./model-store";
import { Setup } from "./Setup";
import type { Mode } from "./seeds";

export interface Game {
  id: number;
  mode: Mode;
  seed: number;
  config: EnvConfig;
}

let nextGameId = 1;

export function App() {
  const [game, setGame] = useState<Game | null>(null);
  const playRef = useRef<HTMLDivElement>(null);

  const start = (mode: Mode, seed: number, config: EnvConfig) =>
    setGame({ id: nextGameId++, mode, seed, config });

  useEffect(() => {
    if (game) playRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [game?.id]);

  return (
    <>
      <header class="site-header">
        <div class="wrap">
          <p class="eyebrow">conlang-gym</p>
          <h1>Learn a language from one informant and a handful of questions</h1>
          <p class="lede">
            Every seed generates a small artificial language. Ask a native speaker how to say a few
            meanings, work out the grammar, then translate sentences you have never heard. Play it
            yourself, or watch a 2B-parameter model trained with reinforcement learning do it.
          </p>
        </div>
      </header>
      <main class="wrap">
        <HowItWorks />
        <Setup game={game} onStart={start} />
        <div ref={playRef} class="play-anchor">
          {game && <Play key={game.id} game={game} onRestart={start} />}
        </div>
      </main>
      <footer class="site-footer wrap">
        <p>
          Source: <a href="https://github.com/nagohachi/conlang-gym">conlang-gym</a>. Model:{" "}
          <a href={`https://huggingface.co/${MODEL_REPO}`}>{MODEL_REPO.split("/")[1]}</a>, quantized to
          4 bits.
        </p>
      </footer>
    </>
  );
}

function HowItWorks() {
  return (
    <section class="how" aria-label="How it works">
      <ol>
        <li>
          <strong>A seed fixes a language.</strong> Its word order is one of SOV, SVO, VSO, VOS, OSV or
          OVS; harder settings add affixes for plural, past tense, object case, agreement and vowel
          harmony.
        </li>
        <li>
          <strong>Ask the informant.</strong> Each question names a meaning (who does what to whom,
          singular or plural, present or past) and the informant says it in the language.
        </li>
        <li>
          <strong>Translate the test.</strong> Then come meanings you have not heard. Only exact
          sentences count, and test items you happened to ask about are not scored.
        </li>
      </ol>
    </section>
  );
}

function Play({
  game,
  onRestart,
}: {
  game: Game;
  onRestart: (mode: Mode, seed: number, config: EnvConfig) => void;
}) {
  // A fresh Episode per game; ModelPlay makes its own for re-runs.
  const [episode] = useState(() => new Episode(game.seed, game.config));
  const other: Mode = game.mode === "human" ? "model" : "human";
  const switchMode = () => onRestart(other, game.seed, game.config);
  return game.mode === "human" ? (
    <HumanPlay episode={episode} onLetModelTry={switchMode} />
  ) : (
    <ModelPlay seed={game.seed} config={game.config} onTryYourself={switchMode} />
  );
}

