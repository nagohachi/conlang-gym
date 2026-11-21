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
            conlang-gym is a game where you work out a small made-up language. You can play it
            yourself, or watch a 2B-parameter model play it. The model was trained on this game with
            reinforcement learning.
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
    <section class="how" aria-labelledby="how-title">
      <h2 id="how-title">How to play</h2>
      <ol>
        <li>
          You get the nouns and verbs of the language, and by default the root of each. The subject,
          object and verb always appear in the same order, which differs from language to language
          (SOV, SVO, VSO, VOS, OSV or OVS). Plural nouns and past-tense verbs take a prefix or a
          suffix, and harder settings add more affixes.
        </li>
        <li>
          Each question names a meaning, such as “the dogs chased the cat”, and the informant says it
          in the language. You have six questions by default.
        </li>
        <li>
          Then you translate meanings you have not heard. An answer counts only if it is exactly the
          sentence the informant would say. Test items you already asked about are not scored.
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

