import { useMemo, useState } from "preact/hooks";

import { type EnvConfig, makeEnvConfig } from "../env";
import type { Game } from "./App";
import { describeSeed, type Mode, pickSeed, sameConfig, TRAINING_CONFIG } from "./seeds";

interface Props {
  game: Game | null;
  onStart: (mode: Mode, seed: number, config: EnvConfig) => void;
}

const CUSTOM_FIELDS: { key: keyof EnvConfig; label: string; min: number; max: number }[] = [
  { key: "numActiveParams", label: "Grammar features besides word order", min: 0, max: 5 },
  { key: "budget", label: "Questions", min: 1, max: 30 },
  { key: "numTest", label: "Test sentences", min: 1, max: 20 },
  { key: "numCompositional", label: "Test sentences combining plural and past", min: 0, max: 20 },
  { key: "numNouns", label: "Nouns", min: 2, max: 10 },
  { key: "numTransitive", label: "Transitive verbs", min: 1, max: 6 },
  { key: "numIntransitive", label: "Intransitive verbs", min: 1, max: 4 },
];

function parseSeed(text: string): number | null {
  if (!/^\d{1,15}$/.test(text.trim())) return null;
  return Number(text.trim());
}

export function Setup({ game, onStart }: Props) {
  const [mode, setMode] = useState<Mode>(game?.mode ?? "human");
  const [seedText, setSeedText] = useState(() => String(pickSeed(TRAINING_CONFIG)));
  const [custom, setCustom] = useState(false);
  const [fields, setFields] = useState<EnvConfig>(TRAINING_CONFIG);

  let config: EnvConfig | null = TRAINING_CONFIG;
  let configError: string | null = null;
  if (custom) {
    try {
      config = makeEnvConfig(fields);
    } catch (error) {
      config = null;
      configError = error instanceof Error ? error.message.replace(/^EnvConfig\./, "") : String(error);
    }
  }
  const seed = parseSeed(seedText);
  const configKey = config ? JSON.stringify(config) : "";
  const info = useMemo(
    () => (seed !== null && config ? describeSeed(seed, config) : null),
    [seed, configKey],
  );
  const offDistribution = mode === "model" && config !== null && !sameConfig(config, TRAINING_CONFIG);

  const setField = (key: keyof EnvConfig, value: number | boolean) =>
    setFields((current) => ({ ...current, [key]: value }));

  return (
    <section class="card setup" aria-labelledby="setup-title">
      <h2 id="setup-title">Set up an episode</h2>

      <div class="field">
        <span class="label" id="mode-label">Who does the field work?</span>
        <div class="segmented" role="radiogroup" aria-labelledby="mode-label">
          <button
            role="radio"
            aria-checked={mode === "human"}
            class={mode === "human" ? "on" : ""}
            onClick={() => setMode("human")}
          >
            You
          </button>
          <button
            role="radio"
            aria-checked={mode === "model"}
            class={mode === "model" ? "on" : ""}
            onClick={() => setMode("model")}
          >
            The model
          </button>
        </div>
        <p class="hint">
          {mode === "human"
            ? "You ask the questions and write the translations."
            : "Qwen3.5-2B, trained with GRPO on this task, plays the episode on your GPU. The first run downloads 1.4 GB."}
        </p>
      </div>

      <div class="field">
        <label class="label" for="seed">
          Language
        </label>
        <div class="row">
          <input
            id="seed"
            class="seed-input"
            inputMode="numeric"
            value={seedText}
            onInput={(event) => setSeedText(event.currentTarget.value)}
            aria-describedby="seed-hint"
            aria-invalid={seed === null}
          />
          <button
            class="secondary"
            onClick={() => config && setSeedText(String(pickSeed(config)))}
            disabled={!config}
          >
            New language
          </button>
        </div>
        <p class="hint" id="seed-hint">
          {seed === null
            ? "A seed is a whole number."
            : info && (
                <>
                  {info.solverQuestions === null &&
                    `This language may not be solvable in ${config!.budget} questions. `}
                  {info.training}
                </>
              )}
        </p>
      </div>

      <div class="field">
        <span class="label" id="difficulty-label">
          Difficulty
        </span>
        <div class="segmented" role="radiogroup" aria-labelledby="difficulty-label">
          <button role="radio" aria-checked={!custom} class={!custom ? "on" : ""} onClick={() => setCustom(false)}>
            As in training
          </button>
          <button role="radio" aria-checked={custom} class={custom ? "on" : ""} onClick={() => setCustom(true)}>
            Custom
          </button>
        </div>
        {!custom ? (
          <p class="hint">
            The model was trained with these settings. Languages differ only in word order, the root
            dictionary is shown, and there are {TRAINING_CONFIG.budget} questions and{" "}
            {TRAINING_CONFIG.numTest} test sentences.
          </p>
        ) : (
          <div class="custom-grid">
            {CUSTOM_FIELDS.map(({ key, label, min, max }) => (
              <label class="number-field" key={key}>
                <span>{label}</span>
                <input
                  type="number"
                  min={min}
                  max={max}
                  value={fields[key] as number}
                  onInput={(event) => setField(key, Number(event.currentTarget.value))}
                />
              </label>
            ))}
            <label class="check-field">
              <input
                type="checkbox"
                checked={fields.showDictionary}
                onChange={(event) => setField("showDictionary", event.currentTarget.checked)}
              />
              <span>Show the root dictionary</span>
            </label>
            <label class="check-field">
              <input
                type="checkbox"
                checked={fields.enableJudge}
                onChange={(event) => setField("enableJudge", event.currentTarget.checked)}
              />
              <span>Allow grammaticality judgements (judge action)</span>
            </label>
          </div>
        )}
        {configError && <p class="error-text">{configError}</p>}
        {offDistribution && (
          <p class="warn-text">
            The model was trained only on the default setting, so expect it to struggle here.
          </p>
        )}
      </div>

      <button
        class="primary"
        disabled={seed === null || !config}
        onClick={() => seed !== null && config && onStart(mode, seed, config)}
      >
        {game ? "Start a new episode" : "Start"}
      </button>
    </section>
  );
}
