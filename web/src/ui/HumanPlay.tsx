import { useState } from "preact/hooks";

import { type Episode, type Meaning, NUMBERS, type ParsedAction, TENSES, makeMeaning, meaningToText } from "../env";
import { gloss } from "./gloss";
import { type Grade, Results } from "./Results";
import { SurveySheet } from "./SurveySheet";

interface LogEntry {
  action: ParsedAction;
  /** What the informant (or the environment) said. */
  reply: string;
}

function actionBody(meaning: Meaning): string {
  return `ask(${meaningToText(meaning)})`;
}

export function HumanPlay({ episode, onLetModelTry }: { episode: Episode; onLetModelTry: () => void }) {
  const [log, setLog] = useState<LogEntry[]>([]);
  const [phase, setPhase] = useState<"investigate" | "test" | "done">("investigate");
  const [answers, setAnswers] = useState<string[]>(() => episode.tests.map(() => ""));
  const [grade, setGrade] = useState<Grade | null>(null);

  const act = (text: string) => {
    const reply = episode.step(text, false);
    setLog((current) => [...current, { action: episode.lastAction!, reply }]);
    if (episode.inTestPhase) setPhase("test");
  };

  const submit = () => {
    const text = answers.map((answer, i) => `<answer id="${i + 1}">${answer}</answer>`).join("\n");
    setGrade(episode.grade(text, false));
    setPhase("done");
  };

  return (
    <div class="play">
      <aside class="play-side">
        <SurveySheet episode={episode} player="human" />
      </aside>
      <div class="play-main">
        <section class="card" aria-labelledby="investigation-title">
          <div class="card-head">
            <h2 id="investigation-title">Investigation</h2>
            <span class="budget" aria-live="polite">
              {episode.budgetLeft} question{episode.budgetLeft === 1 ? "" : "s"} left
            </span>
          </div>
          {log.length === 0 && <p class="hint">Ask how to say something. Each question uses one of your questions, even an invalid one.</p>}
          <ol class="log">
            {log.map((entry, i) => (
              <LogItem key={i} index={i + 1} entry={entry} episode={episode} />
            ))}
          </ol>
          {phase === "investigate" && <AskForm episode={episode} onAct={act} />}
        </section>

        {phase !== "investigate" && (
          <section class="card" aria-labelledby="test-title">
            <h2 id="test-title">Test</h2>
            <p class="hint">
              Translate each meaning. Lowercase letters, words separated by spaces. Answers are compared
              exactly (hyphens and extra spaces are ignored).
            </p>
            <ol class="test-items">
              {episode.tests.map((meaning, i) => (
                <li key={i}>
                  <label for={`answer-${i}`}>
                    <span class="gloss">{gloss(meaning)}</span>
                    <span class="slots">{meaningToText(meaning)}</span>
                    {episode.isAsked(meaning) && <span class="tag">you asked this one: not scored</span>}
                  </label>
                  <input
                    id={`answer-${i}`}
                    class="conlang-input"
                    autocomplete="off"
                    autocapitalize="off"
                    spellcheck={false}
                    disabled={phase === "done"}
                    value={answers[i]}
                    onInput={(event) => {
                      const value = event.currentTarget.value;
                      setAnswers((current) => current.map((a, j) => (j === i ? value : a)));
                    }}
                    onKeyDown={(event) => {
                      if (event.key === "Enter") {
                        const next = document.getElementById(`answer-${i + 1}`);
                        if (next) next.focus();
                        else if (phase === "test") submit();
                      }
                    }}
                  />
                </li>
              ))}
            </ol>
            {phase === "test" && (
              <button class="primary" onClick={submit}>
                Check my answers
              </button>
            )}
          </section>
        )}

        {grade && (
          <Results episode={episode} grade={grade} who="You">
            <button class="primary" onClick={onLetModelTry}>
              Let the model try this language
            </button>
          </Results>
        )}
      </div>
    </div>
  );
}

function LogItem({ index, entry, episode }: { index: number; entry: LogEntry; episode: Episode }) {
  const { action } = entry;
  if (action.kind === "done") return null;
  return (
    <li class="log-item">
      <span class="log-index">Q{index}</span>
      <div class="log-body">
        {action.kind === "ask" ? (
          <>
            <p class="gloss">{gloss(action.meaning)}</p>
            <p class="conlang said">{episode.language.realize(action.meaning)}</p>
          </>
        ) : action.kind === "judge" ? (
          <>
            <p class="gloss">
              Is <span class="conlang">{action.sentence}</span> grammatical?
            </p>
            <p class="said">{entry.reply.split("\n")[0]!.replace("Informant: ", "")}</p>
          </>
        ) : (
          <p class="error-text">{action.error}</p>
        )}
      </div>
    </li>
  );
}

function AskForm({ episode, onAct }: { episode: Episode; onAct: (text: string) => void }) {
  const lang = episode.language;
  const verbs = [...lang.transitive, ...lang.intransitive];
  const [subject, setSubject] = useState(lang.nouns[0]!);
  const [subjectNumber, setSubjectNumber] = useState<string>("singular");
  const [verb, setVerb] = useState(verbs[0]!);
  const [object, setObject] = useState(lang.nouns[1]!);
  const [objectNumber, setObjectNumber] = useState<string>("singular");
  const [tense, setTense] = useState<string>("present");
  const [raw, setRaw] = useState("");
  const transitive = lang.transitive.includes(verb);
  const meaning = transitive
    ? makeMeaning(subject, subjectNumber, verb, tense, object, objectNumber)
    : makeMeaning(subject, subjectNumber, verb, tense);
  const repeated = episode.isAsked(meaning);

  const select = (label: string, value: string, options: readonly string[], set: (v: string) => void) => (
    <label class="select-field">
      <span>{label}</span>
      <select value={value} onChange={(event) => set(event.currentTarget.value)}>
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <div class="ask">
      <div class="ask-grid">
        {select("Subject", subject, lang.nouns, setSubject)}
        {select("number", subjectNumber, NUMBERS, setSubjectNumber)}
        {select("Verb", verb, verbs, setVerb)}
        {transitive && select("Object", object, lang.nouns, setObject)}
        {transitive && select("number", objectNumber, NUMBERS, setObjectNumber)}
        {select("Tense", tense, TENSES, setTense)}
      </div>
      <p class="ask-preview">
        “{gloss(meaning)}”{repeated && <span class="tag">already asked</span>}
      </p>
      <div class="row">
        <button class="primary" onClick={() => onAct(`<action>${actionBody(meaning)}</action>`)}>
          Ask the informant
        </button>
        <button class="secondary" onClick={() => onAct("<action>done</action>")}>
          I'm ready for the test
        </button>
      </div>
      <details class="raw-action">
        <summary>Type an action like the model does</summary>
        <form
          class="row"
          onSubmit={(event) => {
            event.preventDefault();
            if (raw.trim()) {
              onAct(raw.includes("<action>") ? raw : `<action>${raw}</action>`);
              setRaw("");
            }
          }}
        >
          <input
            class="conlang-input"
            placeholder={actionBody(meaning)}
            value={raw}
            spellcheck={false}
            onInput={(event) => setRaw(event.currentTarget.value)}
          />
          <button class="secondary" type="submit">
            Send
          </button>
        </form>
      </details>
    </div>
  );
}
