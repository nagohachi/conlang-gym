import type { ComponentChildren } from "preact";

import { type Episode, meaningToText } from "../env";
import { gloss } from "./gloss";
import { describeSeed } from "./seeds";

export type Grade = ReturnType<Episode["grade"]>;

const ORDER_NAMES: Record<string, string> = { S: "subject", O: "object", V: "verb" };

/** Plain-language statements of the language's grammar, revealed after the test. */
function describeGrammar(episode: Episode): string[] {
  const { grammar, affixes } = episode.language;
  const harmony = grammar.vowelHarmony === "copy";
  // Under vowel harmony an affix is its consonant plus a copy of the root's last vowel (V).
  const form = (name: string) => (harmony ? `${affixes[name]![0]}V` : affixes[name]!);
  const place = (position: string, name: string) =>
    position === "prefix" ? `prefix ${form(name)}-` : `suffix -${form(name)}`;
  const order = grammar.wordOrder;
  const intransitive = order.indexOf("S") < order.indexOf("V") ? "subject verb" : "verb subject";
  const lines = [
    `Word order ${order}: ${[...order].map((r) => ORDER_NAMES[r]).join(" – ")} (intransitive: ${intransitive}).`,
  ];
  if (grammar.plural !== "none") lines.push(`Plural nouns take the ${place(grammar.plural, "plural")}.`);
  if (grammar.objectMarking !== "none") lines.push(`Objects take the ${place("suffix", "object")}.`);
  if (grammar.plural === "suffix" && grammar.objectMarking === "suffix") {
    lines.push(
      grammar.pluralCaseOrder === "plural_first"
        ? "A plural object has the plural suffix first, then the object suffix."
        : "A plural object has the object suffix first, then the plural suffix.",
    );
  }
  if (grammar.past !== "none") lines.push(`Past-tense verbs take the ${place(grammar.past, "past")}.`);
  if (grammar.subjectAgreement !== "none") {
    lines.push(`A verb with a plural subject takes the ${place("suffix", "agreement")}.`);
  }
  if (harmony) lines.push("Vowel harmony: V in an affix copies the last vowel of its root.");
  return lines;
}

export function Results({
  episode,
  grade,
  who,
  children,
}: {
  episode: Episode;
  grade: Grade;
  who: "You" | "The model";
  children?: ComponentChildren;
}) {
  const { score, items } = grade;
  const percent = Math.round(score.reward * 100);
  const training = describeSeed(Number(episode.seed), episode.config).trainingAfter;
  return (
    <section class="card results" aria-labelledby="results-title">
      <h2 id="results-title">
        {who} got {score.numCorrect} of {score.numScorable} right
        <span class="score-pill" data-level={percent >= 75 ? "high" : percent >= 40 ? "mid" : "low"}>
          {percent}%
        </span>
      </h2>
      {score.testTruncated ? (
        <p class="warn-text">The answer ran past the token limit, so it scores 0.</p>
      ) : null}
      <div class="table-scroll">
        <table class="grade-table">
          <thead>
            <tr>
              <th scope="col">#</th>
              <th scope="col">Meaning</th>
              <th scope="col">Answer</th>
              <th scope="col">The language says</th>
              <th scope="col">
                <span class="visually-hidden">Result</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={item.id} class={item.scorable ? (item.correct ? "ok" : "bad") : "skip"}>
                <td>{item.id}</td>
                <td class="meaning" title={meaningToText(item.meaning)}>
                  {gloss(item.meaning)}
                </td>
                <td class="conlang answer" data-label="Answer">
                  {item.answer?.trim() ? item.answer : <em class="missing">no answer</em>}
                </td>
                <td class="conlang expected" data-label="Says">
                  {item.expected}
                </td>
                <td class="mark">
                  {!item.scorable ? (
                    <span title="Asked during the investigation, so not scored">not scored</span>
                  ) : item.correct ? (
                    <span aria-label="correct">✓</span>
                  ) : (
                    <span aria-label="wrong">✗</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <h3>The grammar</h3>
      <ul class="grammar">
        {describeGrammar(episode).map((line) => (
          <li key={line}>{line}</li>
        ))}
      </ul>
      {training && <p class="hint">{training}</p>}
      {children && <div class="actions">{children}</div>}
    </section>
  );
}
