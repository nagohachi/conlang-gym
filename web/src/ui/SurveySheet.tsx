import type { Episode } from "../env";

export function SurveySheet({ episode, player }: { episode: Episode; player: "human" | "model" }) {
  const lang = episode.language;
  const { showDictionary, budget } = episode.config;
  const group = (title: string, concepts: readonly string[]) => (
    <div class="vocab-group">
      <h4>{title}</h4>
      <ul>
        {concepts.map((concept) => (
          <li key={concept}>
            <span class="concept">{concept}</span>
            {showDictionary && <span class="conlang">{lang.roots[concept]}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
  return (
    <section class="card sheet" aria-label="Survey sheet">
      <h3>Survey sheet</h3>
      {group("Nouns", lang.nouns)}
      {group("Transitive verbs", lang.transitive)}
      {group("Intransitive verbs", lang.intransitive)}
      <p class="hint">
        {player === "human" ? (
          <>
            {showDictionary
              ? "The dictionary gives each root. You find out the word order and any affixes by asking."
              : "There is no dictionary, so you find out the roots by asking as well."}{" "}
            Nouns can be singular or plural, verbs present or past. You have {budget} questions.
          </>
        ) : (
          <>
            {showDictionary
              ? "The dictionary gives each root. The model finds out the word order and any affixes by asking."
              : "There is no dictionary, so the model finds out the roots by asking as well."}{" "}
            It has {budget} questions.
          </>
        )}
      </p>
      <details class="prompt">
        <summary>Model prompt</summary>
        <pre>{episode.buildInitialPrompt()}</pre>
      </details>
    </section>
  );
}
