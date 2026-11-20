/** English glosses of meanings, so that a person can read "the men helped a bird" at a glance. */
import type { Meaning } from "../env";

const PLURAL_NOUNS: Record<string, string> = {
  dog: "dogs",
  cat: "cats",
  bird: "birds",
  fish: "fish",
  horse: "horses",
  frog: "frogs",
  child: "children",
  woman: "women",
  man: "men",
  bear: "bears",
};

const VERB_FORMS: Record<string, { third: string; past: string }> = {
  see: { third: "sees", past: "saw" },
  chase: { third: "chases", past: "chased" },
  bite: { third: "bites", past: "bit" },
  call: { third: "calls", past: "called" },
  help: { third: "helps", past: "helped" },
  hear: { third: "hears", past: "heard" },
  sleep: { third: "sleeps", past: "slept" },
  run: { third: "runs", past: "ran" },
  sing: { third: "sings", past: "sang" },
  swim: { third: "swims", past: "swam" },
};

function nounPhrase(noun: string, number: string | null): string {
  return number === "plural" ? `the ${PLURAL_NOUNS[noun] ?? noun + "s"}` : `the ${noun}`;
}

export function gloss(meaning: Meaning): string {
  const forms = VERB_FORMS[meaning.verb] ?? { third: meaning.verb + "s", past: meaning.verb + "ed" };
  const verb =
    meaning.tense === "past" ? forms.past : meaning.subjectNumber === "plural" ? meaning.verb : forms.third;
  const words = [nounPhrase(meaning.subject, meaning.subjectNumber), verb];
  if (meaning.object !== null) words.push(nounPhrase(meaning.object, meaning.objectNumber));
  const sentence = words.join(" ");
  return sentence[0]!.toUpperCase() + sentence.slice(1);
}
