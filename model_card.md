# 🎧 Model Card: Applied AI Music Recommender

This model card extends the original *SongFinder 9000* card from
Module 3 to cover the AI components added during the applied-AI
extension: an LLM profile extractor, a faithfulness-checking critic,
KB retrieval, and a citation-grounded explainer. The Module 3
deterministic scorer is unchanged and still does the actual ranking.

## 1. System Name

SongFinder 9001

---

## 2. Intended Use

SongFinder 9001 helps a listener find songs that match a vibe they
describe in their own words. The listener answers three short
questions about what they're doing, what kind of sound they're
after, and any genres they want or want to avoid, or skips the
questions and just types a free-form description. The system reads
what they said and saves it as a named Vibe Profile they can come
back to. When the listener picks a profile, the system scores a
small catalog against it and recommends a handful of songs, with a
short paragraph per pick explaining why each song fits, grounded in
editorial notes about the songs and genres.

The profile and the recommendations are deliberately separate
artifacts. A listener might build one Vibe Profile for "late-night
studying" and another for "morning runs" and reuse either of them
across sessions without re-describing themselves every time.

It assumes the listener can describe their taste in natural language,
that the seven features in the catalog are enough to capture what
they want, and that a 30-track catalog is the universe of choices.

Not intended for production listening or any context where a bad
recommendation would matter. The catalog is curated, the scoring is
additive similarity, and the profile reduction is opinionated about
which features count.

---

## 3. How the Model Works

The system runs as two pipelines (see `assets/architecture.png`).

Every song has seven features: genre, mood, energy, tempo,
acousticness, valence (how positive the song sounds), and
danceability. The profile carries the same seven back, plus an
optional `avoid_genres` list. Mood is a label that either matches
or doesn't; a match earns `+2.0`. Genre is a label with three
tiers: exact match earns `+3.5`, an adjacent genre (per a fixed
map of related genres like rock ↔ acoustic, lofi ↔ ambient) earns
`+1.75`, no match earns zero. The five numeric features earn
partial credit by closeness, with tempo scaled by a 60 BPM range.
If a song's genre appears in the listener's `avoid_genres`, the
scorer short-circuits at the top with a hard zero before any
other math runs — avoid means avoid, not "score lower."  Every
song gets scored the same way; the top `k` come back with a short
reason breakdown.

The build pipeline takes three question-answers plus a free-form
description and produces a profile. A validation gate raises
`EmptyBuildInputsError` if every field is blank. The first API
call (the extractor prompt) sends the labeled bundle to the model
and parses the response into a structured profile plus a short
suggested name; it retries once on parse failure and falls back to
a default value with a warning for unknown genres or moods. A
second API call (the critic prompt) asks the model whether the
profile faithfully encodes the inputs and returns either `ok` or
specific corrections. The loop runs up to two rounds; if the model
still isn't satisfied at the cap, the result is flagged
`ambiguous_match` and the last candidate is kept. Parse errors and
API glitches degrade to `ok` so a misbehaving fact-check never
blocks the build.

The recommend pipeline ranks the catalog with the deterministic
scorer. For each top-`k` song, a lookup pulls three KB documents:
a genre overview, a mood overview, and a per-song editorial note.
A single batched API call writes short paragraphs per song,
required to quote 1–3 verbatim phrases. A substring check verifies
each citation actually appears in the docs the model was shown. If
a citation is fabricated, the prose drops out and only the
mechanical reasons remain.

Every LLM call goes through one `LLMClient` with a disk cache and
an offline stub. Tests use the stub; the real Gemini API is never
hit from CI or pytest.

---

## 4. Data

The catalog is 30 songs in `data/songs.csv`. The Module 3 starter
shipped with 10 tracks; I added 8 in Module 3 and another 12 during
the extension to bring every genre to at least 3 tracks. Ten genres
are represented (pop, lofi, indie pop, hip hop, electronic, acoustic,
rock, ambient, jazz, synthwave) at 3 tracks each. Eight moods cover
happy, chill, intense, melancholy, focused, moody, relaxed, and
upbeat. Each song has the same seven features as the Module 3
catalog: genre, mood, energy (0.0–1.0), tempo in BPM, valence
(0.0–1.0), danceability (0.0–1.0), and acousticness (0.0–1.0).

Plenty is missing. There's no classical, country, folk, metal, or
anything in a language other than English. No popularity, release
date, or artist metadata. Everything in the catalog is a curated
handful, not a real library. A listener whose taste lives outside
that handful gets a recommender that can't represent them.

The knowledge base under `src/kb/docs/` is 48 hand-authored
Markdown documents: 10 genre overviews, 8 mood overviews, and 30
per-song editorial notes. The genre docs describe typical
tempo/energy/mood ranges; the mood docs describe what listening
contexts the mood fits; the song notes are short editorial
paragraphs about each track. The explainer is required to quote
from these. They were drafted in batches and reviewed before they
landed.

Five preset profiles live in code as `PRESET_PROFILES` in
`src/profiles.py`: high_energy_pop, chill_lofi, deep_intense_rock,
chill_rock, boundary_maximalist. They match the Module 3 demo
profiles exactly. They also double as the recommend-eval inputs in
the reliability harness. Presets are read-only at runtime. User
profiles persist as JSON files under `applied-ai-systems/profiles/`
(gitignored) and can be edited field-by-field via the My Profiles
tab or the `profiles edit` CLI subcommand.

---

## 5. Strengths

The system separates clean taste clusters cleanly. High-Energy Pop,
Chill Lofi, and Deep Intense Rock all return top 5s that match what
I'd expect for those listeners, with little overlap between them.
This carries over from Module 3. The AI layer didn't change the
ranking math, just the way the listener gets a profile in and the
way the picks get explained.

Every recommendation comes with mechanical reasons (the point
breakdown) alongside the LLM explanation. The point breakdown is
always present, even when the LLM's explanation drops out due to a
fabricated citation. So a surprising pick is auditable without
guessing what the model thought.

The LLM never silently fabricates. The explainer prompt requires
verbatim quotes from the retrieved doc, and the substring check
verifies each citation actually exists. If the model paraphrased
instead of quoting, the explanation falls back to mechanical
reasons. This caught real fabrications during eval (about 4 of 10
cases) while still producing a useful output for every
recommendation.

AI features have explicit boundaries. The LLM can extract a profile
from natural language, but the saved profile is plain JSON with
clamped fields. The LLM can write an explanation, but the ranking
is unchanged. The LLM is consulted; it isn't the source of truth
for which song wins.

Genre adjacency handles near-misses gracefully. A listener asking
for "pop" who matches an indie pop song still gets partial credit
for the genre relationship instead of zero. This solves the
"strict label matching is brittle" weakness from the Module 3
card.

---

## 6. Limitations and Bias

- Avoidance is genre-only. `avoid_genres` is honored end-to-end
  (extractor populates it, scorer hard-zeros matching songs), but
  there's no equivalent field for moods, tempo, valence, or
  specific artists. A listener who wants "anything but slow songs"
  has no schema slot for that constraint.
- Genre adjacency is asymmetric. Rock counts acoustic as related,
  but acoustic does not count rock. A rock listener gets partial
  credit for acoustic songs; an acoustic listener does not get
  credit for rock.
- Underrepresented genres exist by design of the catalog. With 3
  tracks per genre, a listener whose taste lives in one genre
  doesn't get many real alternatives.
- Cosmetic: scores aren't normalized to `[0, 1]`. Cards show raw
  sums. Doesn't affect ranking; readability only.

The LLM components have their own failure modes:

- The extractor occasionally invents a fallback genre. If the
  listener says "drum and bass" (not in the catalog), the
  extractor falls back to the first allowed genre alphabetically
  and surfaces a warning. The listener gets a profile, but it
  doesn't reflect what they asked for.
- The critic over-refines on stress inputs. Even with the "default
  to ok" bias and per-field heuristics, contradictory inputs ("calm
  but high-energy") sometimes trigger refinement loops. The
  2-iteration cap and `ambiguous_match` flag bound the cost; the
  listener gets a best-effort profile and a flag indicating the
  system was uncertain.
- The explainer fabricates citations on hard cases. The
  whitespace-tolerant verbatim check cut fabrication from 8/10 to
  4/10 in evals. The remaining 4/10 fall back to mechanical
  reasons only and produce coherent output, but with less prose.
- Latency varies. Each LLM call takes 1–5 seconds; cached calls
  are immediate. The Streamlit UI shows a spinner; the CLI prints
  output incrementally.

There's also bias inherited from the model. Gemini's training data
is opinionated about what "calm", "upbeat", and genre-specific
descriptors mean. If the model's notion of "lofi" or "intense"
diverges from a particular listener's, the extractor will encode
the model's view, not the listener's. The system has no mechanism
to learn a specific listener's vocabulary over time.

---

## 7. Evaluation

I tested the system three ways: the Module 3 pair-by-pair
comparison (preserved from the original card), the new build-eval
(5 cases through the build pipeline), and the new recommend-eval
(5 cases through the recommend pipeline). The unit and integration
test suite is separate and lives in `tests/`.

### Pair-by-pair (preserved from Module 3)

The five preset profiles produce ten pairwise comparisons. One
observation per pair:

1. High-Energy Pop and Chill Lofi produce disjoint top fives with
   zero overlap, the clearest sign the system separates obvious
   extremes.
2. High-Energy Pop and Deep Intense Rock share only Gym Hero (pop
   genre with intense mood acts as the crossover); otherwise they
   pull from different clusters because happy and intense are
   different vibes.
3. High-Energy Pop and Chill Rock share no songs. They sit at
   opposite ends of every numeric axis and on different mood
   labels.
4. High-Energy Pop and Boundary Maximalist overlap heavily (Gym
   Hero, City Afterglow, Summer Polaroid appear on both) because
   both ask for high energy, valence, and danceability.
5. Chill Lofi and Deep Intense Rock have zero overlap. Chill Lofi
   clusters around energy `0.30-0.40` and tempo 70-80 BPM; Deep
   Intense Rock around `0.80-0.90` and 100-150 BPM.
6. Chill Lofi and Chill Rock share little. Chill Rock surfaces
   Lighthouse Hum and Static Waves; Chill Lofi stays in
   lofi/ambient/acoustic territory.
7. Chill Lofi and Boundary Maximalist are opposites on every axis
   with zero overlap, confirming the system handles diametrically
   opposed profiles correctly.
8. Deep Intense Rock and Chill Rock share rock tracks but in
   different orders. Deep Intense Rock leads with Static Waves and
   Storm Runner; Chill Rock leads with Lighthouse Hum, the
   catalog's chill-leaning rock cut.
9. Deep Intense Rock and Boundary Maximalist share four of the top
   5 (Gym Hero, Storm Runner, Backseat Freestyle, Broken Neon) in
   different orders because both want intense mood with high
   energy.
10. Chill Rock and Boundary Maximalist are opposites with zero
    overlap. Chill Rock pulls moody mid-tempo rock; Boundary
    Maximalist pulls loud, fast, electronic-leaning tracks at
    extreme numerics.

Why does Gym Hero keep showing up for both High-Energy Pop and
Boundary Maximalist? It's one of the loudest, most danceable, most
upbeat songs in the catalog, so anyone asking for high-energy
music with a positive feel gets it near the top regardless of
genre.

### Build-eval (5 cases through the build pipeline)

`run_build_eval` exercises the build pipeline against 5 fresh
listener personas in `src/eval/cases.py`. The personas deliberately
don't paraphrase the existing presets. They cover genres the
presets don't (jazz, hip hop, synthwave, acoustic, ambient) so
build-eval and recommend-eval test different territory. Four are
baseline cases (clear listener with all three questions plus a
description filled out). One is a stress case (`ambient_meditation`)
that pushes the limits (extreme low energy, 60 BPM, no percussion)
without being designed to fail.

Each case ships with a hand-authored expected `UserProfile` as
ground truth. The assertion is that the candidate profile lands in
that expected profile's neighborhood: same `favorite_genre`, same
`favorite_mood`, every numeric target within `0.20` (tempo within
30 BPM), and the candidate's `avoid_genres` set matches the
expected set exactly (case-insensitive). The `acoustic_morning`
persona explicitly tests the avoid path. Its inputs include
"definitely no electronic or synthwave" and the expected profile
carries `avoid_genres=["electronic", "synthwave"]`, so a faithful
extraction has to populate the list. The harness also surfaces
`ambiguous_match` and the count of extractor warnings per case so
each score is auditable.

### Recommend-eval (5 cases through the recommend pipeline)

`run_recommend_eval` runs `recommend()` against each of the five
presets directly. Two checks per case. First is the per-preset
structural rule:

- `high_energy_pop`: top 5 has at least 1 pop track and 0
  ambient/lofi.
- `chill_lofi`: top 5 has at least 1 track in {lofi, ambient,
  acoustic}.
- `deep_intense_rock`: top 5 has at least 1 rock track AND mean
  tempo at least 110 BPM.
- `chill_rock`: top 5 contains at least 1 rock track. Lighthouse
  Hum (the catalog's chill-leaning rock cut) is expected to
  surface.
- `boundary_maximalist`: top score positive AND
  bottom-of-top-5 at least 50% of top score AND at least 1
  high-energy (≥0.70) track.

Second is an LLM self-critique. A separate LLM call rates the top
5 against the profile on a 0–1 scale with a 0.6 pass threshold. The
prompt asks: do these top 5 reasonably reflect what someone with
this profile would want? Two scorecards print under one `--confirm`
quota gate; one JSON artifact carries both result arrays.

### Reliability findings

A few things I learned from running the harness:

- The critic over-refines on stress inputs even with the "default
  to ok" bias. Softened in `1833dba` (heuristics + bias toward ok)
  but not eliminated. The 2-iteration cap and `ambiguous_match`
  flag bound the cost.
- Explainer fabrication rate dropped from 8/10 to 4/10 after the
  whitespace-tolerant verbatim check landed. The remaining 4/10
  fall back to mechanical reasons only.
- `chill_rock` used to return 0 rock songs because the original
  `+1.5` genre bonus couldn't compete with a coherent calm
  cohort. Bumping genre weight and retuning the preset to match
  Lighthouse Hum's neighborhood fixed it; rock now surfaces in
  the top 5.
- `boundary_maximalist`'s original 5% score plateau is
  unreachable. Scoring is bimodal at extreme target values, so the
  structural rule was replaced with a weaker but achievable check
  (top-5 cohort coherence plus at least one high-energy track).

### Test suite

99 unit and integration tests across 10 files. All run offline
using `StubLLMClient`; no test makes real API calls. Coverage:
profile store and presets (including `avoid_genres` round-trip and
clear-with-empty-list), slug normalization, profile extractor
(happy path, retry, clamping, fallback warnings, `avoid_genres`
parsing with case-insensitive dedup, invalid-entry drops,
favorite/avoid contradiction guard, suggested-name sanitization,
prompt-includes-only-filled-fields), critic (faithfulness, refine
adjustments including list-typed `avoid_genres`, parse-failure
degradation, clamping, prompt-includes-bundle), end-to-end
`build_profile` (refinement applies avoid_genres adjustment) and
`recommend`, explainer, KB retriever, scorer (genre/mood weights
plus the avoid hard-zero short-circuit, case-insensitive),
eval-harness assertions (avoid_genres set-equality check fails on
missing or extra entries), LLM client (disk-cache hit/miss
accounting, retry-with-backoff on rate-limit errors), parsing.

Two invariants are checked by grep: the genai isolation
(`from google import genai` appears only in `src/llm/client.py`)
and the no-print-in-core rule (`grep -rn "print(" src/` shows
results only in presenter modules).

---

## 8. Future Work

A few things I'd change if I kept going.

Make genre adjacency bidirectional or learned. The map currently
inherits the KB's asymmetry: rock counts acoustic as related but
not vice versa. A real version would either repair the map by
hand or replace it with a similarity learned from the catalog.

Extend avoidance beyond genre. The schema honors `avoid_genres`
but the build form doesn't ask about avoided moods, tempo ranges,
or artists. A more honest "I don't want X" interface would let
the listener constrain any feature, not just the categorical one.

Normalize scores to `[0, 1]`. Cards currently show raw sums.
Cosmetic but would make the score breakdown easier to read.

Surface the extractor's per-field reasoning in the UI, not just
the final profile. Currently the build trace is in a debug pane;
a "why this number?" view per field would make the AI behavior
visible without requiring debug mode.

Track listener corrections over time. If a listener edits the
same field twice in a row to the same value, the system should
notice and adjust its extractor priors. Right now every build is
independent. There's no per-listener calibration.

Replace tag-based KB retrieval with embeddings. The current
retriever is exact id-and-name lookup, which is fast and
deterministic but inflexible. Semantic retrieval would let the
explainer ground claims in related-but-not-identical-tag docs
when the exact-match doc is thin.

---

## 9. Ethical Considerations

The system is small and local. No user data leaves the machine
except the LLM API calls, which carry only the listener's stated
preferences (no PII unless the listener types one in). Profiles
are stored as plain JSON in a gitignored directory. The project
doesn't collect or transmit usage analytics.

Three issues worth naming:

Underrepresented genres. With only 3 tracks per genre, a listener
whose taste lives outside the curated handful gets a recommender
that can't represent them. The math papers over the data gap by
returning a numerically-close song from a different genre. This is
a real bias in the artifact, not in the algorithm. A real catalog
would still need a coverage check before recommending.

Avoidance is honored only at the genre level. The schema and
scorer support `avoid_genres`, so a listener who says "anything
but country" gets country songs hard-zeroed out of the top-`k`.
But a listener who wants "no slow songs" or "anything but
acoustic-style production" has no schema slot for those
constraints. The system reads them as silence rather than as
avoidance. A more honest avoidance interface would cover every
feature in the profile, not just genre.

Inherited LLM bias. The extractor's interpretation of "calm",
"upbeat", and genre-specific descriptors comes from Gemini's
training data. A listener whose vocabulary diverges from the
model's gets a profile that encodes the model's view, not theirs.
The system has no per-user adaptation.

The system isn't appropriate for any context where a bad
recommendation has real consequences (mood-sensitive listening,
medical or therapeutic use, contexts where excluding a genre
carries weight). The catalog is a demo, the scoring is
opinionated, and the LLM components are best-effort.

---

## 10. AI Collaboration during Development

I built this project as a pair with Claude Code. The collaboration
worked best when I came in with a specific shape in mind ("split
`run_pipeline` into `build_profile` + `recommend` with these
signatures") rather than asking for open-ended planning. With a
target in hand it scaffolded fast (the two-pipeline rewrite, the
eval harness structure, the JSON-fence parsing) and it caught real
bugs in code review before I committed.

Two specific examples worth naming. The most useful AI suggestion
was the fix for Streamlit's cross-tab state bug. The assistant
recognized that a widget's `session_state` key can only be set
before the widget exists in a given script run, and proposed
writing to `pending_*` slots and consuming them on the receiving
tab. Unit tests had passed; I would have shipped the bug if I'd
only run pytest and never opened the actual UI.

The most flawed suggestion came at the start. My notes said
"free-form natural language for profile entry" (meaning a
deliberate AI-assisted step that yields a persistent profile), and
the assistant misread that as the system's input shape, producing
a single `run_pipeline(nl) -> result` that extracted, scored,
retrieved, explained, and discarded the profile in one call.
Persistence wasn't just missing, it wasn't even on the table.
Tests passed because no test asserted profiles survived a process
boundary. I caught the mistake only because the resulting
Streamlit flow felt wrong to use. Every interaction started from
scratch even though the system clearly should remember. The lesson:
model confidence is independent of whether it read my intent
right, and "feels wrong to use" is a signal worth trusting over
green tests.

The README's Reflection section has the full development story
(testing surprises, additional limitations, future improvements)
and the Portfolio Reflection summarizing the design principle
that ran through the whole build.
