"""Prompt templates for LLM-driven agents.

Populated incrementally:
- EXPLAINER_PROMPT (Step 1.5)
- PROFILE_EXTRACTOR_PROMPT (Step 2.1)
- CRITIC_PROMPT (Step 2.2)
- EVAL_SELF_CRITIQUE_PROMPT (Step 3.3)
"""


PROFILE_EXTRACTOR_PROMPT = """\
You translate a listener's answers about their music taste into a structured
listener profile for a deterministic recommender. The listener has answered
some or all of five guided questions and may have written a free-form
description. Read what they provided below and output ONLY a JSON object
matching this schema, with no surrounding prose and no markdown fences.

Schema:
  {{
    "favorite_genre": "<one of: {allowed_genres}>",
    "favorite_mood":  "<one of: {allowed_moods}>",
    "target_energy":        <float in [0.0, 1.0]>,
    "target_tempo_bpm":     <float in [40.0, 220.0]>,
    "target_valence":       <float in [0.0, 1.0]>,
    "target_danceability":  <float in [0.0, 1.0]>,
    "target_acousticness":  <float in [0.0, 1.0]>
  }}

Rules:
- favorite_genre and favorite_mood MUST be exact lowercase values from the
  allowed lists above. If the listener mentions a genre or mood that is not
  in the list, pick the nearest allowed value.
- For any numeric field the listener does not explicitly imply, use a
  neutral default: energy 0.5, tempo 100, valence 0.5, danceability 0.5,
  acousticness 0.4.
- The listener may have answered only some of the five questions; treat
  silence on a field as "no preference" rather than as a signal.
- Map descriptive language to numbers consistently:
    "slow" -> tempo around 65-80, "mid-tempo" -> 95-110, "fast"/"upbeat" -> 120-140.
    "mellow"/"calm" -> energy 0.25-0.4, "driving"/"hype" -> energy 0.8+.
    "sad" -> valence 0.15-0.35, "neutral" -> 0.5, "happy"/"bright" -> 0.75+.
    "acoustic"/"warm" -> acousticness 0.7+, "synthetic"/"electronic" -> 0.1-0.2.

{starting_from_block}Listener inputs (only the fields the listener filled are shown):
{inputs_block}

Output the JSON object only.
"""


EVAL_SELF_CRITIQUE_PROMPT = """\
You are evaluating whether a music recommender's top-5 results
reasonably reflect what someone with this listener profile would want.
The profile carries a favorite genre and mood plus five [0,1]-style
numeric targets (energy, valence, danceability, acousticness) and a
tempo target. Read the profile and the top-5 below. Decide:

- Do these top-5 reasonably reflect what someone with this profile
  would want, across genre, mood, energy, and tempo? It does NOT need
  to be perfect — small mismatches are acceptable.

Output ONLY a JSON object of this shape, with no surrounding prose and no
markdown fences:

  {{
    "score": <float in [0.0, 1.0]>,
    "pass":  <true | false>,
    "reason": "<1-2 short sentences>"
  }}

Scoring rubric:
- 1.0: top-5 strongly reflects every aspect of the profile.
- 0.7-0.9: most aspects match; one minor mismatch.
- 0.4-0.6: one or two material mismatches.
- 0.0-0.3: top-5 ignores or contradicts the profile.

"pass" should be true when score >= 0.6.

Listener profile:
{profile_block}

Top-5 (id | title | artist | genre | mood | score):
{top5_block}

Output the JSON object only.
"""


CRITIC_PROMPT = """\
You evaluate whether a candidate listener profile faithfully reflects a
listener's stated preferences. The listener answered some or all of five
guided questions and may have written a free-form description; an extractor
produced a 7-dimension profile from that bundle. Your job is to check
whether the profile's field values encode what the listener said.

Decide one of two verdicts:

- "ok": every field in the candidate profile is a faithful encoding of
  what the listener said (or a defensible default for fields the listener
  did not address).
- "refine": at least one field clearly diverges from what the listener
  said, AND you can name the specific corrected value that would restore
  faithfulness.

DEFAULT TO "ok". Faithfulness means matching the listener's *stated*
preferences, not improving on them. If the listener was vague about a
field, the extractor's default is fine. If two reasonable encodings exist
for what the listener said, the extractor's choice is fine. Only refine
when the candidate is plainly wrong — "the listener said 'around 90 BPM'
but the candidate has target_tempo_bpm=120" — and you have the corrected
value to substitute.

Concrete guidance:
- If the listener said "mellow", "calm", "low-energy", target_energy
  should be in [0.20, 0.40]. Outside that range -> refine with a value
  inside it.
- If the listener said "fast", "driving", "upbeat", target_energy should
  be in [0.70, 0.95].
- If the listener named a tempo number, target_tempo_bpm should be within
  ~10 BPM of it.
- If the listener named a genre present in the catalog, favorite_genre
  should match it.
- If the listener was silent on a field (e.g. did not mention valence),
  the extractor's chosen value is faithful by default — do not refine.
- "ok" is the right verdict when the candidate is *good enough*; reserve
  "refine" for plain mismatches with a specific fix.

Output ONLY a JSON object of this shape, with no surrounding prose and no
markdown fences:

  {{
    "verdict": "ok" | "refine",
    "adjustments": null | {{
        "favorite_genre": "<allowed genre>",
        "favorite_mood":  "<allowed mood>",
        "target_energy":        <float in [0.0, 1.0]>,
        "target_tempo_bpm":     <float in [40.0, 220.0]>,
        "target_valence":       <float in [0.0, 1.0]>,
        "target_danceability":  <float in [0.0, 1.0]>,
        "target_acousticness":  <float in [0.0, 1.0]>
    }},
    "reason": "<1-2 short sentences>"
  }}

Rules:
- adjustments MUST be null when verdict is "ok".
- adjustments MUST be a non-empty object when verdict is "refine"; include
  ONLY the fields that need correcting, with their corrected absolute
  values. You MAY NOT add or modify any keys outside the seven above.
- Numeric values are absolute targets, not deltas.

Listener inputs (only the fields the listener filled are shown):
{inputs_block}

Candidate profile (the extractor's current output):
{profile_block}

Output the JSON object only.
"""



EXPLAINER_PROMPT = """\
You are a music recommender's explanation writer. The user has given you a
listener profile and five candidate songs that a deterministic scorer ranked
as the best matches. For each song you have three reference snippets: a genre
overview, a mood overview, and a per-track editorial note.

Write one short paragraph (2-4 sentences) per song explaining why it fits
this listener. Ground every claim in the reference snippets — quote concrete
details from them rather than inventing facts.

Listener profile:
{profile_block}

Candidates and reference snippets:
{candidates_block}

Return ONLY a JSON object of the following shape, with one entry per song,
in the same order as the candidates above.

{{
  "explanations": [
    {{
      "song_id": <int>,
      "text": "<paragraph>",
      "cited_snippets": ["<verbatim phrase>", ...]
    }},
    ...
  ]
}}

CRITICAL RULES FOR cited_snippets:

1. Each entry MUST be an EXACT character-by-character copy of a continuous
   substring from one of that song's three reference snippets above.
2. Do NOT shorten, abbreviate, paraphrase, or fix grammar. Do not collapse
   "between 60 and 90 BPM" to "60-90 BPM". Do not change "low-energy
   listening state" to "low energy". Copy the exact characters.
3. Choose phrases of 15-120 characters, matching original punctuation,
   capitalisation, and spacing. Hyphens, en-dashes, and em-dashes must
   match the original character.
4. Provide 1-3 such phrases per song. If you cannot find at least one
   phrase you can copy verbatim, that is a bug — re-read the snippets and
   pick a different phrase.

Output only the JSON object. Do not wrap it in markdown fences. Do not add
prose outside the JSON.
"""
