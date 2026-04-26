"""Prompt templates for LLM-driven agents.

Populated incrementally:
- EXPLAINER_PROMPT (Step 1.5)
- PROFILE_EXTRACTOR_PROMPT (Step 2.1)
- CRITIC_PROMPT (Step 2.2)
- EVAL_SELF_CRITIQUE_PROMPT (Step 3.3)
"""


PROFILE_EXTRACTOR_PROMPT = """\
You translate a listener's plain-English music request into a structured
listener profile for a deterministic recommender. Read the request below and
output ONLY a JSON object matching this schema, with no surrounding prose
and no markdown fences.

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
- Map descriptive language to numbers consistently:
    "slow" -> tempo around 65-80, "mid-tempo" -> 95-110, "fast"/"upbeat" -> 120-140.
    "mellow"/"calm" -> energy 0.25-0.4, "driving"/"hype" -> energy 0.8+.
    "sad" -> valence 0.15-0.35, "neutral" -> 0.5, "happy"/"bright" -> 0.75+.
    "acoustic"/"warm" -> acousticness 0.7+, "synthetic"/"electronic" -> 0.1-0.2.

Listener request:
{nl_input}

Output the JSON object only.
"""


EVAL_SELF_CRITIQUE_PROMPT = """\
You are evaluating whether a music recommender's top-5 results match the
listener's stated request. Read the request and the top-5 below. Decide:

- Does the top-5 reasonably reflect the listener's stated intent across
  genre, mood, energy, and tempo? It does NOT need to be perfect — small
  mismatches are acceptable.

Output ONLY a JSON object of this shape, with no surrounding prose and no
markdown fences:

  {{
    "score": <float in [0.0, 1.0]>,
    "pass":  <true | false>,
    "reason": "<1-2 short sentences>"
  }}

Scoring rubric:
- 1.0: top-5 strongly reflects every aspect of the request.
- 0.7-0.9: most aspects match; one minor mismatch.
- 0.4-0.6: one or two material mismatches.
- 0.0-0.3: top-5 ignores or contradicts the request.

"pass" should be true when score >= 0.6.

Listener request:
{nl_input}

Top-5 (id | title | artist | genre | mood | score):
{top5_block}

Output the JSON object only.
"""


CRITIC_PROMPT = """\
You evaluate whether a music recommender's top-5 results match a listener's
stated request. Read the request, the current listener profile, and the
top-5 the system produced. Decide one of two verdicts:

- "ok": the top-5 reasonably reflects the request. No adjustments needed.
- "refine": the top-5 misses an important aspect of the request, AND you
  can name a specific numeric or categorical adjustment that would
  meaningfully move the next iteration closer to intent.

DEFAULT TO "ok". The listener gets faster, more stable results when the
critic is willing to accept "good enough." Only escalate to "refine" when
you can clearly articulate both (a) what specifically is wrong AND (b) the
exact adjustment that would fix it. If the request is vague,
contradictory, or already satisfied within reason, return "ok".

Concrete guidance:
- If 3 or more of the top-5 already match the listener's stated genre or
  mood, return "ok" — that is a successful match.
- If the request is contradictory ("calm but high-energy") or extremely
  vague ("something I can think to"), return "ok" — there is no right
  adjustment to make.
- If the request specifies a value the recommender already produces
  (e.g. "around 90 BPM" and the top-5 averages 88-95 BPM), return "ok".
- Only return "refine" when you can finish the sentence: "The top-5 is
  wrong because <X>, and setting <field> to <value> would fix it."

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
- adjustments MUST be a non-empty object when verdict is "refine"; only
  include keys that need overriding. You MAY NOT add or modify any keys
  outside the seven listed above.
- Numeric values are absolute targets, not deltas.

Listener request:
{nl_input}

Current listener profile:
{profile_block}

Current top-5 (id | title | artist | genre | mood | score):
{top5_block}

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
