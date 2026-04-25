"""Prompt templates for LLM-driven agents.

Populated incrementally:
- EXPLAINER_PROMPT (Step 1.5)
- PROFILE_EXTRACTOR_PROMPT (Step 2.1)
- CRITIC_PROMPT (Step 2.2)
- EVAL_SELF_CRITIQUE_PROMPT (Step 3.3)
"""


EXPLAINER_PROMPT = """\
You are a music recommender's explanation writer. The user has given you a
listener profile and five candidate songs that a deterministic scorer ranked
as the best matches. For each song you have three reference snippets: a genre
overview, a mood overview, and a per-track editorial note.

Write one short paragraph (2-4 sentences) per song explaining why it fits
this listener. Ground every claim in the reference snippets — quote concrete
details from them rather than inventing facts. Mention specific numeric or
descriptive details from the snippets at least once per song.

Listener profile:
{profile_block}

Candidates and reference snippets:
{candidates_block}

Return ONLY a JSON object of the following shape, with one entry per song,
in the same order as the candidates above. Each `cited_snippets` array must
contain 1-3 short verbatim phrases (10-200 characters each) copied from any
of that song's three reference snippets — no paraphrasing, no invention.

{{
  "explanations": [
    {{
      "song_id": <int>,
      "text": "<paragraph>",
      "cited_snippets": ["<verbatim snippet>", ...]
    }},
    ...
  ]
}}

Output only the JSON object. Do not wrap it in markdown fences. Do not add
prose outside the JSON.
"""
