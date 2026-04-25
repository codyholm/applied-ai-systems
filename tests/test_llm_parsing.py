from src.llm.parsing import strip_json_fences


def test_strip_json_fences_no_fence_passes_through():
    assert strip_json_fences('{"a": 1}') == '{"a": 1}'


def test_strip_json_fences_handles_json_label():
    raw = '```json\n{"a": 1}\n```'
    assert strip_json_fences(raw) == '{"a": 1}'


def test_strip_json_fences_handles_bare_fence():
    raw = '```\n{"a": 1}\n```'
    assert strip_json_fences(raw) == '{"a": 1}'


def test_strip_json_fences_strips_surrounding_whitespace():
    raw = '   \n```json\n{"a": 1}\n```\n  '
    assert strip_json_fences(raw) == '{"a": 1}'
