"""Tests for scripts/check_data_files.py."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "check_data_files", Path(__file__).parents[1] / "scripts" / "check_data_files.py"
)
check_data_files = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(check_data_files)

CONFIG, MATTER = check_data_files.yaml_configs()


def test_valid_json_passes():
    assert check_data_files.json_problems("a.json", b'{"a": [1, 2]}\n', lines=False) == []


def test_json_duplicate_key_fails():
    problems = check_data_files.json_problems("a.json", b'{"a": 1, "a": 2}\n', lines=False)
    assert len(problems) == 1
    assert "duplicate key 'a'" in problems[0]


def test_json_syntax_error_and_missing_newline_fail():
    problems = check_data_files.json_problems("a.json", b'{"a": 1,}', lines=False)
    assert problems[0] == "a.json: no newline at the end"
    assert len(problems) == 2


def test_json_not_utf8_fails():
    problems = check_data_files.json_problems("a.json", b'"\xff"\n', lines=False)
    assert problems[0].startswith("a.json: not UTF-8")


def test_jsonl_reports_line_numbers():
    raw = b'{"a": 1}\n\n{"b": \n'
    problems = check_data_files.json_problems("a.jsonl", raw, lines=True)
    assert problems[0] == "a.jsonl:2: blank line"
    assert problems[1].startswith("a.jsonl:3: ")
    assert len(problems) == 2


def test_jsonl_whole_file_as_one_value_fails():
    assert check_data_files.json_problems("a.jsonl", b'{\n"a": 1\n}\n', lines=True)


def test_yaml_workflow_key_on_passes():
    text = "name: CI\non:\n  push:\n    branches: [main]\n"
    assert check_data_files.yaml_problems("ci.yml", text, CONFIG) == []


def test_yaml_duplicate_key_fails_with_offset_line():
    problems = check_data_files.yaml_problems("a.md", "name: a\nname: b\n", CONFIG, first_line=2)
    assert len(problems) == 1
    assert problems[0].startswith("a.md:3:")
    assert "key-duplicates" in problems[0]


def test_yaml_long_line_fails_unless_unbreakable():
    long_words = "key: " + "word " * 30 + "\n"
    long_url = "url: https://example.com/" + "a" * 120 + "\n"
    assert check_data_files.yaml_problems("a.yml", long_words, CONFIG)
    assert check_data_files.yaml_problems("a.yml", long_url, CONFIG) == []


def test_front_matter_has_no_line_limit(tmp_path, monkeypatch):
    page = tmp_path / "SKILL.md"
    page.write_text("---\nname: x\ndescription: " + "word " * 40 + "end\n---\n\n# Title\n")
    monkeypatch.setattr(check_data_files, "REPO", tmp_path)
    assert check_data_files.file_problems("SKILL.md", CONFIG, MATTER) == []


def test_repository_passes():
    assert check_data_files.main() == 0
