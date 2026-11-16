import json

import pytest

from conftest import REPO
from data import DataError, load_examples, to_hf_dataset, write_drop_log


def jsonl(tmp_path, lines):
    p = tmp_path / "d.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return p


def test_trial_dataset_loads():
    rows, report = load_examples(REPO / "data" / "trial.jsonl")
    assert len(rows) == 16 and not report.dropped
    assert rows[0]["prompt"] == [{"role": "user", "content": "What is 7 * 6? Answer with just the number."}]
    assert rows[0]["answer"] == "42"


def test_malformed_rows_are_reported_not_silently_dropped(tmp_path):
    p = jsonl(tmp_path, [
        json.dumps({"prompt": "ok", "answer": 5, "family": "x"}),
        "{not json",
        json.dumps({"prompt": "", "answer": "1"}),
        json.dumps({"answer": "1"}),
        json.dumps({"prompt": "no answer"}),
        json.dumps({"prompt": "blank answer", "answer": "  "}),
        json.dumps(["a", "list"]),
        "",
    ])
    rows, report = load_examples(p)
    assert len(rows) == 1 and rows[0]["answer"] == "5" and rows[0]["family"] == "x"
    reasons = [d["reason"].split(":")[0] for d in report.dropped]
    assert reasons == ["invalid_json", "empty_prompt", "missing_prompt", "missing_answer", "empty_answer", "not_an_object"]
    assert [d["line"] for d in report.dropped] == [2, 3, 4, 5, 6, 7]
    out = tmp_path / "dropped.jsonl"
    write_drop_log(report, out)
    assert len(out.read_text().splitlines()) == 6


def test_all_invalid_raises(tmp_path):
    with pytest.raises(DataError, match="no valid examples"):
        load_examples(jsonl(tmp_path, [json.dumps({"prompt": ""})]))


def test_missing_and_empty_files(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_examples(tmp_path / "nope.jsonl")
    (tmp_path / "empty.jsonl").write_text("\n\n")
    with pytest.raises(DataError, match="empty"):
        load_examples(tmp_path / "empty.jsonl")


def test_custom_fields_system_prompt_and_max_samples(tmp_path):
    p = jsonl(tmp_path, [json.dumps({"question": f"q{i}", "target": i}) for i in range(5)])
    rows, _ = load_examples(p, prompt_field="question", answer_field="target", system_prompt="Be brief.", max_samples=3)
    assert len(rows) == 3
    assert rows[0]["prompt"][0] == {"role": "system", "content": "Be brief."}
    assert "question" not in rows[0] and rows[2]["answer"] == "2"


def test_prompt_length_filter_uses_chat_template(tmp_path, tiny_tokenizer):
    p = jsonl(tmp_path, [json.dumps({"prompt": "short", "answer": "1"}), json.dumps({"prompt": "word " * 300, "answer": "1"})])
    rows, report = load_examples(p, tokenizer=tiny_tokenizer, max_prompt_length=200)
    assert len(rows) == 1
    assert report.dropped[0]["reason"].startswith("prompt_too_long")


def test_inconsistent_extra_columns_are_filled(tmp_path):
    p = jsonl(tmp_path, [json.dumps({"prompt": "a", "answer": "1", "family": "f"}), json.dumps({"prompt": "b", "answer": "2"})])
    rows, _ = load_examples(p)
    ds = to_hf_dataset(rows)
    assert ds.column_names == ["prompt", "answer", "family"] and ds[1]["family"] is None
