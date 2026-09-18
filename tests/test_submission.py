"""Тесты проверки answer.csv: ловим и громкие ошибки формата, и тихие (регистр, чужие id)."""

from src.submission import validate_answer, write_answer

QIDS = {"00WuFMaXSFZBxSzT", "03ztb1gtRFC4K4vP"}
CORPUS = {"1382564bf8994a83", "121fa7f5e765ce00", "f3801ec4fa472597"}


def _write(tmp_path, text):
    p = tmp_path / "answer.csv"
    p.write_text(text, encoding="utf-8")
    return p


def test_written_answer_is_valid(tmp_path):
    p = tmp_path / "answer.csv"
    write_answer(
        {
            "00WuFMaXSFZBxSzT": ["1382564bf8994a83", "1382564bf8994a83", "121fa7f5e765ce00"],
            "03ztb1gtRFC4K4vP": ["f3801ec4fa472597"],
        },
        p,
    )
    assert validate_answer(p, QIDS, CORPUS) == []
    assert "1382564bf8994a83 121fa7f5e765ce00" in p.read_text()


def test_silent_errors_are_caught(tmp_path):
    p = _write(
        tmp_path,
        "query_id,answer\n00WuFMaXSFZBxSzT,1382564BF8994A83 deadbeefdeadbeef\n03ztb1gtRFC4K4vP,f3801ec4fa472597\n",
    )
    errors = validate_answer(p, QIDS, CORPUS)
    assert any("нижнем регистре" in e for e in errors)
    assert any("нет в корпусе" in e for e in errors)


def test_structure_errors_are_caught(tmp_path):
    p = _write(tmp_path, ",query_id,answer\n0,00WuFMaXSFZBxSzT,1382564bf8994a83\n")
    assert validate_answer(p, QIDS, CORPUS)
    p = _write(tmp_path, "query_id,answer\n00WuFMaXSFZBxSzT,1382564bf8994a83 1382564bf8994a83\n")
    errors = validate_answer(p, QIDS, CORPUS)
    assert any("повторы" in e for e in errors)
    assert any("нет строк" in e for e in errors)
