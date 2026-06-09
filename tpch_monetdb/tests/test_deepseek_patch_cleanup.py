from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TPCH_MONETDB_ROOT = ROOT / "tpch_monetdb"


def _production_python_files() -> list[Path]:
    files: list[Path] = []
    for path in TPCH_MONETDB_ROOT.rglob("*.py"):
        rel = path.relative_to(TPCH_MONETDB_ROOT)
        if rel.parts[0] == "tests":
            continue
        files.append(path)
    return sorted(files)


def test_production_deepseek_replay_mentions_are_limited_to_main_tpch_monetdb() -> None:
    matches: list[str] = []
    for path in _production_python_files():
        if "reasoning_content" in path.read_text(encoding="utf-8"):
            matches.append(path.relative_to(TPCH_MONETDB_ROOT).as_posix())

    assert matches == [
        "llm_cache/cached_litellm.py",
        "llm_cache/deepseek_reasoning_replay.py",
        "main_tpch_monetdb.py",
    ]
    return None


def test_production_code_does_not_reference_provider_internal_reasoning_fields() -> None:
    offenders: list[str] = []
    for path in _production_python_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            uses_provider_specific_reasoning = (
                "provider_specific_fields" in line and "reasoning_content" in line
            )
            uses_provider_data_reasoning = (
                "provider_data" in line and "reasoning_content" in line
            )
            if uses_provider_specific_reasoning or uses_provider_data_reasoning:
                offenders.append(path.relative_to(TPCH_MONETDB_ROOT).as_posix())
                break

    assert offenders == [
        "llm_cache/deepseek_reasoning_replay.py",
    ]
    return None
