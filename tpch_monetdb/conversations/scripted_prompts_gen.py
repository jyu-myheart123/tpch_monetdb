from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Mapping

_SCRIPTED_PROMPTS_ROOT_DIR = Path(__file__).parent / "prompts" / "scripted"


def get_scripted_prompts_root() -> Path:
    """Return the root directory for scripted prompt assets."""
    return _SCRIPTED_PROMPTS_ROOT_DIR


def list_scripted_prompt_assets() -> tuple[Path, ...]:
    """List all scripted prompt assets in a stable order."""
    return tuple(sorted(_SCRIPTED_PROMPTS_ROOT_DIR.rglob("*.txt")))


def load_scripted_prompt_asset(*relative_parts: str) -> str:
    """Load a scripted prompt asset and fail if the file is missing."""
    asset_path = _SCRIPTED_PROMPTS_ROOT_DIR.joinpath(*relative_parts)
    if not asset_path.exists():
        raise FileNotFoundError(f"Missing scripted prompt asset: {asset_path}")
    text = asset_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Scripted prompt asset is empty: {asset_path}")
    return text


def render_scripted_prompt_asset(
    *relative_parts: str,
    variables: Mapping[str, object],
) -> str:
    """Render a scripted prompt asset with strict placeholder substitution."""
    template = Template(load_scripted_prompt_asset(*relative_parts))
    try:
        return template.substitute({key: str(value) for key, value in variables.items()})
    except KeyError as exc:
        raise ValueError(
            f"Missing placeholder for scripted prompt asset {'/'.join(relative_parts)}: {exc.args[0]}"
        ) from exc
