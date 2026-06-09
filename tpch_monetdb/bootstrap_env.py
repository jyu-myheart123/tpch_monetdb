"""Bootstrap TPC-H MonetDB runtime environment before any LiteLLM import."""

import os
from pathlib import Path

from dotenv import load_dotenv


def bootstrap_runtime_env() -> None:
    """Load tpch_monetdb/.env and force LiteLLM to use the local model cost map."""
    tpch_monetdb_root = Path(__file__).resolve().parent
    env_path = tpch_monetdb_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    return None
