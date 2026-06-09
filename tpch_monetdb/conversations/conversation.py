import asyncio
import json
import logging
import os
from abc import abstractmethod
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.filters import is_multiline
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

from tpch_monetdb.llm_cache import send_notification
from tpch_monetdb.llm_cache.cached_openai import CachedOpenAIResponsesModel
from tpch_monetdb.llm_cache.utils import atomic_write, create_parent_and_set_permissions

COMPACTION_MARKER = "<<COMPACTION>>"
VALIDATE_ON = "<<VALIDATE_ON>>"
VALIDATE_OFF = "<<VALIDATE_OFF>>"
VALIDATE_OUTPUT_STDOUT_ON = "<<VALIDATE_OUTPUT_STDOUT_ON>>"
VALIDATE_OUTPUT_STDOUT_OFF = "<<VALIDATE_OUTPUT_STDOUT_OFF>>"
NOTIFY_AFTER_SEC = 60

# Display labels for each choice key (order is preserved in the prompt).
_CHOICE_LABELS: dict[str, str] = {
    "u": "<b>[u]</b>se",
    "r": "<b>[r]</b>eplace",
    "i": "<b>[i]</b>nsert before",
    "c": "<b>[c]</b>ompaction",
}

logger = logging.getLogger(__name__)

_PROMPT_METADATA_SCALAR_FIELDS: tuple[str, ...] = (
    "tool_profile",
    "rule_area",
    "active_unit_id",
    "active_unit_kind",
    "patch_scope_verdict",
)
_PROMPT_METADATA_SEQUENCE_FIELDS: tuple[str, ...] = (
    "active_query_ids",
    "active_unit_files",
    "active_unit_query_ids",
    "objective_ids",
    "data_law_ids",
    "required_control_artifacts",
    "control_artifacts_injected",
)


def normalize_prompt_metadata(
    prompt_metadata: Optional[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Normalize prompt metadata so callback consumers receive stable types."""
    if prompt_metadata is None:
        return None
    normalized: dict[str, Any] = dict(prompt_metadata)
    for field_name in _PROMPT_METADATA_SCALAR_FIELDS:
        value = normalized.get(field_name)
        if value is not None:
            normalized[field_name] = str(value)
    for field_name in _PROMPT_METADATA_SEQUENCE_FIELDS:
        value = normalized.get(field_name)
        if isinstance(value, (list, tuple)):
            normalized[field_name] = [str(item) for item in value]
    return normalized


class AbstractConversation:
    def __init__(
        self,
        conversation_json_path: Path,  # where to persist the conversation (list of accepted prompts)
        callback: Callable[
            [str, Optional[str], int, Optional[int], Optional[dict[str, Any]]],
            Any,
        ],
        replay: bool = False,
        notify: bool = False,
        auto_finish: bool = False,
        allowed_choices: Tuple[str, ...] = (
            "u",
            "r",
            "i",
            "c",
        ),  # use, replace, insert-before, compaction
        model: Optional[CachedOpenAIResponsesModel] = None,
        auto_u: bool = False,
        replay_cache: bool = False,
        workspace_root: Optional[Path] = None,
    ):
        self.conversation_json_path = conversation_json_path
        self.callback = callback
        self.replay = replay
        self.notify = notify
        self.auto_finish = auto_finish
        self.allowed_choices = allowed_choices
        self.workspace_root = workspace_root
        self.production_auto_mode = bool(auto_u)
        self._validate_production_choice_policy()

        # create cache dir if not existing
        create_parent_and_set_permissions(self.conversation_json_path)

        # create auto mode callbacks
        if auto_u:
            logger.warning(
                "Auto-U mode enabled: automatically proceeding with all prompts without asking for user confirmation. Make sure this is what you want!"
            )
            assert not replay_cache, "auto_u and replay_cache cannot both be enabled"
            assert "u" in allowed_choices, (
                "auto_u requires 'u' to be in allowed_choices"
            )
            self.get_choice = lambda: "u"
        elif replay_cache:
            # auto-approve if last LLM response was cached, otherwise ask user (same as auto_u but only for cached responses - executes only the cached prompts and the first non-cached prompt, then stops and waits for user input for the rest)
            assert model is not None, (
                "model must be provided when replay_cache is enabled"
            )
            self.get_choice = lambda: "u" if model.llm_was_cached else None
        else:
            self.get_choice = None

        self._session = self._create_session()

        # for type hinting clarity - will be initialized in run()
        self.used: List[str] = None  # type: ignore
        return None

    @abstractmethod
    async def run(self) -> Optional[List[str]]:
        pass

    # ---------- interaction ----------

    async def process_prompt(
        self,
        prompt: str,
        prompt_descriptor: Optional[
            str
        ] = None,  # short description of the prompt, used for logging and callbacks
        max_turns: Optional[int] = None,
        additional_out_str: Optional[str] = None,
        prompt_metadata: Optional[dict[str, Any]] = None,
    ) -> Tuple[str, str, Optional[str]]:
        """
        Handle one interaction round for `prompt`.

        Resolves the user choice by consulting `self.get_choice` first (set by
        auto_u / replay_cache modes), then falling back to interactive input.
        Executes the chosen action, appends to `used`, and persists via `_save`.

        Returns
        -------
        ("advance", last_output)  – caller should move to the next prompt
        ("stay",    last_output)  – caller should re-show the same prompt
                                    (insert-before and compaction cases)
        """

        # Show the prompt before asking for the choice, so user can see what they're acting on while deciding.
        self._show_prompt(prompt, additional_out_str)

        choice = self.get_choice() if self.get_choice else None
        if choice is None:
            choice = await self._ask_choice(prompt)

        assert self.used is not None, (
            "self.used should have been initialized in run() by children class by now"
        )

        last_output = None
        callback_metadata = normalize_prompt_metadata(prompt_metadata)
        if choice == "u":
            self.used.append(prompt)
            last_output = await self._maybe_await_callback(
                prompt,
                prompt_descriptor,
                len(self.used) - 1,
                max_turns,
                callback_metadata,
            )

        elif choice == "r":
            new_prompt = await self._ask_multiline("Replacement (Ctrl+D to submit)")
            if new_prompt.strip():
                self.used.append(new_prompt)
                last_output = await self._maybe_await_callback(
                    new_prompt,
                    new_prompt[:20],
                    len(self.used) - 1,
                    max_turns,
                    callback_metadata,
                )

        elif choice == "i":
            new_prompt = await self._ask_multiline(
                "Insert before (Ctrl+D to submit)",
            )
            if new_prompt.strip():
                self.used.append(new_prompt)
                self._save(self.used)  # save progress before the callback
                last_output = await self._maybe_await_callback(
                    new_prompt,
                    new_prompt[:20],
                    len(self.used) - 1,
                    max_turns,
                    callback_metadata,
                )

        elif choice == "c":
            self.used.append(COMPACTION_MARKER)
            self._save(self.used)  # save progress before the callback
            last_output = await self._maybe_await_callback(
                COMPACTION_MARKER,
                "compaction",
                len(self.used) - 1,
                max_turns,
                callback_metadata,
            )

        else:
            raise ValueError(f"Unexpected choice: {choice!r}")

        # Save progress after each accepted prompt.
        self._save(self.used)

        # return choice, last prompt, last output
        return choice, self.used[-1], last_output

    async def ask_to_finish_and_save(self) -> List[str]:
        if getattr(self, "production_auto_mode", False):
            self._save(self.used)
            return self.used
        if not self.auto_finish:
            logger.info(
                "\nAdd new prompts (Ctrl+D to submit, empty submits nothing and finishes):"
            )
            while True:
                text = await self._ask_multiline("> ")
                if not text.strip():
                    break
                self.used.append(text)
                self._save(self.used)
                await self._maybe_await_callback(
                    text,
                    text[:20],
                    len(self.used) - 1,
                    None,
                    None,
                )

            self._save(self.used)

        return self.used

    def _validate_production_choice_policy(self) -> None:
        """Reject production automation choices that bypass registered prompt assets."""
        if not self.production_auto_mode:
            return None
        forbidden = {"r", "i"} & set(self.allowed_choices)
        if forbidden:
            raise ValueError(
                "Production auto_u conversations cannot enable prompt "
                f"replace/insert choices: {', '.join(sorted(forbidden))}"
            )
        return None

    # ---------- persistence ----------

    def _save(self, prompts: List[str]) -> None:
        self._atomic_write_json(prompts)

    def _atomic_write_json(self, prompts: List[str]) -> None:
        atomic_write(
            path=self.conversation_json_path,
            data=(json.dumps(prompts, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8"
            ),
        )

    # ---------- UI ----------

    def _create_session(self) -> PromptSession:
        kb = KeyBindings()

        @kb.add("c-d", filter=is_multiline)
        def _(event):
            event.app.current_buffer.validate_and_handle()

        return PromptSession(key_bindings=kb)

    def _show_prompt(self, prompt: str, additional_info: Optional[str] = None) -> None:
        logger.info("=" * 60)
        logger.info(f"Prompt {additional_info if additional_info is not None else ''}:")
        logger.info(prompt)
        logger.info("=" * 60)

    async def _ask_choice(self, prompt: str) -> str:
        labels = " / ".join(
            _CHOICE_LABELS[c] for c in self.allowed_choices if c in _CHOICE_LABELS
        )
        prompt_text = HTML(f"{labels} ? ")

        notified = False
        hostname = os.uname().nodename
        notify_msg = (
            f"**LLM requires action on prompt ({hostname}):**\n"
            f"```quote\n{prompt[:1000]}\n```"
        )

        while True:
            if not notified and self.notify:
                send_notification(notify_msg, check_tmux=True)

            prompt_task = asyncio.create_task(self._session.prompt_async(prompt_text))

            while True:
                try:
                    raw = await asyncio.wait_for(
                        asyncio.shield(prompt_task),
                        timeout=NOTIFY_AFTER_SEC,
                    )
                except asyncio.TimeoutError:
                    if self.notify and not notified:
                        send_notification(notify_msg, check_tmux=False)
                        notified = True
                    continue

                choice = (raw or "").strip().lower()
                if choice in self.allowed_choices:
                    return choice

                # invalid input: restart a fresh prompt
                prompt_task.cancel()
                break

    async def _ask_multiline(self, label: str) -> str:
        text = await self._session.prompt_async(
            HTML(f"<b>{label}</b> "),
            multiline=True,
        )
        return text.strip()

    async def _maybe_await_callback(
        self,
        prompt: str,
        prompt_descriptor: Optional[str],  # short description of the prompt
        index: int,
        max_turns: Optional[int] = None,
        prompt_metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        res = self.callback(
            prompt,
            prompt_descriptor,
            index,
            max_turns,
            prompt_metadata,
        )
        if hasattr(res, "__await__"):
            return await res  # type: ignore
        return res
