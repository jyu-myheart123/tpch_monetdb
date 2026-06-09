import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tpch_monetdb.conversations.conversation import AbstractConversation
from tpch_monetdb.conversations.agent_text_registry import render_agent_text_asset
from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
from tpch_monetdb.utils.generated_query_checks import run_generated_code_checks
from tpch_monetdb.utils.control_artifacts import build_storage_plan_alignment
from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

logger = logging.getLogger(__name__)
_FILE_CONTRACT_REMEDIATION_LIMIT = 3
_VALIDATION_RERUN_POSTCONDITION = "validation was not rerun after the latest file write"
_STORAGE_PLAN_CONTRACT_POSTCONDITION = "storage_plan_contract_complete"
_STORAGE_PLAN_CONTRACT_FAILURE_PREFIX = "storage_plan_contract_complete advisory"
_PROMOTABLE_STORAGE_PLAN_ALIGNMENT_STATUSES = frozenset({"contract_valid", "aligned"})


@dataclass(frozen=True)
class PromptStep:
    text: str
    max_turns: int | None = None
    descriptor: str | None = None
    tool_profile: str | None = None
    rule_area: str | None = None
    required_nonempty_files: tuple[str, ...] = field(default_factory=tuple)
    required_updated_files: tuple[str, ...] = field(default_factory=tuple)
    stop_conditions: tuple[str, ...] = field(default_factory=tuple)
    expected_query_id: str | None = None
    generated_code_checks: tuple[str, ...] = field(default_factory=tuple)
    active_unit_id: str | None = None
    active_unit_kind: str | None = None
    active_unit_files: tuple[str, ...] = field(default_factory=tuple)
    active_unit_query_ids: tuple[str, ...] = field(default_factory=tuple)
    required_control_artifacts: tuple[str, ...] = field(default_factory=tuple)
    control_artifacts_injected: tuple[str, ...] = field(default_factory=tuple)
    advisory_postconditions: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_json_value(cls, value: Any) -> "PromptStep":
        """Parse one prompt-step JSON object into a strongly typed PromptStep."""
        if isinstance(value, str):
            return cls(text=value, tool_profile="legacy_general")
        if not isinstance(value, dict):
            raise ValueError("JSON file must contain strings or prompt step objects")
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("Prompt step object must contain string field 'text'")
        max_turns = value.get("max_turns")
        descriptor = value.get("descriptor")
        tool_profile = value.get("tool_profile")
        rule_area = value.get("rule_area")
        required_files = value.get("required_nonempty_files", [])
        required_updated_files = value.get("required_updated_files", [])
        stop_conditions = value.get("stop_conditions", [])
        expected_query_id = value.get("expected_query_id")
        generated_code_checks = value.get("generated_code_checks", [])
        active_unit_id = value.get("active_unit_id")
        active_unit_kind = value.get("active_unit_kind")
        active_unit_files = value.get("active_unit_files", [])
        active_unit_query_ids = value.get("active_unit_query_ids", [])
        required_control_artifacts = value.get("required_control_artifacts", [])
        control_artifacts_injected = value.get("control_artifacts_injected", [])
        advisory_postconditions = value.get("advisory_postconditions", [])
        if max_turns is not None and not isinstance(max_turns, int):
            raise ValueError("Prompt step field 'max_turns' must be an integer")
        if descriptor is not None and not isinstance(descriptor, str):
            raise ValueError("Prompt step field 'descriptor' must be a string")
        if tool_profile is not None and not isinstance(tool_profile, str):
            raise ValueError("Prompt step field 'tool_profile' must be a string")
        if rule_area is not None and not isinstance(rule_area, str):
            raise ValueError("Prompt step field 'rule_area' must be a string")
        if expected_query_id is not None and not isinstance(expected_query_id, str):
            raise ValueError("Prompt step field 'expected_query_id' must be a string")
        if active_unit_id is not None and not isinstance(active_unit_id, str):
            raise ValueError("Prompt step field 'active_unit_id' must be a string")
        if active_unit_kind is not None and not isinstance(active_unit_kind, str):
            raise ValueError("Prompt step field 'active_unit_kind' must be a string")
        if not isinstance(required_files, list) or not all(
            isinstance(item, str) for item in required_files
        ):
            raise ValueError(
                "Prompt step field 'required_nonempty_files' must be a list of strings"
            )
        if not isinstance(required_updated_files, list) or not all(
            isinstance(item, str) for item in required_updated_files
        ):
            raise ValueError(
                "Prompt step field 'required_updated_files' must be a list of strings"
            )
        if not isinstance(stop_conditions, list) or not all(
            isinstance(item, str) for item in stop_conditions
        ):
            raise ValueError(
                "Prompt step field 'stop_conditions' must be a list of strings"
            )
        if not isinstance(generated_code_checks, list) or not all(
            isinstance(item, str) for item in generated_code_checks
        ):
            raise ValueError(
                "Prompt step field 'generated_code_checks' must be a list of strings"
            )
        if not isinstance(active_unit_files, list) or not all(
            isinstance(item, str) for item in active_unit_files
        ):
            raise ValueError(
                "Prompt step field 'active_unit_files' must be a list of strings"
            )
        if not isinstance(active_unit_query_ids, list) or not all(
            isinstance(item, str) for item in active_unit_query_ids
        ):
            raise ValueError(
                "Prompt step field 'active_unit_query_ids' must be a list of strings"
            )
        if not isinstance(required_control_artifacts, list) or not all(
            isinstance(item, str) for item in required_control_artifacts
        ):
            raise ValueError(
                "Prompt step field 'required_control_artifacts' must be a list of strings"
            )
        if not isinstance(control_artifacts_injected, list) or not all(
            isinstance(item, str) for item in control_artifacts_injected
        ):
            raise ValueError(
                "Prompt step field 'control_artifacts_injected' must be a list of strings"
            )
        if not isinstance(advisory_postconditions, list) or not all(
            isinstance(item, str) for item in advisory_postconditions
        ):
            raise ValueError(
                "Prompt step field 'advisory_postconditions' must be a list of strings"
            )
        return cls(
            text=text,
            max_turns=max_turns,
            descriptor=descriptor,
            tool_profile=tool_profile,
            rule_area=rule_area,
            required_nonempty_files=tuple(required_files),
            required_updated_files=tuple(required_updated_files),
            stop_conditions=tuple(stop_conditions),
            expected_query_id=expected_query_id,
            generated_code_checks=tuple(generated_code_checks),
            active_unit_id=active_unit_id,
            active_unit_kind=active_unit_kind,
            active_unit_files=tuple(active_unit_files),
            active_unit_query_ids=tuple(active_unit_query_ids),
            required_control_artifacts=tuple(required_control_artifacts),
            control_artifacts_injected=tuple(control_artifacts_injected),
            advisory_postconditions=tuple(advisory_postconditions),
        )

    def to_callback_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        if self.tool_profile is not None:
            metadata["tool_profile"] = self.tool_profile
        if self.rule_area is not None:
            metadata["rule_area"] = self.rule_area
        if self.active_unit_id is not None:
            metadata["active_unit_id"] = self.active_unit_id
        if self.active_unit_kind is not None:
            metadata["active_unit_kind"] = self.active_unit_kind
        if self.active_unit_files:
            metadata["active_unit_files"] = list(self.active_unit_files)
        if self.active_unit_query_ids:
            metadata["active_unit_query_ids"] = list(self.active_unit_query_ids)
        if self.required_control_artifacts:
            metadata["required_control_artifacts"] = list(self.required_control_artifacts)
        if self.control_artifacts_injected:
            metadata["control_artifacts_injected"] = list(self.control_artifacts_injected)
        if self.advisory_postconditions:
            metadata["advisory_postconditions"] = list(self.advisory_postconditions)
        return metadata


class ScriptedConversation(AbstractConversation):
    def __init__(
        self,
        **kwargs,
    ) -> None:
        allowed_choices = ("u", "c") if kwargs.get("auto_u") else ("u", "r", "i", "c")
        super().__init__(
            allowed_choices=allowed_choices,
            **kwargs,
        )
        self.prompts: list[PromptStep] = self._load()
        self._budget_tracker = StageBudgetTracker()
        self.completed_stage_summaries: list[StageRunSummary] = []
        return None

    @dataclass(frozen=True)
    class _RecoverablePostcondition:
        failed_postcondition: str
        remediation_prompt: str
        descriptor_suffix: str = "file_contract_remediation"

    async def run(self) -> Optional[list[str]]:
        """Run prompt steps and retry recoverable postcondition failures in-place."""
        if self.replay:
            for idx, step in enumerate(self.prompts):
                await self._maybe_await_callback(
                    step.text,
                    step.descriptor,
                    idx,
                    step.max_turns,
                    step.to_callback_metadata(),
                )
            return None

        self.used = []
        idx = 0

        while idx < len(self.prompts):
            step = self.prompts[idx]
            display_label = step.descriptor or str(idx)
            effective_max_turns = self._budget_tracker.compute_effective_max_turns(
                stage_descriptor=step.descriptor or "",
                static_budget=step.max_turns,
            )
            remediation_attempts = 0
            current_prompt_text = step.text
            current_descriptor = step.descriptor
            current_label = display_label
            soft_advisory_seen = False
            last_hard_valid_callback_result: Any | None = None
            while True:
                choice, _, callback_result = await self.process_prompt(
                    current_prompt_text,
                    prompt_descriptor=current_descriptor,
                    max_turns=effective_max_turns,
                    additional_out_str=current_label,
                    prompt_metadata=step.to_callback_metadata(),
                )
                if choice not in ["u", "r"]:
                    break
                if isinstance(callback_result, StageRunSummary):
                    self._budget_tracker.record_stage_result(
                        stage_descriptor=step.descriptor or "",
                        summary=callback_result,
                    )
                recoverable = self._get_recoverable_postcondition(
                    step=step,
                    idx=idx,
                    callback_result=callback_result,
                )
                if (
                    recoverable is not None
                    and self._is_soft_advisory_failure(
                        recoverable.failed_postcondition
                    )
                ):
                    soft_advisory_seen = True
                    last_hard_valid_callback_result = callback_result
                if (
                    recoverable is not None
                    and remediation_attempts < _FILE_CONTRACT_REMEDIATION_LIMIT
                ):
                    remediation_attempts += 1
                    current_prompt_text = recoverable.remediation_prompt
                    current_descriptor = (
                        f"{step.descriptor or 'prompt'}__{recoverable.descriptor_suffix}"
                    )
                    current_label = f"{display_label} remediation {remediation_attempts}"
                    logger.warning(
                        "Recoverable postcondition failure for %s: %s. Running remediation turn %d/%d.",
                        step.descriptor or idx,
                        recoverable.failed_postcondition,
                        remediation_attempts,
                        _FILE_CONTRACT_REMEDIATION_LIMIT,
                    )
                    continue
                if (
                    recoverable is not None
                    and soft_advisory_seen
                    and remediation_attempts >= _FILE_CONTRACT_REMEDIATION_LIMIT
                ):
                    logger.warning(
                        "Advisory postcondition for %s still failing after %d remediation attempts; continuing without fatal failure: %s",
                        step.descriptor or idx,
                        remediation_attempts,
                        recoverable.failed_postcondition,
                    )
                    if last_hard_valid_callback_result is not None:
                        callback_result = last_hard_valid_callback_result
                self._validate_step_postconditions(
                    step=step,
                    idx=idx,
                    callback_result=callback_result,
                )
                if isinstance(callback_result, StageRunSummary):
                    self.completed_stage_summaries.append(callback_result)
                idx += 1
                break

        used = await self.ask_to_finish_and_save()
        return used

    def _load(self) -> list[PromptStep]:
        if not self.conversation_json_path.exists():
            self.conversation_json_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write_json([])
            return []

        with self.conversation_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("JSON file must contain an array")
        steps = [PromptStep.from_json_value(item) for item in data]
        return steps

    _DIAGNOSTIC_BUDGET = 1200

    def _format_postcondition_diagnostic(
        self,
        step: PromptStep,
        idx: int,
        failed_postcondition: str,
        summary: StageRunSummary | None,
    ) -> str:
        parts: list[str] = [
            f"[ERROR:STAGE_POSTCONDITION_FAILED] Prompt {idx} ({step.descriptor or 'prompt'}) failed: {failed_postcondition}",
        ]
        if summary is not None:
            recent_writes = list(summary.written_files)[:8]
            if recent_writes:
                parts.append(f"Recent writes: {', '.join(recent_writes)}")
            if summary.last_compile_summary is not None:
                compile_text = summary.last_compile_summary
                if len(compile_text) > 200:
                    compile_text = compile_text[:200] + "...[TRUNCATED]"
                parts.append(f"Last compile: {compile_text}")
            if summary.last_validation_summary is not None:
                val_text = summary.last_validation_summary
                if len(val_text) > 200:
                    val_text = val_text[:200] + "...[TRUNCATED]"
                parts.append(f"Last validation: {val_text}")
            if summary.has_writes:
                parts.append(
                    "Validation/write revisions: "
                    f"last_run={summary.run_write_revision}, "
                    f"current_write={summary.write_revision}"
                )
        result = "\n".join(parts)
        if len(result) > self._DIAGNOSTIC_BUDGET:
            result = result[:self._DIAGNOSTIC_BUDGET] + "\n...[TRUNCATED]"
        return result

    def _validate_step_postconditions(
        self,
        step: PromptStep,
        idx: int,
        callback_result: Any,
    ) -> None:
        summary = callback_result if isinstance(callback_result, StageRunSummary) else None
        if step.required_nonempty_files:
            self._validate_required_files(step=step, idx=idx, summary=summary)
        self._validate_required_control_artifacts(step=step, idx=idx, summary=summary)
        self._validate_generated_code_checks(step=step, idx=idx, summary=summary)
        if not step.required_updated_files and not step.stop_conditions:
            return None
        if summary is None:
            raise RuntimeError(
                self._format_postcondition_diagnostic(
                    step=step,
                    idx=idx,
                    failed_postcondition="stage callback did not return StageRunSummary",
                    summary=None,
                )
            )
        self._validate_required_updates(step, idx, summary)
        self._validate_stop_conditions(step, idx, summary)
        return None

    def _validate_required_control_artifacts(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary | None,
    ) -> None:
        if not step.required_control_artifacts:
            return None
        if summary is None:
            raise RuntimeError(
                self._format_postcondition_diagnostic(
                    step=step,
                    idx=idx,
                    failed_postcondition="required_control_artifacts require StageRunSummary",
                    summary=None,
                )
            )
        available = set(summary.control_artifacts_read) | set(summary.control_artifacts_injected)
        missing = [
            artifact for artifact in step.required_control_artifacts
            if artifact not in available
        ]
        if missing:
            raise RuntimeError(
                self._format_postcondition_diagnostic(
                    step=step,
                    idx=idx,
                    failed_postcondition=(
                        "required_control_artifacts missing: "
                        + ", ".join(missing)
                    ),
                    summary=summary,
                )
            )
        return None

    def _get_recoverable_postcondition(
        self,
        step: PromptStep,
        idx: int,
        callback_result: Any,
    ) -> _RecoverablePostcondition | None:
        """Return a remediation prompt for postcondition failures worth retrying."""
        summary = callback_result if isinstance(callback_result, StageRunSummary) else None
        failed_postcondition = self._detect_recoverable_postcondition(
            step=step,
            idx=idx,
            summary=summary,
        )
        if failed_postcondition is None:
            return None
        if failed_postcondition == _VALIDATION_RERUN_POSTCONDITION:
            remediation_prompt = self._build_validation_rerun_remediation_prompt(
                step=step,
                idx=idx,
                failed_postcondition=failed_postcondition,
                summary=summary,
            )
            descriptor_suffix = "validation_rerun_remediation"
        elif self._is_soft_advisory_failure(failed_postcondition):
            remediation_prompt = self._build_storage_plan_contract_remediation_prompt(
                step=step,
                idx=idx,
                failed_postcondition=failed_postcondition,
                summary=summary,
            )
            descriptor_suffix = "storage_plan_contract_remediation"
        else:
            remediation_prompt = self._build_file_contract_remediation_prompt(
                step=step,
                idx=idx,
                failed_postcondition=failed_postcondition,
                summary=summary,
            )
            descriptor_suffix = "file_contract_remediation"
        return self._RecoverablePostcondition(
            failed_postcondition=failed_postcondition,
            remediation_prompt=remediation_prompt,
            descriptor_suffix=descriptor_suffix,
        )

    def _detect_recoverable_postcondition(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary | None,
    ) -> str | None:
        """Detect postcondition failures where another model turn can repair state."""
        if step.required_nonempty_files:
            missing_or_empty = self._get_required_file_failure(
                step=step,
                idx=idx,
                summary=summary,
            )
            if missing_or_empty is not None:
                return missing_or_empty
        if summary is None:
            return None
        if step.required_updated_files:
            missing_update = self._get_required_update_failure(step, idx, summary)
            if missing_update is not None:
                return missing_update
        if _STORAGE_PLAN_CONTRACT_POSTCONDITION in step.advisory_postconditions:
            storage_plan_failure = self._get_storage_plan_contract_advisory_failure()
            if storage_plan_failure is not None:
                return storage_plan_failure
        if (
            "validation_passed" in step.stop_conditions
            and summary.validation_passed is not True
            and summary.has_writes
            and summary.run_write_revision < summary.write_revision
        ):
            return _VALIDATION_RERUN_POSTCONDITION
        return None

    def _get_storage_plan_contract_advisory_failure(self) -> str | None:
        """Return storage-plan contract repair guidance without making it fatal."""
        if self.workspace_root is None:
            return None
        alignment = build_storage_plan_alignment(
            self.workspace_root / "storage_plan.txt"
        )
        status = str(alignment.get("status") or "")
        if status in _PROMOTABLE_STORAGE_PLAN_ALIGNMENT_STATUSES:
            return None
        departures = alignment.get("departures") or []
        missing_query_ids = alignment.get("missing_query_ids") or []
        missing_obligation_query_ids = (
            alignment.get("missing_obligation_query_ids") or []
        )
        covered_query_ids = alignment.get("covered_critical_query_ids") or []
        details = [
            f"status={status or 'missing'}",
            "departures=" + self._format_advisory_list(departures),
            "missing_query_ids=" + self._format_advisory_list(missing_query_ids),
            "missing_obligation_query_ids="
            + self._format_advisory_list(missing_obligation_query_ids),
            "covered_critical_query_ids="
            + self._format_advisory_list(covered_query_ids),
        ]
        return f"{_STORAGE_PLAN_CONTRACT_FAILURE_PREFIX}: " + "; ".join(details)

    def _format_advisory_list(self, values: Any) -> str:
        """Format advisory details compactly for model-facing remediation prompts."""
        if not isinstance(values, list | tuple):
            return str(values)
        if not values:
            return "(none)"
        text = ", ".join(str(value) for value in values[:12])
        if len(values) > 12:
            text += ", ..."
        return text

    def _is_soft_advisory_failure(self, failed_postcondition: str) -> bool:
        """Return whether a failed postcondition may continue after retries."""
        return failed_postcondition.startswith(_STORAGE_PLAN_CONTRACT_FAILURE_PREFIX)

    def _get_required_file_failure(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary | None,
    ) -> str | None:
        if self.workspace_root is None:
            return None
        for relative_path in step.required_nonempty_files:
            target = (self.workspace_root / relative_path).resolve()
            try:
                target.relative_to(self.workspace_root.resolve())
            except ValueError:
                return None
            if not target.exists():
                return f"required file {relative_path} does not exist"
            if target.stat().st_size == 0:
                return f"required file {relative_path} is empty"
        return None

    def _get_required_update_failure(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary,
    ) -> str | None:
        updated_files = set(summary.written_files)
        for relative_path in step.required_updated_files:
            if relative_path not in updated_files:
                return f"required file {relative_path} was not updated"
        return None

    def _build_file_contract_remediation_prompt(
        self,
        step: PromptStep,
        idx: int,
        failed_postcondition: str,
        summary: StageRunSummary | None,
    ) -> str:
        required_targets = ", ".join(step.required_nonempty_files) or "(none)"
        recent_writes = ", ".join(summary.written_files[:8]) if summary is not None and summary.written_files else "(none)"
        last_validation = (
            summary.last_validation_summary[:300] + ("...[TRUNCATED]" if len(summary.last_validation_summary) > 300 else "")
            if summary is not None and summary.last_validation_summary
            else "(none)"
        )
        return render_agent_text_asset(
            "scripted.remediation.file_contract",
            {
                "original_prompt": step.text,
                "stage_label": step.descriptor or idx,
                "failed_postcondition": failed_postcondition,
                "required_targets": required_targets,
                "recent_writes": recent_writes,
                "last_validation": last_validation,
                "remediation_limit": _FILE_CONTRACT_REMEDIATION_LIMIT,
            },
        )

    def _build_validation_rerun_remediation_prompt(
        self,
        step: PromptStep,
        idx: int,
        failed_postcondition: str,
        summary: StageRunSummary | None,
    ) -> str:
        recent_writes = ", ".join(summary.written_files[:8]) if summary is not None and summary.written_files else "(none)"
        last_validation = (
            summary.last_validation_summary[:300] + ("...[TRUNCATED]" if len(summary.last_validation_summary) > 300 else "")
            if summary is not None and summary.last_validation_summary
            else "(none)"
        )
        return render_agent_text_asset(
            "scripted.remediation.validation_rerun",
            {
                "original_prompt": step.text,
                "stage_label": step.descriptor or idx,
                "failed_postcondition": failed_postcondition,
                "expected_query_id": step.expected_query_id or "(stage scope)",
                "recent_writes": recent_writes,
                "last_validation": last_validation,
                "remediation_limit": _FILE_CONTRACT_REMEDIATION_LIMIT,
            },
        )

    def _build_storage_plan_contract_remediation_prompt(
        self,
        step: PromptStep,
        idx: int,
        failed_postcondition: str,
        summary: StageRunSummary | None,
    ) -> str:
        """Build a Storage Plan contract repair prompt from advisory lint output."""
        recent_writes = ", ".join(summary.written_files[:8]) if summary is not None and summary.written_files else "(none)"
        return render_agent_text_asset(
            "scripted.remediation.storage_plan_contract",
            {
                "original_prompt": step.text,
                "stage_label": step.descriptor or idx,
                "failed_postcondition": failed_postcondition,
                "recent_writes": recent_writes,
                "remediation_limit": _FILE_CONTRACT_REMEDIATION_LIMIT,
            },
        )

    def _validate_required_files(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary | None,
    ) -> None:
        if self.workspace_root is None:
            raise RuntimeError(
                self._format_postcondition_diagnostic(
                    step=step,
                    idx=idx,
                    failed_postcondition="workspace_root is not configured",
                    summary=summary,
                )
            )
        for relative_path in step.required_nonempty_files:
            target = (self.workspace_root / relative_path).resolve()
            try:
                target.relative_to(self.workspace_root.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition=f"postcondition outside workspace: {relative_path}",
                        summary=summary,
                    )
                ) from exc
            if not target.exists():
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition=f"required file {relative_path} does not exist",
                        summary=summary,
                    )
                )
            if target.stat().st_size == 0:
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition=f"required file {relative_path} is empty",
                        summary=summary,
                    )
                )
        return None

    def _validate_required_updates(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary,
    ) -> None:
        updated_files = set(summary.written_files)
        for relative_path in step.required_updated_files:
            if relative_path not in updated_files:
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition=f"required updated file {relative_path} was not modified",
                        summary=summary,
                    )
                )
        return None

    def _validate_generated_code_checks(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary | None,
    ) -> None:
        """Run static generated-code checks configured on the active prompt step."""
        if not step.generated_code_checks:
            return None
        if self.workspace_root is None:
            raise RuntimeError(
                self._format_postcondition_diagnostic(
                    step=step,
                    idx=idx,
                    failed_postcondition="workspace_root is not configured",
                    summary=summary,
                )
            )
        violations = run_generated_code_checks(
            workspace_root=self.workspace_root,
            expected_query_id=step.expected_query_id,
            checks=step.generated_code_checks,
            active_unit_files=step.active_unit_files,
        )
        blocking_violations = [
            violation for violation in violations
            if violation.severity != "diagnostic"
        ]
        diagnostic_violations = [
            violation for violation in violations
            if violation.severity == "diagnostic"
        ]
        for violation in diagnostic_violations:
            logger.info(
                "Generated-code diagnostic for %s: %s",
                violation.file_path,
                violation.message,
            )
        if not blocking_violations:
            return None
        violation_text = "; ".join(
            f"{violation.code}: {violation.message}"
            for violation in blocking_violations[:4]
        )
        raise RuntimeError(
            self._format_postcondition_diagnostic(
                step=step,
                idx=idx,
                failed_postcondition=violation_text,
                summary=summary,
            )
        )

    def _validate_stop_conditions(
        self,
        step: PromptStep,
        idx: int,
        summary: StageRunSummary,
    ) -> None:
        for condition in step.stop_conditions:
            if condition == "write_required" and not summary.has_writes:
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition="the stage finished without any file write",
                        summary=summary,
                    )
                )
            if condition == "validation_passed" and summary.validation_passed is not True:
                failed_postcondition = "validation did not pass"
                if summary.has_writes and summary.run_write_revision < summary.write_revision:
                    failed_postcondition = _VALIDATION_RERUN_POSTCONDITION
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition=failed_postcondition,
                        summary=summary,
                    )
                )
            if condition == "primary_file_present":
                if not step.required_nonempty_files:
                    raise RuntimeError(
                        self._format_postcondition_diagnostic(
                            step=step,
                            idx=idx,
                            failed_postcondition="primary_file_present requires required_nonempty_files",
                            summary=summary,
                        )
                    )
            if condition == "primary_file_written":
                if not step.required_nonempty_files:
                    raise RuntimeError(
                        self._format_postcondition_diagnostic(
                            step=step,
                            idx=idx,
                            failed_postcondition="primary_file_written requires required_nonempty_files",
                            summary=summary,
                        )
                    )
                primary_file = step.required_nonempty_files[0]
                if primary_file not in set(summary.written_files):
                    raise RuntimeError(
                        self._format_postcondition_diagnostic(
                            step=step,
                            idx=idx,
                            failed_postcondition=f"primary file {primary_file} was not written",
                            summary=summary,
                        )
                    )
            if condition == "todo_progress" and not summary.todo_progressed:
                raise RuntimeError(
                    self._format_postcondition_diagnostic(
                        step=step,
                        idx=idx,
                        failed_postcondition="TODO state did not progress",
                        summary=summary,
                    )
                )
        return None
