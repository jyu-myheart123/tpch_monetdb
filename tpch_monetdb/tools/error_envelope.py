from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ErrorEnvelope:
    code: str
    category: str
    stage: str
    message: str
    recoverable: bool = True
    relevant_files: tuple[str, ...] = ()
    allowed_next_actions: tuple[str, ...] = ()
    recommended_next_action: str | None = None

    def __str__(self) -> str:
        lines = [f"[ERROR:{self.code}] {self.message}"]
        lines.append(f"Category: {self.category}")
        lines.append(f"Stage: {self.stage}")
        lines.append(f"Recoverable: {'yes' if self.recoverable else 'no'}")
        if self.relevant_files:
            lines.append(f"Relevant files: {', '.join(self.relevant_files)}")
        if self.allowed_next_actions:
            lines.append(
                f"Allowed next actions: {', '.join(self.allowed_next_actions)}"
            )
        if self.recommended_next_action is not None:
            lines.append(
                f"Recommended next action: {self.recommended_next_action}"
            )
        return "\n".join(lines)
