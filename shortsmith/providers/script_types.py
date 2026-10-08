"""Shared script data types.

Kept in their own module so the offline writer and the model-backed writer can
both import them without a circular import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Scene:
    narration: str
    visual_prompt: str


@dataclass
class ScriptResult:
    title: str
    hook: str
    scenes: list[Scene]
    cta: str
    description: str
    hashtags: list[str] = field(default_factory=list)
    provider: str = "offline"
    notes: str = ""

    @property
    def narration_text(self) -> str:
        parts = [self.hook] + [s.narration for s in self.scenes]
        if self.cta:
            parts.append(self.cta)
        return " ".join(p.strip() for p in parts if p and p.strip())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["narration_text"] = self.narration_text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScriptResult":
        return cls(
            title=data.get("title", ""),
            hook=data.get("hook", ""),
            scenes=[
                Scene(narration=s.get("narration", ""), visual_prompt=s.get("visual_prompt", ""))
                for s in data.get("scenes", [])
            ],
            cta=data.get("cta", ""),
            description=data.get("description", ""),
            hashtags=list(data.get("hashtags", [])),
            provider=data.get("provider", "offline"),
            notes=data.get("notes", ""),
        )
