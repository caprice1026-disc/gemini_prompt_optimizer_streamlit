from dataclasses import dataclass


@dataclass
class PromptCandidate:
    id: int
    label: str
    prompt: str
