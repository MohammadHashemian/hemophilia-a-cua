from pydantic import BaseModel


class InputMetadata(BaseModel):
    """Human-readable provenance stored next to model inputs."""

    description: str | None = None
    reference: str | list[str] | None = None
