"""Model profile data structures.

Encapsulates architectural, compression, and computational metadata for speech models.
Allows unknown or unmeasured values to remain None.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelProfile:
    """Represents the computational profile of a candidate speech model configuration.

    Attributes:
        name: Identifier or configuration tag for the model.
        parameters: Total parameter count (or active parameter count).
        model_size_mb: Serialized model file footprint on disk in Megabytes.
        precision: Arithmetic precision (e.g., 'fp32', 'fp16', 'int8', 'int4').
        sparsity: Structured sparsity ratio in range [0.0, 1.0].
        flops: Theoretical Floating Point Operations per second of audio.
        runtime: Target runtime engine (e.g., 'onnxruntime', 'sherpa-onnx', 'executorch').
        chunk_ms: Streaming acoustic chunk window in milliseconds.
    """

    name: str
    parameters: Optional[int] = None
    model_size_mb: Optional[float] = None
    precision: str = "fp32"
    sparsity: Optional[float] = None
    flops: Optional[float] = None
    runtime: Optional[str] = None
    chunk_ms: Optional[int] = None

    def __post_init__(self) -> None:
        if self.sparsity is not None and not (0.0 <= self.sparsity <= 1.0):
            raise ValueError(f"Sparsity must be in range [0.0, 1.0], got {self.sparsity}")
        if self.model_size_mb is not None and self.model_size_mb <= 0:
            raise ValueError(f"Model size must be positive, got {self.model_size_mb}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to dictionary representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelProfile:
        """Construct a ModelProfile from a dictionary."""
        return cls(
            name=data.get("name", "unnamed_profile"),
            parameters=data.get("parameters"),
            model_size_mb=data.get("model_size_mb"),
            precision=data.get("precision", "fp32"),
            sparsity=data.get("sparsity"),
            flops=data.get("flops"),
            runtime=data.get("runtime"),
            chunk_ms=data.get("chunk_ms"),
        )
