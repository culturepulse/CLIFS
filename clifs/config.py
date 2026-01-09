"""Configuration management using pydantic-settings."""

from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import torch


class PathSettings(BaseSettings):
    """Path configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CLIFS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
        description="Project root directory"
    )

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def mbert_model_path(self) -> Path:
        return (
            self.models_dir
            / "best_mbert_model"
            / "best_mbert_model"
            / "modern_BERT_fusion_augmented_data_finegrain"
        )

    @property
    def rf_models_dir(self) -> Path:
        return self.models_dir / "best_rf"

    @property
    def exec_dir(self) -> Path:
        return self.project_root / "exec_dir"

    def ensure_directories(self) -> None:
        """Create necessary directories."""
        self.exec_dir.mkdir(parents=True, exist_ok=True)


class RuntimeSettings(BaseSettings):
    """Runtime configuration."""

    model_config = SettingsConfigDict(
        env_prefix="CLIFS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Device settings
    force_cpu: bool = Field(default=False, description="Force CPU usage")
    batch_size: int = Field(default=1, description="Batch size for inference")

    # Torch settings
    disable_torch_compile: bool = Field(default=True, description="Disable torch.compile")
    float32_matmul_precision: str = Field(default="high", description="Matmul precision")

    # Logging
    verbose: bool = Field(default=False, description="Enable verbose logging")
    suppress_warnings: bool = Field(default=True, description="Suppress torch warnings")

    @property
    def device(self) -> torch.device:
        """Get compute device."""
        if self.force_cpu:
            return torch.device("cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_environment(self) -> None:
        """Configure torch environment."""
        import os
        import logging
        import warnings
        import torch
        import torch._dynamo

        if self.disable_torch_compile:
            os.environ["TORCH_COMPILE_DISABLE"] = "1"
            os.environ["TORCHDYNAMO_VERBOSE"] = "0"
            os.environ.pop("TORCH_LOGS", None)

        torch.set_float32_matmul_precision(self.float32_matmul_precision)
        torch._dynamo.disable()
        torch._dynamo.config.suppress_errors = True

        if self.suppress_warnings:
            for name in ("torch._dynamo", "torch._inductor", "torch.overrides"):
                logging.getLogger(name).setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", module="torch._inductor")
            warnings.filterwarnings("ignore", module="torch.overrides")


class CLIFSConfig:
    """Combined configuration."""

    def __init__(
        self,
        paths: Optional[PathSettings] = None,
        runtime: Optional[RuntimeSettings] = None
    ):
        self.paths = paths or PathSettings()
        self.runtime = runtime or RuntimeSettings()

    @classmethod
    def from_env(cls) -> "CLIFSConfig":
        """Load configuration from environment."""
        return cls(
            paths=PathSettings(),
            runtime=RuntimeSettings()
        )