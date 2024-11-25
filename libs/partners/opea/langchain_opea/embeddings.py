"""opea embeddings wrapper."""

from __future__ import annotations

from typing import Dict, Optional, List, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from pydantic import ConfigDict, Field, SecretStr, model_validator
import openai
from langchain_core.utils import from_env, secret_from_env
from typing_extensions import Self

DEFAULT_API_BASE = "https://localhost:6006/v1"
DEFAULT_MODEL = "thenlper/gte-large"


class OPEAEmbeddings(OpenAIEmbeddings):
    """`OPEA` OPENAI Compatible Embeddings API."""

    model_name: str = Field(alias="model")
    """Model name to use."""
    opea_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("OPEA_API_KEY", default=None),
    )
    """OPEA_API_KEY.

    Automatically read from env variable `OPEA_API_KEY` if not provided.
    """
    opea_api_base: str = Field(default="https://localhost:6006/v1/")
    """Base URL path for API requests."""

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "llms", "opea"]

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "opea_api_key": "OPEA_API_KEY"
        }

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        client_params: dict = {
            "api_key": self.opea_api_key.get_secret_value(),
            "base_url": self.opea_api_base,
        }
        if not self.client:
            sync_specific = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).completions  # type: ignore[arg-type]
        if not self.async_client:
            async_specific = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).completions

        return self


    @property
    def _invocation_params(self) -> Dict[str, Any]:
        openai_params = {"model": self.model_name}
        return {**openai_params, **super()._invocation_params}

    @property
    def _llm_type(self) -> str:
        return "opea-embedding"