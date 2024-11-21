"""OPEA Endpoints chat wrapper. Relies heavily on ChatOpenAI."""

from typing import (
    Any,
    Callable,
    Dict,
    Literal,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field, SecretStr

from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.utils.openai import is_openai_v1

DEFAULT_API_BASE = "https://localhost:8080/v1/"
DEFAULT_MODEL = "llama-2-13b-chat"


class ChatOPEA(ChatOpenAI):
    """OPEA Chat large language models.

    See https://opea.dev/ for information about OPEA.

    To use, you should have the ``openai`` python package installed and the
    environment variable ``opea_api_key`` set with your API token.
    Alternatively, you can use the opea_api_key keyword argument.

    Any parameters that are valid to be passed to the `openai.create` call can be passed
    in, even if not explicitly saved on this class.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatOPEA
            chat = ChatOctoAI(model_name="mixtral-8x7b-instruct")
    """

    opea_api_base: str = Field(default=DEFAULT_API_BASE)
    opea_api_key: SecretStr = Field(default=None, alias="api_key")
    model_name: str = Field(default=DEFAULT_MODEL, alias="model")

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "opea-chat"

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"opea_api_key": "opea_api_key"}

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["opea_api_base"] = get_from_dict_or_env(
            values,
            "opea_api_base",
            "OPEA_API_BASE",
            default=DEFAULT_API_BASE,
        )
        values["opea_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "opea_api_key", "opea_api_key")
        )
        values["model_name"] = get_from_dict_or_env(
            values,
            "model_name",
            "MODEL_NAME",
            default=DEFAULT_MODEL,
        )

        try:
            import openai

            if is_openai_v1():
                client_params = {
                    "api_key": values["opea_api_key"].get_secret_value(),
                    "base_url": values["opea_api_base"],
                }
                if not values.get("client"):
                    values["client"] = openai.OpenAI(**client_params).chat.completions
                if not values.get("async_client"):
                    values["async_client"] = openai.AsyncOpenAI(
                        **client_params
                    ).chat.completions
            else:
                values["openai_api_base"] = values["opea_api_base"]
                values["openai_api_key"] = values["opea_api_key"].get_secret_value()
                values["client"] = openai.ChatCompletion  # type: ignore[attr-defined]
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )

        return values
