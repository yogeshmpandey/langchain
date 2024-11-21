from typing import Any, Dict, Optional
from langchain_community.llms.openai import BaseOpenAI
from langchain_community.utils.openai import is_openai_v1


class OPEALLM(BaseOpenAI):
    """OpenAI-compatible API client for OPEA supported LLMs"""
    model_name: str
    api_key: Optional[str] = ""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return False

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        """Get the parameters used to invoke the model."""
        params: Dict[str, Any] = {
            "model": self.model_name,
            **self._default_params,
            "logit_bias": None,
        }
        if not is_openai_v1():
            print("here")
            params.update(
                {
                    "api_key": self.api_key,
                    "api_base": self.base_url,
                }
            )

        return params

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "opea"