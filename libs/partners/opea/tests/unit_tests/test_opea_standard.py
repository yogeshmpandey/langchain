"""Standard LangChain interface tests"""

from typing import Tuple, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_opea import ChatOPEA


OPEA_API_BASE="http://localhost:9009/v1"
OPEA_API_KEY="my_secret_value"
MODEL_NAME="Intel/neural-chat-7b-v3-3"


class TestOPEAStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatOPEA

    @property
    def chat_model_params(self) -> dict:
        return {
        "opea_api_base" : OPEA_API_BASE,
        "opea_api_key" : OPEA_API_KEY,
        "model_name" : MODEL_NAME,
        }

