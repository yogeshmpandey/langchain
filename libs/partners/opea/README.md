# langchain-opea

This package contains the LangChain integrations for [OPEA](https://opea.dev/) Microservices.

## Installation and Setup

- Install the LangChain partner package

```bash
pip install -U langchain-opea
```

## Chat Models

`ChatOPEA` class exposes chat models from OPEA.

```python
from langchain_opea import ChatOPEA

llm = ChatOPEA(model="Intel/neural-chat-7b-v3-3",opea_api_key="my_secret_value",opea_api_base="http://localhost:9009/v1")
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`OPEAEmbeddings` class exposes embeddings from OPEA.

```python
from langchain_opea import OPEAEmbeddings

embeddings = OPEAEmbeddings(model="BAAI/bge-large-en-v1.5",opea_api_key="my_secret_value",opea_api_base="http://localhost:6006/v1",)
embeddings.embed_query("What is the meaning of life?")
```

## LLMs

`OPEALLM` class exposes LLMs from OPEA.

```python
from langchain_opea import OPEALLM

llm = OPEALLM(model="Intel/neural-chat-7b-v3-3",opea_api_key="my_secret_value",opea_api_base="http://localhost:9009/v1")
llm.invoke("The meaning of life is")
```
