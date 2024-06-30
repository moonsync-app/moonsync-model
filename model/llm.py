from dataclasses import dataclass
from os import environ

from llama_index.llms.openai import OpenAI
from llama_index.llms.groq import Groq
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.perplexity import Perplexity


@dataclass
class LLMModel:
    """
    Language models used in the application.
    """

    @staticmethod
    def get_groq_model(
        model: str,
        api_key: str,
        temperature: float = 0.1,
    ):
        return Groq(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    @staticmethod
    def get_openai_model(
        model: str,
        api_key: str,
        temperature: float = 0.1,
    ):
        return OpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    @staticmethod
    def get_openai_like_model(
        model: str,
        api_base: str,
        api_key: str,
        is_function_calling_model: bool,
        is_chat_model: bool,
        temperature: float = 0.1,
    ):
        return OpenAILike(
            model=model,
            api_base=api_base,
            api_key=api_key,
            is_function_calling_model=is_function_calling_model,
            is_chat_model=is_chat_model,
            temperature=temperature,
        )

    def get_azure_embedding_model(
        model: str,
        deployment_name: str,
        api_key: str,
        azure_endpoint: str,
        api_version: str,
    ):
        return AzureOpenAIEmbedding(
            model=model,
            deployment_name=deployment_name,
            api_key=api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    def get_perplexity_model(
        model: str,
        api_key: str,
        temperature: float = 0.1,
    ):
        return Perplexity(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )
