from dataclasses import dataclass
from typing import List
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import (
    ChatPromptTemplate,
)

from llama_index.core.indices import EmptyIndex


@dataclass
class QueryEngine:
    """
    Contains the query engines for the query processing pipeline.
    """

    @staticmethod
    def get_vector_query_engines(
        vector_indexes: List[VectorStoreIndex],
        llm: str,
        similarity_top_k: int,
        text_qa_template: ChatPromptTemplate,
    ):
        query_engines = [
            vector_index.as_query_engine(
                llm=llm,
                similarity_top_k=similarity_top_k,
                text_qa_template=text_qa_template,
            )
            for vector_index in vector_indexes
        ]

        return query_engines

    @staticmethod
    def get_dashboard_query_engines(
        vector_indexes: List[VectorStoreIndex],
        llm: str,
        similarity_top_k: int,
    ):
        query_engines = [
            vector_index.as_query_engine(
                llm=llm,
                similarity_top_k=similarity_top_k,
            )
            for vector_index in vector_indexes
        ]

        return query_engines

    @staticmethod
    def get_empty_query_engine():
        return EmptyIndex.as_query_engine()
