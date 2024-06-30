from dataclasses import dataclass
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import (
    ChatPromptTemplate,
)

from llama_index.core.indices import EmptyIndex
from sqlalchemy import create_engine, MetaData
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine


from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import (
    SubQuestionQueryEngine,
)

from llama_index.core.llms import ChatMessage, MessageRole
from config.prompts import (
    SYSTEM_PROMPT_ENTIRE_CHAT,
    SOURCE_QA_PROMPT_USER,
    REFINE_SYSTEM_PROMPT,
    REFINE_USER_PROMPT,
    SUB_QUESTION_PROMPT_TMPL,
    FORWARD_PROMPT,
    SYSTEM_PROMPT,
)
from llama_index.core import get_response_synthesizer

from llama_index.core.question_gen.prompts import build_tools_text
from typing import List, Optional, Sequence, cast
from llama_index.core.llms.llm import LLM
from llama_index.core.question_gen.prompts import build_tools_text
from llama_index.core.question_gen.types import (
    SubQuestion,
    SubQuestionList,
)
from llama_index.core.schema import QueryBundle
from llama_index.core.settings import Settings
from llama_index.core.tools.types import ToolMetadata as ToolMetadataType

from llama_index.question_gen.openai import OpenAIQuestionGenerator
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core import PromptTemplate


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

    @staticmethod
    def get_sql_query_engine(
        db_url: str,
        tables: List["str"],
        llm: str,
    ):
        engine = create_engine(db_url, future=True)
        metadata_obj = MetaData()
        metadata_obj.create_all(engine)
        sql_database = SQLDatabase(engine, include_tables=tables)
        sql_query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database,
            tables=tables,
            llm=llm,
        )

        return sql_query_engine

    @staticmethod
    def get_query_engine_tools(tools_data: List[Dict[str, Any]]):
        query_engine_tools = [
            QueryEngineTool(
                query_engine=tool_data["query_engine"],
                metadata=ToolMetadata(
                    name=tool_data["name"],
                    description=tool_data["description"],
                ),
            )
            for tool_data in tools_data
        ]

        return query_engine_tools

    @classmethod
    def get_subquestion_query_engine(
        cls,
        query_engine_tools: List[QueryEngineTool],
        subquestion_llm: Any,
    ):
        question_gen = CustomOpenAIQuestionGenerator.from_defaults(
            prompt_template_str=SUB_QUESTION_PROMPT_TMPL, llm=subquestion_llm
        )
        return SubQuestionQueryEngine.from_defaults(
            query_engine_tools=query_engine_tools,
            response_synthesizer=cls._get_response_synthesizer(),
            question_gen=question_gen,
        )

    def _get_response_synthesizer():
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT_ENTIRE_CHAT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=SOURCE_QA_PROMPT_USER,
            ),
        ]
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

        chat_refine_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=REFINE_SYSTEM_PROMPT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=REFINE_USER_PROMPT,
            ),
        ]
        refine_template = ChatPromptTemplate(chat_refine_msgs)

        return get_response_synthesizer(
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            use_async=False,
            streaming=True,
        )

    @staticmethod
    def get_chat_engine(query_engine, user_info_content):

        prompt_forward_history = PromptTemplate(FORWARD_PROMPT)
        chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=user_info_content,
            ),
        ]
        return CustomCondenseQuestionChatEngine.from_defaults(
            query_engine=query_engine,
            condense_question_prompt=prompt_forward_history,
            chat_history=chat_history,
            verbose=True,
        )


class CustomOpenAIQuestionGenerator(OpenAIQuestionGenerator):
    def generate(
        self, tools: Sequence[ToolMetadataType], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        question_list = cast(
            SubQuestionList,
            self._program(
                query_str=query_str.split("<Follow Up Message>")[1],
                tools_str=tools_str,
            ),
        )
        return question_list.items

    async def agenerate(
        self, tools: Sequence[ToolMetadataType], query: QueryBundle
    ) -> List[SubQuestion]:
        tools_str = build_tools_text(tools)
        query_str = query.query_str
        question_list = cast(
            SubQuestionList,
            await self._program.acall(
                query_str=query_str.split("<Follow Up Message>")[1],
                tools_str=tools_str,
            ),
        )
        assert isinstance(question_list, SubQuestionList)
        return question_list.items


class CustomCondenseQuestionChatEngine(CondenseQuestionChatEngine):
    def _condense_question(
        self, chat_history: List[ChatMessage], last_message: str
    ) -> str:
        chat_str = "<Chat History>\n"

        for message in chat_history:
            role = message.role
            content = message.content
            chat_str += f"{role}: {content}\n"

        chat_str += "<Follow Up Message>\n"
        chat_str += last_message

        return chat_str
