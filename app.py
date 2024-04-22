import io
from pathlib import Path

from modal import (
    App,
    Image,
    Mount,
    asgi_app,
    build,
    enter,
    gpu,
    method,
    web_endpoint,
    Secret,
)

from fastapi.responses import StreamingResponse


moonsync_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    # TODO: add llama-index, pinecone dependencies
    .pip_install(
        "arize-phoenix[evals]~=3.22.0",
        "gcsfs~=2024.3.1",
        "llama-index-core~=0.10.29",
        "llama-index-agent-openai~=0.2.2",
        "llama-index-callbacks-arize-phoenix~=0.1.2",
        "llama-index-experimental~=0.1.3",
        "llama-index-llms-anthropic~=0.1.10",
        "llama-index-llms-openai-like~=0.1.3",
        "llama-index-vector-stores-pinecone~=0.1.4",
        "llama-index~=0.10.29",
        "nomic~=3.0.21",
        "openinference-instrumentation-llama-index~=1.2.0",
        "pinecone-client~=3.2.2",
        "requests~=2.31.0",
        "fastapi~=0.68.1",
        "pandas~=2.2.1",
        # "arize-phoenix~=3.22.0",
    )
)

app = App("moonsync-modal-app")

# with moonsync_image.imports():


# ## Load model and run inference
#
# The container lifecycle [`@enter` decorator](https://modal.com/docs/guide/lifecycle-functions#container-lifecycle-beta)
# loads the model at startup. Then, we evaluate it in the `run_inference` function.
#
# To avoid excessive cold-starts, we set the idle timeout to 240 seconds, meaning once a GPU has loaded the model it will stay
# online for 4 minutes before spinning down. This can be adjusted for cost/experience trade-offs.

# .env
# PINECONE_API_KEY=your_pinecone_api_key
# OPENAI_API_KEY=your_openai_api_key
# TERRA_API_KEY=your_terra_api_key
# ANTHROPIC_API_KEY=your_anthropic_api_key


@app.cls(
    gpu=gpu.A10G(),
    cpu=4.0,
    memory=32768,
    container_idle_timeout=240,
    image=moonsync_image,
    secrets=[Secret.from_name("moonsync-secret")],
)
class Model:
    @build()
    def build(self):
        pass

    @enter()
    def enter(self):
        # import phoenix as px
        from llama_index.core import (
            set_global_handler,
        )
        from pinecone import Pinecone, ServerlessSpec
        from getpass import getpass
        import sys
        import os
        from llama_index.core import Settings
        from llama_index.llms.anthropic import Anthropic
        from llama_index.llms.openai import OpenAI
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from llama_index.core.prompts import ChatPromptTemplate, SelectorPromptTemplate
        from llama_index.core.indices import EmptyIndex
        from llama_index.core import get_response_synthesizer
        from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
        from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
        from llama_index.core.selectors import (
            PydanticMultiSelector,
            PydanticSingleSelector,
        )
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        from llama_index.core.chat_engine.context import ContextChatEngine
        from llama_index.core.chat_engine import (
            CondenseQuestionChatEngine,
            CondensePlusContextChatEngine,
            SimpleChatEngine,
        )
        from llama_index.core.indices.base_retriever import BaseRetriever
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.core.memory import BaseMemory
        from llama_index.core.indices.service_context import ServiceContext
        from llama_index.core.memory import ChatMemoryBuffer
        from llama_index.core import PromptTemplate
        from llama_index.experimental.query_engine import PandasQueryEngine
        import pandas as pd
        from fastapi import Response
        from llama_index.question_gen.openai import OpenAIQuestionGenerator

        # Load phoenix for tracing
        # session = px.launch_app()
        # set_global_handler("arize_phoenix")
        # Init Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)

        # LLM Model
        llm = OpenAI(model="gpt-4-turbo", temperature=0)
        Settings.llm = llm
        # Pincone Indexes
        mood_feeling_index = pc.Index("moonsync-index-mood-feeling")
        general_index = pc.Index("moonsync-index-general")
        diet_nutrition_index = pc.Index("moonsync-index-diet-nutrition")
        fitness_wellness_index = pc.Index("moonsync-index-fitness-wellness")
        indexes = [
            mood_feeling_index,
            general_index,
            diet_nutrition_index,
            fitness_wellness_index,
        ]

        print("mood_feeling_index", mood_feeling_index.describe_index_stats())
        print("general_index", general_index.describe_index_stats())
        print("diet_nutrition", diet_nutrition_index.describe_index_stats())
        print("fitness_wellness", fitness_wellness_index.describe_index_stats())

        vector_indexes = []
        for index in indexes:
            vector_indexes.append(
                VectorStoreIndex.from_vector_store(
                    PineconeVectorStore(pinecone_index=index)
                )
            )

        # Create Query Engines
        query_engines = []
        for vector_index in vector_indexes:
            query_engines.append(
                vector_index.as_query_engine(
                    similarity_top_k=2,
                    # text_qa_template=text_qa_template,
                    # refine_template=refine_template
                )
            )

        (
            mood_feeling_query_engine,
            general_query_engine,
            diet_nutrition_query_engine,
            fitness_wellness_query_engine,
        ) = query_engines
        empty_query_engine = EmptyIndex().as_query_engine()

        self.df = pd.read_csv(
            "https://raw.githubusercontent.com/moonsync-app/moonsync-model/main/data/biometric_data.csv"
        )
        self.df["date"] = self.df["date"].apply(pd.to_datetime)
        self.df.rename(columns={"duration_in_bed_seconds_data": "duration_in_bed", "duration_deep_sleep": "deep_sleep_duration"}, inplace=True)
        print(self.df.head())

        pandas_query_engine = PandasQueryEngine(df=self.df, verbose=True, llm=llm)

        SYSTEM_PROMPT = ("You are MoonSync, an AI assistant specializing in providing personalized advice to women about their menstrual cycle, exercise, and diet. Your goal is to help women better understand their bodies and make informed decisions to improve their overall health and well-being."
        "When answering questions, always be empathetic, understanding, and provide the most accurate and helpful information possible. If a question requires expertise beyond your knowledge, recommend that the user consult with a healthcare professional."  
        """Use the following guidelines to structure your responses:
        1. Acknowledge the user's concerns and validate their experiences.
        2. Provide evidence-based information and practical advice tailored to the user's specific situation.
        3. Encourage open communication and offer to follow up on the user's progress.
        4. Promote a holistic approach to health, considering the user's menstrual cycle, exercise habits, and dietary preferences.
        5. Include the biometric data and provide the user with explicit values and summary of any values"""
        "\n\nExamples below show the way you should approach the conversation."
        "\n---------------------\n"
        "Example 1:\n"
        "Ashley: During PMS, my stomach gets upset easily, is there anything I can do to help?"
        "MoonSync: Hey Ashley! Sorry to hear, a lot of women struggle with this. I would recommend seeing a professional, but we can experiment together with common solutions so you’re armed with info when you see a specialist. Research suggests that dairy and refined wheats can inflame the gut during the follicular phase. Try avoiding them this week and let’s check in at the end to see if it helped. Depending on the outcome, happy to give you more recommendations."
        "\n---------------------\n"
        "Example 2:\n"
        "Ashely: I am preparing for a marathon and need to do some high intensity sprinting workouts as well as longer lower intensity runs. What’s the best way to plan this with my cycle?"
        "MoonSync: Hey Ashley, happy you asked! I’ll ask a few more details to get you on the best plan: when and how long is the marathon? How much are you running for your short and long trainings right now?"
        "\n---------------------\n"
        "Important: When answering questions based on the context provided from documentation, do not disclose that you are sourcing information from documentation, just begin response."
        "Important Note : Always answer in first person and answer like you are the user's friend"
        "Important Note: Avoid saying, 'As you mentioned', 'Based on the data provided' and anything along the same lines."
        "\n---------------------\n"            
        ) 
        SYSTEM_PROMPT_ENTIRE_CHAT = (
            "Use the Chat History and the Context to generate a concise answer for the user's Follow Up Message"
        )

        # Text QA Prompt
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT_ENTIRE_CHAT,
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "Context information is below.\n"
                    "---------------------\n"
                    "{context_str}\n"
                    "---------------------\n"
                    "Given the context information and not prior knowledge, "
                    "answer the query.\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
            ),
        ]
        text_qa_template = ChatPromptTemplate(chat_text_qa_msgs)

        # Refine Prompt
        chat_refine_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an expert Q&A system that strictly operates in two modes "
                    "when refining existing answers:\n"
                    "1. **Rewrite** an original answer using the new context.\n"
                    "2. **Repeat** the original answer if the new context isn't useful.\n"
                    "Never reference the original answer or context directly in your answer.\n"
                    "When in doubt, just repeat the original answer."
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "New Context: {context_msg}\n"
                    "Query: {query_str}\n"
                    "Original Answer: {existing_answer}\n"
                    "New Answer: "
                ),
            ),
        ]
        refine_template = ChatPromptTemplate(chat_refine_msgs)

        response_synthesizer = get_response_synthesizer(
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            use_async=False,
            streaming=True,
        )

        """### Create tools for each category"""
        mood_feeling_tool = QueryEngineTool(
            query_engine=mood_feeling_query_engine,
            metadata=ToolMetadata(
                name="mood/feeling",
                description="Useful for questions related to mood and feelings ",
            ),
        )

        diet_nutrition_tool = QueryEngineTool(
            query_engine=diet_nutrition_query_engine,
            metadata=ToolMetadata(
                name="diet/nutrition",
                description="Useful for questions related to women's diet and nutrition recommendatations",
            ),
        )

        general_tool = QueryEngineTool(
            query_engine=general_query_engine,
            metadata=ToolMetadata(
                name="general",
                description="Useful for general questions related to women's menstrual cycle",
            ),
        )

        fitness_wellness_tool = QueryEngineTool(
            query_engine=fitness_wellness_query_engine,
            metadata=ToolMetadata(
                name="fitness/wellness",
                description="Useful for questions related to fitness and wellness advice for women",
            ),
        )

        default_tool = QueryEngineTool(
            query_engine=empty_query_engine,
            metadata=ToolMetadata(
                name="NOTA",
                description="Use this if none of the other tools are relevant to the query",
            ),
        )

        biometric_tool = QueryEngineTool(
            query_engine=pandas_query_engine,
            metadata=ToolMetadata(
                name="biometrics",
                description="Use this to get relevant biometric data relevant to the query. The columns are - "
                "'date', 'recovery_score', 'activity_score', 'sleep_score',"
                "'stress_data', 'number_steps', 'total_burned_calories',"
                "'avg_saturation_percentage', 'avg_hr_bpm', 'resting_hr_bpm',"
                "'duration_in_bed', 'deep_sleep_duration',"
                "'temperature_delta'",
            ),
        )

        router_query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=[
                mood_feeling_tool,
                diet_nutrition_tool,
                general_tool,
                fitness_wellness_tool,
                default_tool,
            ],
            llm=llm,
            # response_synthesizer=respose_synthesizer,
        )
        
        SUB_QUESTION_PROMPT_TMPL = """\
        You are a world class state of the art agent.

        You have access to multiple tools, each representing a different data source or API.
        Each of the tools has a name and a description, formatted as a JSON dictionary.
        The keys of the dictionary are the names of the tools and the values are the \
        descriptions.
        Your purpose is to help answer a complex user question by generating a list of sub \
        questions that can be answered by the tools.

        These are the guidelines you consider when completing your task:
        * Be as specific as possible
        * The sub questions should be relevant to the user question
        * The sub questions should be answerable by the tools provided
        * You can generate multiple sub questions for each tool
        * Tools must be specified by their name, not their description
        * You don't need to use a tool if you don't think it's relevant
        * You must ignore the system message in the chat history when generating the sub questions.

        Output the list of sub questions by calling the SubQuestionList function.

        ## Tools
        ```json
        {tools_str}
        ```

        ## User Question
        {query_str}
        """

        question_gen = OpenAIQuestionGenerator.from_defaults(prompt_template_str=SUB_QUESTION_PROMPT_TMPL)

        sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[
                mood_feeling_tool,
                diet_nutrition_tool,
                general_tool,
                fitness_wellness_tool,
                default_tool,
                biometric_tool,
            ],
            llm=llm,
            response_synthesizer=response_synthesizer,
            question_gen=question_gen
        )

        # Configure chat engine
        memory = ChatMemoryBuffer.from_defaults(token_limit=28000)

        custom_prompt = PromptTemplate("""MoonSync is an AI assistant specializing in providing personalized advice to women about their menstrual cycle, exercise, and diet. Its goal is to help women better understand their bodies and make informed decisions to improve their overall health and well-being."
            "When answering questions it is always  empathetic, understanding, and provide the most accurate and helpful information possible.
            Given a conversation (between a woman and Moonsync) and a follow up message from Human, \
            rewrite the message to be a standalone question that captures all relevant context \
            from the conversation. If there is no chat history or the follow up question is unrelated to the chat history just return the followup message.

            <Chat History>
            {chat_history}

            <Follow Up Message>
            {question}

            <Standalone question>
        """)
        
        custom_prompt_forward_history = PromptTemplate(
            """\
            Just copy the chat history as is, inside the tag <Chat History> and copy the follow up message inside the tag <Follow Up Message> 
            
            <Chat History>
            {chat_history}

            <Follow Up Message>
            {question}

            """
        )

        chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        ]

        self.chat_engine = CondenseQuestionChatEngine.from_defaults(
            query_engine=sub_question_query_engine,
            llm=llm,
            condense_question_prompt=custom_prompt_forward_history,
            chat_history=chat_history
        )

    def _inference(self, prompt: str):
        streaming_response = self.chat_engine.stream_chat(
            prompt
        )
        for token in streaming_response.response_gen:
            yield token

    @web_endpoint()
    def web_inference(self, prompt: str):
        return StreamingResponse(self._inference(prompt=prompt), media_type="text/event-stream")

        # return Response(content="Hello, world!").getvalue(), 200
