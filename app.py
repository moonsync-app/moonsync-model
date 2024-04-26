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
    Volume,
)

from fastapi import Request
from fastapi.responses import StreamingResponse
from typing import Dict

moonsync_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
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
        "terra-python~=0.0.12",
        "llama-index-llms-perplexity~=0.1.3",
        "llama-index-question-gen-guidance~=0.1.2",
    )
)

volume = Volume("moonsync")
biometric_data_latest = volume.get("/data/biometric_data_latest.csv")

app = App("moonsync-modal-app")


@app.cls(
    # gpu=gpu.A10G(),
    cpu=4.0,
    memory=32768,
    container_idle_timeout=240,
    image=moonsync_image,
    secrets=[Secret.from_name("moonsync-secret")],
    keep_warm=1,
)
class Model:
    @build()
    def build(self):
        pass

    @enter()
    def enter(self):
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
        from llama_index.core.prompts import (
            ChatPromptTemplate,
            SelectorPromptTemplate,
            PromptType,
        )
        from llama_index.core.indices import EmptyIndex
        from llama_index.core import get_response_synthesizer
        from llama_index.core.query_engine import (
            RouterQueryEngine,
            SubQuestionQueryEngine,
        )
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
        from llama_index.core.indices.empty.retrievers import EmptyIndexRetriever
        from llama_index.core.response_synthesizers import BaseSynthesizer
        from llama_index.core.query_engine import CustomQueryEngine
        from typing import List
        from llama_index.llms.perplexity import Perplexity

        # Init Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)

        # LLM Model
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
        # self.llm = Anthropic(model="claude-3-opus-20240229", temperature=0)
        self.pplx_llm = Perplexity(
            api_key=os.environ["PPLX_API_KEY"],
            model="sonar-small-online",
            temperature=0.5,
        )

        Settings.llm = self.llm
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

        # Update prompt to include sources
        sources_qa_prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(
                    "You are an expert Q&A system that is trusted around the world.\n"
                    "Always answer the query using the provided context information with sources, "
                    "and not prior knowledge.\n"
                    "Some rules to follow:\n"
                    "1. IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"
                    "2. Never directly reference the given context in your answer.\n"
                    "3. Avoid statements like 'Based on the context, ...' or "
                    "'The context information ...' or anything along "
                    "those lines."
                ),
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
                    "IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"
                    "Query: {query_str}\n"
                    "Answer: "
                ),
            ),
        ]
        sources_prompt = ChatPromptTemplate(sources_qa_prompt)

        # Create Query Engines
        query_engines = []
        dashboard_data_query_engines = []
        for vector_index in vector_indexes:
            query_engines.append(
                vector_index.as_query_engine(
                    similarity_top_k=2,
                    text_qa_template=sources_prompt,
                    # refine_template=refine_template
                )
            )
            dashboard_data_query_engines.append(
                vector_index.as_query_engine(
                    similarity_top_k=2,
                )
            )

        (
            mood_feeling_query_engine,
            general_query_engine,
            diet_nutrition_query_engine,
            fitness_wellness_query_engine,
        ) = query_engines

        # self.mood_feeling_qe, self.diet_nutrition_qe, self.fitness_wellness_qe = mood_feeling_query_engine, diet_nutrition_query_engine, fitness_wellness_query_engine
        self.mood_feeling_qe, _, self.diet_nutrition_qe, self.fitness_wellness_qe = (
            dashboard_data_query_engines
        )

        empty_query_engine = EmptyIndex().as_query_engine()

        # probably can mounted as modal volume
        self.df = pd.read_csv(biometric_data_latest)
        self.df["date"] = self.df["date"].apply(pd.to_datetime)
        self.df.rename(
            columns={
                "duration_in_bed_seconds_data": "duration_in_bed",
                "duration_deep_sleep": "deep_sleep_duration",
            },
            inplace=True,
        )
        print(self.df.head())

        # Pandas Query Engine
        # TODO update the date
        DEFAULT_PANDAS_TMPL = (
            "You are working with a pandas dataframe in Python.\n"
            "The name of the dataframe is `df`.\n"
            "This is the result of `print(df.head())`:\n"
            "{df_str}\n\n"
            "Follow these instructions:\n"
            "{instruction_str}\n"
            "Scrictly use these columns name - date, recovery_score, activity_score, sleep_score, stress_data, number_steps, total_burned_calories, avg_saturation_percentage, avg_hr_bpm, resting_hr_bpm, duration_in_bed, deep_sleep_duration, temperature_delta, menstrual_phase\n"
            "You only have data till 25th April 2024. Always use the past data to make future predictions\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )

        DEFAULT_PANDAS_PROMPT = PromptTemplate(
            DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
        )

        pandas_query_engine = PandasQueryEngine(
            df=self.df, verbose=True, llm=self.llm, pandas_prompt=DEFAULT_PANDAS_PROMPT
        )

        # # Online PPLX Query Engine
        # empty_index_retriever = EmptyIndexRetriever(index=EmptyIndex())

        # pplx_prompt = PromptTemplate(
        #     "Query: {query_str}\n"
        #     "Answer: "
        # )

        # class PPLXOnlineQueryEngine(CustomQueryEngine):
        #     retriever: BaseRetriever
        #     response_synthesizer: BaseSynthesizer
        #     pplx: OpenAI
        #     qa_prompt: PromptTemplate

        #     def custom_query(self, query_str: str):
        #         response = self.pplx.complete(
        #             pplx_prompt.format(query_str=query_str)
        #         )

        #         return str(response)

        # synthesizer = get_response_synthesizer(response_mode="generation")
        # pplx_query_engine = PPLXOnlineQueryEngine(
        #     retriever=empty_index_retriever, response_synthesizer=synthesizer, pplx=pplx_llm, qa_prompt=pplx_prompt
        # )

        self.SYSTEM_PROMPT = (
            "You are MoonSync, an AI assistant specializing in providing personalized advice to women about their menstrual cycle, exercise, feelings, and diet. Your goal is to help women better understand their bodies and make informed decisions to improve their overall health and well-being."
            "When answering questions, always be empathetic, understanding, and provide the most accurate and helpful information possible. If a question requires expertise beyond your knowledge, recommend that the user consult with a healthcare professional."
            """\nUse the following guidelines to structure your responses:
        1. Acknowledge the user's concerns and validate their experiences.
        2. Provide evidence-based information and practical advice tailored to the user's specific situation.
        3. Encourage open communication and offer to follow up on the user's progress.
        4. Ask follow up questions to get more information from the user.
        5. Include the biometric data and provide the user with explicit values and summary of any values
        6. Answer the query in a natural, friendly, encouraging and human-like manner.
        7. When answering questions based on the context provided, do not disclose that you are refering to context, just begin response.
        8. IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"""
            "\n\nExamples below show the way you should approach the conversation."
            "\n---------------------\n"
            "Example 1:\n"
            "Ashley: During PMS, my stomach gets upset easily, is there anything I can do to help?"
            "MoonSync: Hey Ashley! Sorry to hear, a lot of women struggle with this. I would recommend seeing a professional, but we can experiment together with common solutions so you’re armed with info when you see a specialist. Research suggests that dairy and refined wheats can inflame the gut during the follicular phase. Try avoiding them this week and let’s check in at the end to see if it helped. Depending on the outcome, happy to give you more recommendations."
            "\n---------------------\n"
            "Example 2:\n"
            "Ashely: I am preparing for a marathon and need to do some high intensity sprinting workouts as well as longer lower intensity runs. What’s the best way to plan this with my cycle?"
            "MoonSync: Hey Ashley, happy you asked! I’ll ask a few more details to get you on the best plan: when and how long is the marathon? How much are you running for your short and long trainings right now?\n\n"
        )
        SYSTEM_PROMPT_ENTIRE_CHAT = (
            "Remember you are MoonSync. Use the Chat History and the Context to generate a detailed answer for the user's Follow Up Message.\n"
            "IMPORTANT - You are given the current menstrual phase, date, and location in the context. Use this information if relevant to the user's message\n"
            "IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"
            "IMPORTANT: Avoid saying, 'As you mentioned', 'Based on the data provided' and anything along the same lines.\n"
            "IMPORTANT: Provide specific information and advice based on the context and user's message.\n"
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
                    "IMPORTANT: Include the list of sources of the context in the end of your final answer if you are using that information\n"
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
                    "IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"
                ),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    "IMPORTANT - Include the list of sources of the context in the end of your final answer if you are using that information\n"
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
                # TODO: make a decision to remove the phase from prompt
                description="Use this to get relevant biometric data relevant to the query. Always get the user's menstrual_phase. The columns are - "
                "'date', 'recovery_score', 'activity_score', 'sleep_score',"
                "'stress_data', 'number_steps', 'total_burned_calories',"
                "'avg_saturation_percentage', 'avg_hr_bpm', 'resting_hr_bpm',"
                "'duration_in_bed', 'deep_sleep_duration',"
                "'temperature_delta', 'menstrual_phase'",
            ),
        )

        # online_tool = QueryEngineTool(
        #     query_engine=pplx_query_engine,
        #     metadata=ToolMetadata(
        #         name="internet",
        #         description="Use this to get relevant information from the internet",
        #     ),
        # )

        router_query_engine = RouterQueryEngine(
            selector=LLMMultiSelector.from_defaults(),
            query_engine_tools=[
                mood_feeling_tool,
                diet_nutrition_tool,
                general_tool,
                fitness_wellness_tool,
                default_tool,
            ],
            llm=self.llm,
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
        * Always use the 'biometrics' tool to get the user's menstrual phase
        * Tools must be specified by their name, not their description
        * You must not use a tool if you don't think it's relevant
        * Only use the text after the <Follow Up Message> tag to generate the sub questions

        Output the list of sub questions by calling the SubQuestionList function.

        ## Tools
        ```json
        {tools_str}
        ```

        ## User Question
        {query_str}
        """

        question_gen = OpenAIQuestionGenerator.from_defaults(
            prompt_template_str=SUB_QUESTION_PROMPT_TMPL
        )

        self.sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[
                mood_feeling_tool,
                diet_nutrition_tool,
                general_tool,
                fitness_wellness_tool,
                default_tool,
                biometric_tool,
            ],
            llm=self.llm,
            response_synthesizer=response_synthesizer,
            question_gen=question_gen,
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

        self.custom_prompt_forward_history = PromptTemplate(
            """\
            Just copy the chat history as is, inside the tag <Chat History> and copy the follow up message inside the tag <Follow Up Message>

            <Chat History>
            {chat_history}

            <Follow Up Message>
            {question}

            """
        )

        self.content_template = f"\nImportant information:\nCurrent Mensural Phase: {self.df.iloc[-1]['menstrual_phase']} \nToday's date: {self.df.iloc[-1]['date']} \nDay of the week: Thursday \n Current Location: New York City"

        self.chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=self.content_template,
            ),
        ]

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

        self.chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
            query_engine=self.sub_question_query_engine,
            llm=self.llm,
            condense_question_prompt=self.custom_prompt_forward_history,
            chat_history=self.chat_history,
            verbose=True,
        )

    def _inference(self, prompt: str, messages):
        print("Prompt: ", prompt)
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.core.chat_engine import CondenseQuestionChatEngine
        from typing import List

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

        print("incoming messages", messages)  # role and content
        if len(messages) > 0:
            self.chat_engine.reset()
            curr_history = [
                ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
                ChatMessage(
                    role=MessageRole.USER,
                    content=self.content_template,
                ),
            ]
            for message in messages:
                role = message["role"]
                content = message["content"]
                curr_history.append(ChatMessage(role=role, content=content))

            self.chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
                query_engine=self.sub_question_query_engine,
                llm=self.llm,
                condense_question_prompt=self.custom_prompt_forward_history,
                chat_history=curr_history,
                verbose=True,
            )
        streaming_response = self.chat_engine.stream_chat(prompt)
        self.chat_engine.reset()
        self.chat_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=self.content_template,
            ),
        ]
        self.chat_engine = CustomCondenseQuestionChatEngine.from_defaults(
            query_engine=self.sub_question_query_engine,
            llm=self.llm,
            condense_question_prompt=self.custom_prompt_forward_history,
            chat_history=self.chat_history,
            verbose=True,
        )

        for token in streaming_response.response_gen:
            yield token

    def _online_inference(self, prompt: str, messages):
        print("Prompt: ", prompt)
        prompt = prompt.replace("@internet", "")
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.core.chat_engine import CondenseQuestionChatEngine
        from typing import List

        # TODO change current location
        curr_history = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=self.SYSTEM_PROMPT + self.content_template,
            )
        ]
        for message in messages:
            role = message["role"]
            content = message["content"]
            curr_history.append(ChatMessage(role=role, content=content))

        curr_history.append(ChatMessage(role=MessageRole.USER, content=prompt))
        resp = self.pplx_llm.stream_chat(curr_history)
        for r in resp:
            yield r.delta

    @web_endpoint(method="POST")
    def web_inference(self, request: Request, item: Dict):
        prompt = item["prompt"]
        messages = item["messages"]

        # Get the headers
        city = request.headers.get("x-vercel-ip-city", "Unknown")
        region = request.headers.get("x-vercel-ip-country-region", "Unknown")
        country = request.headers.get("x-vercel-ip-country", "Unknown")

        print(f"City: {city}, Region: {region}, Country: {country}")

        if "@internet" in prompt:
            return StreamingResponse(
                self._online_inference(prompt=prompt, messages=messages),
                media_type="text/event-stream",
            )

        return StreamingResponse(
            self._inference(prompt=prompt, messages=messages),
            media_type="text/event-stream",
        )

    @web_endpoint(method="POST", label="dashboard")
    def dashboard_details(self):
        # prompt = item['test']

        # TODO read phase from dataframe
        phase = self.df.iloc[-1]["menstrual_phase"]
        age = 35
        mood_filler = "on how the user might be feeling today. one point should suggest a way to improve mood"
        nutrition_filler = "on what the user needs to eat. one point should suggest an interesting recipe."
        exercise_filler = "on what exercises the user should perform today"

        PROMPT_TEMPLATE = """
        Current Menstrual Phase: {phase}
        Age: {age}

        Based on the above menstrual phase and other details give me 3 concise points on seperate lines (don't add index number) and nothing else {template}
        Answer in a friendly way and in second person perspective.
        """

        mood_resp = self.mood_feeling_qe.query(
            PROMPT_TEMPLATE.format(phase=phase, age=age, template=mood_filler)
        ).response
        nutrition_resp = self.diet_nutrition_qe.query(
            PROMPT_TEMPLATE.format(phase=phase, age=age, template=nutrition_filler)
        ).response
        exercise_resp = self.fitness_wellness_qe.query(
            PROMPT_TEMPLATE.format(phase=phase, age=age, template=exercise_filler)
        ).response

        response_json = {
            "mood_resp": mood_resp,
            "nutrition_resp": nutrition_resp,
            "exercise_resp": exercise_resp,
        }
        return response_json

    def _get_weather(self):
        import requests
        import os

        api_key = os.environ["WEATHER_API_KEY"]
        base_url = "http://api.weatherapi.com/v1/current.json"

        params = {"key": api_key, "q": "New York City", "aqi": "no"}

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            data = response.json()

            location = data["location"]["name"]
            temp_f = data["current"]["temp_f"]
            condition = data["current"]["condition"]["text"]
            print(f"Location: {location}")
            print(f"Condition: {condition}")
            print(f"Current temperature: {temp_f}°F")
        else:
            print("Error fetching weather data")

        return {"location": location, "condition": condition, "temp_f": temp_f}

    @web_endpoint(method="POST", label="biometrics")
    def biometrics_details(self):
        # TODO read user id from body

        menstrual_phase = self.df.iloc[-1]["menstrual_phase"]
        sleep = self.df.iloc[-1]["duration_in_bed"]
        temperature = 98.6 + self.df.iloc[-1]["temperature_delta"]

        m, s = divmod(sleep, 60)
        hours, mins = divmod(m, 60)

        sleep = f"{hours} hours {mins} mins"

        weather_data = self._get_weather()

        response_json = {
            "menstrual_phase": menstrual_phase,
            "sleep": sleep,
            "body_temperature": round(temperature, 2),
        }
        response_json.update(weather_data)
        return response_json
