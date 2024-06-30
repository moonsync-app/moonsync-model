from modal import (
    App,
    Image,
    build,
    enter,
    web_endpoint,
    Secret,
    Volume,
)

from os import environ

from fastapi import Request
from fastapi.responses import StreamingResponse
from typing import Dict

from config.base import (
    MODAL_CPU,
    MODAL_MEMORY,
    MODAL_CONTAINER_IDLE_TIMEOUT,
    OPENAI_MODEL,
    OPENAI_MODEL_TEMPERATURE,
    PPLX_MODEL,
    PPLX_MODEL_TEMPERATURE,
    WEATHER_API_URL,
)

from config.prompts import (
    SOURCE_QA_PROMPT_SYSTEM,
    SOURCE_QA_PROMPT_USER,
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_ENTIRE_CHAT,
    SUB_QUESTION_PROMPT_TMPL,
    REFINE_SYSTEM_PROMPT,
    REFINE_USER_PROMPT,
    FORWARD_PROMPT,
)

from model import LLMModel, VectorDB, QueryEngine, CustomCondenseQuestionChatEngine


from utils.biometrics import get_onboarding_data, get_dashboard_data

moonsync_image = Image.debian_slim(python_version="3.10").pip_install(
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
    "llama-index-tools-google",
    "llama-index-multi-modal-llms-openai",
    "llama-index-llms-azure-openai",
    "llama-index-multi-modal-llms-azure-openai",
    "langfuse",
    "llama-index-llms-groq",
    "llama-index-embeddings-azure-openai",
    "supabase",
    "psycopg2-binary",
)

moonsync_volume = Volume.from_name("moonsync")

app = App("moonsync-modal-app")


@app.cls(
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY,
    container_idle_timeout=MODAL_CONTAINER_IDLE_TIMEOUT,
    image=moonsync_image,
    secrets=[Secret.from_name("moonsync-secret")],
    volumes={"/volumes/moonsync": moonsync_volume},
    keep_warm=1,
)
class Model:
    @build()
    def build(self):
        pass

    @enter()
    def enter(self):
        from pinecone import Pinecone
        import os
        import shutil
        from llama_index.core import Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.core import VectorStoreIndex
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from llama_index.core.prompts import (
            ChatPromptTemplate,
            PromptType,
        )
        from llama_index.core.indices import EmptyIndex
        from llama_index.core import get_response_synthesizer
        from llama_index.core.query_engine import (
            SubQuestionQueryEngine,
        )
        from llama_index.core.tools import QueryEngineTool, ToolMetadata
        from llama_index.core.chat_engine import (
            CondenseQuestionChatEngine,
        )
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.core import PromptTemplate
        from llama_index.experimental.query_engine import PandasQueryEngine
        import pandas as pd
        from llama_index.question_gen.openai import OpenAIQuestionGenerator
        from typing import List
        from llama_index.llms.perplexity import Perplexity
        from datetime import datetime
        from llama_index.llms.azure_openai import AzureOpenAI
        from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
        from llama_index.core.callbacks import CallbackManager
        from langfuse.llama_index import LlamaIndexCallbackHandler
        from llama_index.llms.groq import Groq
        from llama_index.llms.openai_like import OpenAILike
        from sqlalchemy import create_engine, MetaData
        from llama_index.core import SQLDatabase
        from llama_index.core.query_engine import NLSQLTableQueryEngine

        # TODO: remove the above once this is complete

        # SUPABASE SETUP
        from supabase import create_client, Client

        self.SYSTEM_PROMPT = SYSTEM_PROMPT

        url: str = os.environ["SUPABASE_URL"]
        key: str = os.environ["SUPABASE_KEY"]

        # TODO: see the use case for this
        self.api_key = os.environ["AZURE_CHAT_API_KEY"]
        self.azure_endpoint = os.environ["AZURE_CHAT_ENDPOINT"]

        self.supabase: Client = create_client(url, key)

        self.groq = LLMModel.get_groq_model(
            model="llama3-8b-8192", api_key=environ["GROQ_API_KEY"]
        )
        self.llm = LLMModel.get_openai_model(
            model="gpt-4-turbo", api_key=environ["OPENAI_API_KEY"]
        )

        self.subquestion_llm = LLMModel.get_openai_like_model(
            model="llama3-8b-8192",
            api_base="https://api.groq.com/openai/v1",
            api_key=environ["GROQ_API_KEY"],
            temperature=0.1,
            is_function_calling_model=True,
            is_chat_model=True,
        )

        self.embed_model = LLMModel.get_azure_embedding_model(
            model="text-embedding-ada-002",
            deployment_name="embedding-model",
            api_key=environ["AZURE_CHAT_API_KEY"],
            azure_endpoint=self.azure_endpoint,
            api_version="2023-10-01-preview",
        )

        self.pplx_llm = LLMModel.get_perplexity_model(
            model=PPLX_MODEL,
            api_key=environ["PPLX_API_KEY"],
            temperature=PPLX_MODEL_TEMPERATURE,
        )

        langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([langfuse_callback_handler])

        # Init Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        vector_db = VectorDB(api_key=api_key)

        Settings.llm = self.llm
        Settings.embed_model = self.embed_model

        # TERRA ENVIRONMENT VARIABLES
        self.TERRA_DEV_ID = os.environ["TERRA_DEV_ID"]
        self.TERRA_API_KEY = os.environ["TERRA_API_KEY"]

        # setup token.json for gcal
        token_json = "/volumes/moonsync/google_credentials/token.json"
        destination_path = "token.json"
        shutil.copy(token_json, destination_path)

        # Pincone Indexes
        vector_indexes = vector_db.get_vector_indexes(
            [
                "moonsync-index-mood-feeling",
                "moonsync-index-general",
                "moonsync-index-diet-nutrition",
                "moonsync-index-fitness-wellness",
            ]
        )

        # Update prompt to include sources
        sources_qa_prompt = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=(SOURCE_QA_PROMPT_SYSTEM),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(SOURCE_QA_PROMPT_USER),
            ),
        ]
        sources_prompt = ChatPromptTemplate(sources_qa_prompt)

        # Create Query Engines

        query_engines = QueryEngine.get_vector_query_engines(
            vector_indexes=vector_indexes,
            llm=self.groq,
            similarity_top_k=2,
            text_qa_template=sources_prompt,
        )
        dashboard_data_query_engines = QueryEngine.get_dashboard_query_engines(
            vector_indexes=vector_indexes,
            llm=self.groq,
            similarity_top_k=2,
        )
        (
            mood_feeling_query_engine,
            general_query_engine,
            diet_nutrition_query_engine,
            fitness_wellness_query_engine,
        ) = query_engines

        self.mood_feeling_qe, _, self.diet_nutrition_qe, self.fitness_wellness_qe = (
            dashboard_data_query_engines
        )

        empty_query_engine = EmptyIndex().as_query_engine()

        # SQL Query Engine
        db_url = environ["SUPABASE_DB_URL"]
        sql_query_engine = QueryEngine.get_sql_query_engine(
            db_url=db_url,
            tables=["user_biometrics"],
            llm=self.llm,
        )

        # Text QA Prompt
        # Get the current date
        # TODO: refactor this
        self.current_date = datetime.strptime(
            datetime.today().strftime("%Y-%m-%d"), "%Y-%m-%d"
        ).date()
        print("Current date: ", self.current_date)
        day_of_week = self.current_date.weekday()
        day_names = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        self.day_name = day_names[day_of_week]
        # TODO: change this to get user specific data - fix self.df.iloc[-1]['menstrual_phase']
        self.content_template = f"\nImportant information to be considered while answering the query:\nCurrent Mensural Phase: Follicular \nToday's date: {self.current_date} \nDay of the week: {self.day_name} \n Current Location: New York City"
        self.phase_info = f"My current mensural phase is: Follicular"

        tools_data = [
            {
                "query_engine": mood_feeling_query_engine,
                "name": "mood/feeling",
                "description": "Useful for questions related to women's mood and feelings",
            },
            {
                "query_engine": diet_nutrition_query_engine,
                "name": "diet/nutrition",
                "description": "Useful for questions related to women's diet and nutrition recommendatations",
            },
            {
                "query_engine": general_query_engine,
                "name": "general",
                "description": "Useful for general questions related to women's menstrual cycle",
            },
            {
                "query_engine": fitness_wellness_query_engine,
                "name": "fitness/wellness",
                "description": "Useful for questions related to fitness and wellness advice for women",
            },
            {
                "query_engine": empty_query_engine,
                "name": "NOTA",
                "description": "Use this if none of the other tools are relevant to the query",
            },
            {
                "query_engine": sql_query_engine,
                "name": "database",
                "description": """Use this to get relevant biometrics (health parameters) data relevant to the query from the 'user_biometrics' SQL table.
            Always use the terra_user_id to filter data for the given user. You have access to the following columns - 
            id, avg_hr_bpm, resting_hr_bpm, duration_in_bed_seconds_data, duration_deep_sleep, temperature_delta, avg_saturation_percentage, recovery_score, activity_score, sleep_score, stress_data, number_steps, total_burned_calories, date, terra_user_id
            """,
            },
        ]

        query_engine_tools = QueryEngine.get_query_engine_tools(tools_data)

        self.sub_question_query_engine = QueryEngine.get_subquestion_query_engine(
            query_engine_tools=query_engine_tools, subquestion_llm=self.subquestion_llm
        )

        self.chat_engine = QueryEngine.get_chat_engine(
            query_engine=self.sub_question_query_engine,
            user_info_content=self.content_template,
        )

    def _inference(self, prompt: str, messages):
        print("Prompt: ", prompt)
        if len(messages) == 0:
            prompt = prompt + "\n" + self.phase_info
        from llama_index.core.llms import ChatMessage, MessageRole

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
                condense_question_prompt=FORWARD_PROMPT,
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
            condense_question_prompt=FORWARD_PROMPT,
            chat_history=self.chat_history,
            verbose=True,
        )

        for token in streaming_response.response_gen:
            yield token

    def _online_inference(self, prompt: str, messages):
        print("Prompt: ", prompt)
        prompt = prompt.replace("@internet", "")
        from llama_index.core.llms import ChatMessage, MessageRole

        # TODO change current location
        curr_history = [
            ChatMessage(role=MessageRole.SYSTEM, content=self.SYSTEM_PROMPT)
        ]
        for message in messages:
            role = message["role"]
            content = message["content"]
            curr_history.append(ChatMessage(role=role, content=content))

        curr_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content=prompt
                + "\n"
                + self.content_template
                + "\nGive the output in a markdown format and ask the user if they want to schedule the event if relevant to the context. STRICTLY FOLLOW - Give a short and concise answer.",
            )
        )
        resp = self.pplx_llm.stream_chat(curr_history)
        for r in resp:
            yield r.delta

    # Event schedule runner
    def _event_schedule_runner(self, prompt: str, messages):
        from llama_index.tools.google import GoogleCalendarToolSpec
        from llama_index.agent.openai import OpenAIAgent
        from llama_index.core.llms import ChatMessage, MessageRole
        from llama_index.llms.openai import OpenAI

        curr_history = []
        for message in messages:
            role = message["role"]
            content = message["content"]
            curr_history.append(ChatMessage(role=role, content=content))

        curr_history.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"Very important - The timezone is EST (UTC−05:00) and the location is New York City. Current Date: {self.current_date}",
            )
        )
        tool_spec = GoogleCalendarToolSpec()
        self.agent = OpenAIAgent.from_tools(
            tool_spec.to_tool_list(),
            verbose=True,
            llm=OpenAI(model="gpt-4-turbo", temperature=0.1),
            chat_history=curr_history,
        )
        response = self.agent.stream_chat(prompt)
        self.agent.reset()
        response_gen = response.response_gen
        for token in response_gen:
            yield token

    @web_endpoint(method="POST")
    def web_inference(self, request: Request, item: Dict):
        import io, base64
        from PIL import Image
        from llama_index.readers.file.image import ImageReader
        from llama_index.multi_modal_llms.openai import OpenAIMultiModal
        import uuid
        from llama_index.multi_modal_llms.azure_openai import AzureOpenAIMultiModal
        import os

        prompt, image_url, image_response = "", "", ""
        if isinstance(item["prompt"], list):
            for value in item["prompt"]:
                if value["type"] == "text":
                    prompt = value["text"]
                if value["type"] == "image_url":
                    image_url = value["image_url"]["url"]
        else:
            prompt = item["prompt"]

        messages = item["messages"]
        if image_url:
            print("IMAGE_URL", image_url[:100])
            id = str(uuid.uuid4())
            extension = image_url.split(",")[0].split("/")[1].split(";")[0]
            img = Image.open(
                io.BytesIO(base64.decodebytes(bytes(image_url.split(",")[1], "utf-8")))
            )
            img.save(f"/volumes/moonsync/data/img-{id}.{extension}")

            image_doc = ImageReader().load_data(
                file=f"/volumes/moonsync/data/img-{id}.{extension}"
            )
            print("Image Doc", image_doc)

            api_key = os.environ["AZURE_MULTI_MODAL_API_KEY"]
            azure_endpoint = os.environ["AZURE_MULTI_MODAL_ENDPOINT"]

            # azure_openai_mm_llm = AzureOpenAIMultiModal(
            #     model="gpt-4-vision-preview",
            #     deployment_name="moonsync-vision",
            #     api_key=api_key,
            #     azure_endpoint=azure_endpoint,
            #     api_version="2023-10-01-preview",
            #     max_new_tokens=300,
            # )

            # image_response = azure_openai_mm_llm.complete(
            #     prompt="Describe the images as an alternative text. Give me a title and a description for the image.",
            #     image_documents=image_doc,
            # )
            openai_mm_llm = OpenAIMultiModal(
                model="gpt-4-vision-preview", max_new_tokens=300
            )

            image_response = openai_mm_llm.complete(
                prompt="Describe the images as an alternative text. Give me a title and a detailed description for the image.",
                image_documents=image_doc,
            )

            print("Image description", image_response)

        # Get the headers
        city = request.headers.get("x-vercel-ip-city", "NYC")
        region = request.headers.get("x-vercel-ip-country-region", "New York")
        country = request.headers.get("x-vercel-ip-country", "USA")

        # Get user terra id
        terra_user_id = item.get("terra_user_id", None)
        if terra_user_id:
            prompt = prompt + f"\nTerra User ID: {terra_user_id}"
        print(f"City: {city}, Region: {region}, Country: {country}")

        if "@internet" in prompt:
            return StreamingResponse(
                self._online_inference(prompt=prompt, messages=messages),
                media_type="text/event-stream",
            )

        if "schedule" in prompt:
            return StreamingResponse(
                self._event_schedule_runner(prompt=prompt, messages=messages),
                media_type="text/event-stream",
            )

        if image_response != "":
            prompt = (
                prompt
                + "\n"
                + "Additional information about the image uploaded \n "
                + image_response.text
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
        base_url = WEATHER_API_URL

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

    @web_endpoint(method="POST", label="init-biometrics")
    def initial_biometric_data_load(self, request: Request, item: Dict):
        user_id = item["user_id"]

        # Get the biometric data
        get_onboarding_data(
            self.TERRA_DEV_ID, self.TERRA_API_KEY, user_id, self.supabase
        )

        response_json = {"status": "complete"}

        return response_json

    @web_endpoint(method="POST", label="dashboard-biometrics")
    def dasboard_biometric_data_load(self, request: Request, item: Dict):
        import os

        user_id = item["user_id"]
        DEV_ID = os.environ["TERRA_DEV_ID"]
        API_KEY = os.environ["TERRA_API_KEY"]

        # TODO - Update menstrual phase
        menstrual_phase = "Follicular"

        sleep, temperature = get_dashboard_data(DEV_ID, API_KEY, user_id)
        print("[DASHBOARD DATA]", sleep, temperature)
        m, _ = divmod(sleep, 60)
        hours, mins = divmod(m, 60)

        sleep = f"{hours} hours {mins} mins"

        weather_data = self._get_weather()

        response_json = {
            "status": "complete",
            "sleep": sleep,
            "body_temperature": round(temperature if temperature else 0 + 98.6, 2),
            "menstrual_phase": menstrual_phase,
        }

        response_json.update(weather_data)

        return response_json
