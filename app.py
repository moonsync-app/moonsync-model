from modal import (
    App,
    Image,
    build,
    enter,
    web_endpoint,
    Secret,
    Volume,
)

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
)

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
    "llama-index-tools-google==0.1.4",
    "llama-index-multi-modal-llms-openai",
    "llama-index-llms-azure-openai",
    "llama-index-multi-modal-llms-azure-openai",
    "langfuse",
    "llama-index-llms-groq"
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
        from llama_index.core.callbacks import CallbackManager
        from langfuse.llama_index import LlamaIndexCallbackHandler
        from llama_index.llms.groq import Groq
        
        self.api_key = os.environ["AZURE_CHAT_API_KEY"]
        self.azure_endpoint = os.environ["AZURE_CHAT_ENDPOINT"]
        
        # LLM Model
        self.llm = AzureOpenAI(
                model="gpt-4-turbo-2024-04-09",
                deployment_name="moonsync-gpt4-turbo",
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,
                api_version="2023-10-01-preview",
                temperature=0.1,
        )         
        self.small_llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
        # llama3-70b-8192
        self.groq = Groq(model="llama3-8b-8192", api_key=os.environ["GROQ_API_KEY"], temperature=0.1)
        self.groq_70b = Groq(model="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"], temperature=0.1)
        self.llm = OpenAI(model="gpt-4-turbo", temperature=0.1)
 
        langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([langfuse_callback_handler])
        
        # Init Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)
        
        # self.anthropic = Anthropic(model="claude-3-opus-20240229", temperature=0.2)
        
        self.pplx_llm = Perplexity(
            api_key=os.environ["PPLX_API_KEY"],
            model=PPLX_MODEL,
            temperature=PPLX_MODEL_TEMPERATURE,
        )

        Settings.llm = self.groq_70b
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
                content=(SOURCE_QA_PROMPT_SYSTEM),
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=(SOURCE_QA_PROMPT_USER),
            ),
        ]
        sources_prompt = ChatPromptTemplate(sources_qa_prompt)

        # Create Query Engines
        query_engines = []
        dashboard_data_query_engines = []
        for vector_index in vector_indexes:
            query_engines.append(
                vector_index.as_query_engine(
                    llm=self.groq,
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
        biometric_data_latest = "/volumes/moonsync/data/biometric_data_latest.csv"
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
            "IMPORTANT - Always get the 'date' column for any query\n"
            "You only have data till the date "
            + str(self.df.iloc[-1]["date"])
            + " Always use the past data to make future predictions\n"
            "Query: {query_str}\n\n"
            "Expression:"
        )

        DEFAULT_PANDAS_PROMPT = PromptTemplate(
            DEFAULT_PANDAS_TMPL, prompt_type=PromptType.PANDAS
        )

        pandas_query_engine = PandasQueryEngine(
            df=self.df, verbose=True, pandas_prompt=DEFAULT_PANDAS_PROMPT, llm=self.llm
        )

        # setup token.json for gcal
        token_json = "/volumes/moonsync/google_credentials/token.json"
        destination_path = "token.json"
        shutil.move(token_json, destination_path)

        self.SYSTEM_PROMPT = SYSTEM_PROMPT

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
                description="Useful for questions related to women's mood and feelings",
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
                description="Use this to get relevant biometric data relevant to the query. Always get the user's menstrual_phase. You have access to the following parameters - "
                "'date', 'recovery_score', 'activity_score', 'sleep_score',"
                "'stress_data', 'number_steps', 'total_burned_calories',"
                "'avg_saturation_percentage', 'avg_hr_bpm', 'resting_hr_bpm',"
                "'duration_in_bed', 'deep_sleep_duration',"
                "'temperature_delta', 'menstrual_phase'",
            ),
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
        * You must not use a tool if you don't think it's relevant
        
        Only Output the list of sub questions by calling the SubQuestionList function, nothing else.

        ## Tools
        ```json
        {tools_str}
        ```

        ## User Question
        {query_str}
        """
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
        from llama_index.core.tools.types import ToolMetadata
        
        class CustomOpenAIQuestionGenerator(OpenAIQuestionGenerator):
            def generate(self, tools: Sequence[ToolMetadata], query: QueryBundle) -> List[SubQuestion]:
                tools_str = build_tools_text(tools)
                query_str = query.query_str
                question_list = cast(
                    SubQuestionList, self._program(query_str=query_str.split("<Follow Up Message>")[1], tools_str=tools_str)
                )
                return question_list.items

            async def agenerate(
                self, tools: Sequence[ToolMetadata], query: QueryBundle
            ) -> List[SubQuestion]:
                tools_str = build_tools_text(tools)
                query_str = query.query_str
                question_list = cast(
                    SubQuestionList,
                    await self._program.acall(query_str=query_str.split("<Follow Up Message>")[1], tools_str=tools_str),
                )
                assert isinstance(question_list, SubQuestionList)
                return question_list.items
            
        question_gen = CustomOpenAIQuestionGenerator.from_defaults(
            prompt_template_str=SUB_QUESTION_PROMPT_TMPL, llm=self.llm
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
            response_synthesizer=response_synthesizer,
            question_gen=question_gen,
        )

        self.custom_prompt_forward_history = PromptTemplate(
            """\
            Just copy the chat history as is, inside the tag <Chat History> and copy the follow up message inside the tag <Follow Up Message>

            <Chat History>
            {chat_history}

            <Follow Up Message>
            {question}

            """
        )

        # Get the current date
        timestamp = datetime.fromisoformat(str(self.df.iloc[-1]["date"]))
        self.current_date = datetime.strptime(datetime.today().strftime('%Y-%m-%d'), '%Y-%m-%d').date()
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

        self.content_template = f"\nImportant information:\nCurrent Mensural Phase: {self.df.iloc[-1]['menstrual_phase']} \nToday's date: {self.current_date} \nDay of the week: {self.day_name} \n Current Location: New York City"

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

        curr_history.append(ChatMessage(role=MessageRole.USER, content=prompt + "\n" + "Give the output in a markdown format and ask the user if they want to schedule the event if relevant to the context."))
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
                content="The timezone is EST and the location is New York City. If there are no attendees, please use a empty list for attendees. Always get the current date using get_date function",
            )
        )
        tool_spec = GoogleCalendarToolSpec()
        self.agent = OpenAIAgent.from_tools(
            tool_spec.to_tool_list(),
            verbose=True,
            llm=OpenAI(model="gpt-4-turbo", temperature=0),
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
        if (isinstance(item['prompt'], list)):
            for value in item['prompt']:
                if value['type'] == 'text':
                    prompt = value['text']
                if value['type'] == 'image_url':
                    image_url = value['image_url']['url']   
        else: 
            prompt = item["prompt"]
            
            
        messages = item["messages"]        
        if(image_url):
            print('IMAGE_URL', image_url[:100])
            id = str(uuid.uuid4())
            extension = image_url.split(',')[0].split('/')[1].split(';')[0]
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(image_url.split(',')[1], "utf-8"))))
            img.save(f"/volumes/moonsync/data/img-{id}.{extension}")
            
            image_doc = ImageReader().load_data(file=f"/volumes/moonsync/data/img-{id}.{extension}")
            print('Image Doc', image_doc)
                        
            api_key = os.environ["AZURE_MULTI_MODAL_API_KEY"]
            azure_endpoint = os.environ["AZURE_MULTI_MODAL_ENDPOINT"]

            azure_openai_mm_llm = AzureOpenAIMultiModal(
                model="gpt-4-vision-preview",
                deployment_name="moonsync-vision",
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version="2023-10-01-preview",
                max_new_tokens=300,
            )
            
            image_response = azure_openai_mm_llm.complete(
                prompt="Describe the images as an alternative text. Give me a title and a description for the image.",
                image_documents=image_doc,
            )
            
            print("Image description", image_response)

        # Get the headers
        city = request.headers.get("x-vercel-ip-city", "NYC")
        region = request.headers.get("x-vercel-ip-country-region", "New York")
        country = request.headers.get("x-vercel-ip-country", "USA")

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
            prompt = prompt + "\n" + "Additional information about the image uploaded \n " +  image_response.text
        
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
