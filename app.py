
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
)


moonsync_image = (
    Image.debian_slim(python_version="3.10")
    .apt_install(
        "libglib2.0-0", "libsm6", "libxrender1", "libxext6", "ffmpeg", "libgl1"
    )
    #TODO: add llama-index, pinecone dependencies
    .pip_install(
      "llama-index-vector-stores-pinecone",
      "pinecone-client~=3.0.0",
      "arize-phoenix[evals]",
      "gcsfs",
      "nest-asyncio",
      "llama-index~=0.10.3",
      "openinference-instrumentation-llama-index~=1.0.0",
      "llama-index-callbacks-arize-phoenix~=0.1.2",
      "llama-index-llms-anthropic",
      "llama-index",
      "nomic",
      "llama-index-llms-openai-like",
      "terra-python",
      "llama-index-agent-openai"
    )
)

app = App(
    "moonsync-modal-app"
)

with moonsync_image.imports():
    import phoenix as px
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
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.prompts import ChatPromptTemplate, SelectorPromptTemplate
    from llama_index.core.indices import EmptyIndex
    from llama_index.core import get_response_synthesizer
    from llama_index.core.query_engine import RouterQueryEngine, SubQuestionQueryEngine
    from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector
    from llama_index.core.selectors import (
        PydanticMultiSelector,
        PydanticSingleSelector,
    )

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

@app.cls(gpu=gpu.A10G(), container_idle_timeout=240, image=moonsync_image, secrets=[modal.Secret.from_name("moonsync-secret")])
class Model:
    @build()
    def build(self):
      pass


    @enter()
    def enter(self):

        # Load phoenix for tracing
        session = px.launch_app()
        set_global_handler("arize_phoenix")


        # Init Pinecone
        api_key = os.environ["PINECONE_API_KEY"]
        pc = Pinecone(api_key=api_key)

        # LLM Model
        llm = OpenAI(model="gpt-4-turbo", temperature = 0)
        Settings.llm = llm

        # Pincone Indexes
        mood_feeling_index = pc.Index("moonsync-index-mood-feeling")
        general_index = pc.Index("moonsync-index-general")
        diet_nutrition_index = pc.Index("moonsync-index-diet-nutrition")
        fitness_wellness_index = pc.Index("moonsync-index-fitness-wellness")
        indexes = [mood_feeling_index, general_index, diet_nutrition_index, fitness_wellness_index]


        print('mood_feeling_index', mood_feeling_index.describe_index_stats())
        print('general_index',general_index.describe_index_stats())
        print('diet_nutrition',diet_nutrition_index.describe_index_stats())
        print('fitness_wellness',fitness_wellness_index.describe_index_stats())


        vector_indexes = []
        for index in indexes:
          vector_indexes.append(VectorStoreIndex.from_vector_store(PineconeVectorStore(pinecone_index=index)))

        # Create Query Engines
        query_engines = []
        for vector_index in vector_indexes:
          query_engines.append(vector_index.as_query_engine(similarity_top_k=2,
                                                            # text_qa_template=text_qa_template,
                                                            # refine_template=refine_template
                                                            ))


        mood_feeling_query_engine, general_query_engine, diet_nutrition_query_engine, fitness_wellness_query_engine = query_engines
        empty_query_engine = EmptyIndex().as_query_engine()


        SYSTEM_PROMPT = (
          "You are MoonSync, an AI assistant specializing in providing personalized advice to women about their menstrual cycle, exercise, and diet. Your goal is to help women better understand their bodies and make informed decisions to improve their overall health and well-being."
            "When answering questions, always be empathetic, understanding, and provide the most accurate and helpful information possible. If a question requires expertise beyond your knowledge, recommend that the user consult with a healthcare professional."
            """Use the following guidelines to structure your responses:
            1. Acknowledge the user's concerns and validate their experiences.
            2. Provide evidence-based information and practical advice tailored to the user's specific situation.
            3. Encourage open communication and offer to follow up on the user's progress.
            4. Promote a holistic approach to health, considering the user's menstrual cycle, exercise habits, and dietary preferences."""
            "Examples below show the way you should approach the conversation."
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
        )

        # Text QA Prompt
        chat_text_qa_msgs = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                content=SYSTEM_PROMPT,
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
            streaming=False,
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
                description="Useful for general questions related to women's menstrual cycle"
            ),
        )

        fitness_wellness_tool = QueryEngineTool(
            query_engine=fitness_wellness_query_engine,
            metadata=ToolMetadata(
                name="fitness/wellness",
                description="Useful for questions related to fitness and wellness advice for women"
            ),
        )

        default_tool=QueryEngineTool(
                query_engine=empty_query_engine,
                metadata=ToolMetadata(
                name="NOTA",
                description="Use this if none of the other tools are relevant to the query"
                ),
            )


        router_query_engine = RouterQueryEngine(
          selector=LLMMultiSelector.from_defaults(),
          query_engine_tools=[
              mood_feeling_tool,
              diet_nutrition_tool,
              general_tool,
              fitness_wellness_tool,
              default_tool
          ],
          llm=llm,
        # response_synthesizer=respose_synthesizer,
        )

        sub_question_query_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[
                mood_feeling_tool,
                diet_nutrition_tool,
                general_tool,
                fitness_wellness_tool,
                default_tool
            ],
            llm=llm,
            response_synthesizer=response_synthesizer,
        )



    # def _inference(self, prompt, n_steps=24, high_noise_frac=0.8):
    #     pass


    # @method()
    # def inference(self, prompt, n_steps=24, high_noise_frac=0.8):
    #     return self._inference(
    #         prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
    #     ).getvalue()

    # @web_endpoint()
    # def web_inference(self, prompt, n_steps=24, high_noise_frac=0.8):
    #     return Response(
    #         content=self._inference(
    #             prompt, n_steps=n_steps, high_noise_frac=high_noise_frac
    #         ).getvalue(),
    #         media_type="image/jpeg",
    #     )


# And this is our entrypoint; where the CLI is invoked. Explore CLI options
# with: `modal run stable_diffusion_xl.py --help


# @app.local_entrypoint()
# def main(prompt: str = "Unicorns and leprechauns sign a peace treaty"):
#     image_bytes = Model().inference.remote(prompt)

#     dir = Path("/tmp/stable-diffusion-xl")
#     if not dir.exists():
#         dir.mkdir(exist_ok=True, parents=True)

#     output_path = dir / "output.png"
#     print(f"Saving it to {output_path}")
#     with open(output_path, "wb") as f:
#         f.write(image_bytes)


# ## A user interface
#
# Here we ship a simple web application that exposes a front-end (written in Alpine.js) for
# our backend deployment.
#
# The Model class will serve multiple users from a its own shared pool of warm GPU containers automatically.
#
# We can deploy this with `modal deploy stable_diffusion_xl.py`.

# frontend_path = Path(__file__).parent / "frontend"

# web_image = Image.debian_slim().pip_install("jinja2")


# @app.function(
#     image=web_image,
#     # mounts=[Mount.from_local_dir(frontend_path, remote_path="/assets")],
#     allow_concurrent_inputs=20,
# )
# @asgi_app()
# def ui():
#     import fastapi.staticfiles
#     from fastapi import FastAPI, Request
#     from fastapi.templating import Jinja2Templates

#     web_app = FastAPI()
#     templates = Jinja2Templates(directory="/assets")

#     @web_app.get("/")
#     async def read_root(request: Request):
#         return templates.TemplateResponse(
#             "index.html",
#             {
#                 "request": request,
#                 "inference_url": Model.web_inference.web_url,
#                 "model_name": "Stable Diffusion XL",
#                 "default_prompt": "A cinematic shot of a baby raccoon wearing an intricate italian priest robe.",
#             },
#         )

#     web_app.mount(
#         "/static",
#         fastapi.staticfiles.StaticFiles(directory="/assets"),
#         name="static",
#     )

#     return web_app
