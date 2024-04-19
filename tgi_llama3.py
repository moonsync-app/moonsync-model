#Meta-Llama-3-70B-Instruct is gated model and requires access on hf first to be able to successfully run this
import os
import subprocess
from modal import Image, Secret, Stub, gpu, web_server

MODEL_DIR = "/model"
DOCKER_IMAGE = "ghcr.io/huggingface/text-generation-inference:1.4"
PORT = 8000
MODEL_ID = "meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_REVISION = "81ca4500337d94476bda61d84f0c93af67e4495f"
# Add `["--quantize", "gptq"]` for TheBloke GPTQ models.
LAUNCH_FLAGS = [
    "--model-id",
    MODEL_ID,
    "--port",
    "8000",
    "--revision",
    MODEL_REVISION,
]

def download_model():
    subprocess.run(
         [
            "text-generation-server",
            "download-weights",
            MODEL_ID,
            "--revision",
            MODEL_REVISION,
        ],
        env={
            **os.environ,
            "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
        },
        check=True,
    )
    
GPU_CONFIG = gpu.H100(count=2)  # 2 H100s

stub = Stub("llama3-70b-instruct")

tgi_image = (
    Image.from_registry(DOCKER_IMAGE, add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .run_function(download_model, timeout=60 * 60, secrets=[Secret.from_name("huggingface-secret")])
    .pip_install("text-generation")
)

@stub.function(
    secrets=[Secret.from_name("huggingface-secret")],
    gpu=GPU_CONFIG,
    allow_concurrent_inputs=15,
    container_idle_timeout=60 * 10,
    timeout=60 * 60,
    image=tgi_image,
    keep_warm=1
)
@web_server(port=PORT, startup_timeout=600)
def run_server():
    cmd = ["text-generation-launcher"]  + LAUNCH_FLAGS
    print("cmd", cmd)
    subprocess.Popen(cmd, 
            env={
                **os.environ,
                "HUGGING_FACE_HUB_TOKEN": os.environ["HF_TOKEN"],
            }
    )
    
# Command to deploy LLAMA-3-70B-Instruct model on Modal
# modal deploy tgi_llama3.py



