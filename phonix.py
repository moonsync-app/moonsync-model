from modal import App, Image, web_endpoint, forward

image = Image.from_registry(
    "arizephoenix/phoenix:version-2.9.3"
)
app = App("phoenix", image=image)

# app = App(
#     "phoenix",
#     image=Image.debian_slim(python_version="3.10")
#     .pip_install("arize-phoenix[evals]"),
# )

@app.function()
@web_endpoint(method="GET")
def f():
    import subprocess
    with forward(6006) as tunnel:
      print("Tunneling to Phoenix", tunnel.url)
      # subprocess.run(["docker run -p 6006:6006 arizephoenix/phoenix:version-2.9.3"])
      subprocess.run(["python3", "-m", "phoenix.server.main --port 6006 --host 0.0.0.0 serve"])
