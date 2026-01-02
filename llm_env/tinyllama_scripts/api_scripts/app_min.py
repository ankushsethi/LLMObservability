from telemetry import trace  # MUST be first

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry import trace as otel_trace, metrics
import torch

# ---- Model metadata ----

MODEL_NAME = "tinyllama"
MODEL_BACKEND = "transformers"
MODEL_DEVICE = "cpu"
MODEL_DTYPE = "float32"

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\tinyllama"

# ---- Tokenizer / Model ----

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    fix_mistral_regex=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"}
)

model = torch.compile(model)

# ---- FastAPI ----

app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

tracer = otel_trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# ---- Metrics ----

output_token_counter = meter.create_counter(
    name="llm_output_tokens_total",
    description="Total number of LLM output tokens generated",
    unit="tokens"
)

# ---- Request schema ----

class Request(BaseModel):
    prompt: str
    max_tokens: int = 250

# ---- Endpoint ----

@app.post("/generate")
def generate(req: Request):
    with tracer.start_as_current_span("llm.request") as span:
        span.set_attribute("llm.model.name", MODEL_NAME)
        span.set_attribute("llm.model.backend", MODEL_BACKEND)
        span.set_attribute("llm.model.device", MODEL_DEVICE)
        span.set_attribute("llm.model.dtype", MODEL_DTYPE)
        span.set_attribute("llm.max_output_tokens", req.max_tokens)

        # ---- Tokenization ----
        with tracer.start_as_current_span("llm.tokenize"):
            ids = tokenizer(req.prompt, return_tensors="pt").input_ids
            input_tokens = ids.shape[1]

        span.set_attribute("llm.input_tokens", input_tokens)

        # ---- Inference ----
        with tracer.start_as_current_span("llm.forward") as forward_span:
            out = model.generate(
                ids,
                max_new_tokens=req.max_tokens,
                do_sample=False,
                num_beams=1
            )

            total_tokens = out.shape[1]
            output_tokens = total_tokens - input_tokens

            forward_span.set_attribute("llm.output_tokens", output_tokens)
            forward_span.set_attribute("llm.total_tokens", total_tokens)

            output_token_counter.add(
                output_tokens,
                attributes={
                    "llm.model.name": MODEL_NAME
                }
            )

        # ---- Decode ----
        with tracer.start_as_current_span("llm.decode"):
            text = tokenizer.decode(out[0], skip_special_tokens=True)

        return {"text": text}
