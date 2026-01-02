from telemetry import tracer, meter  # MUST be first

from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import uuid

MODEL_NAME = "tinyllama"
MODEL_DEVICE = "cpu"
MODEL_DTYPE = "float32"

model_path = r"C:\Users\AnkushSethi\Desktop\Projects\vscode-workspace\tinyllama"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    use_fast=True,
    fix_mistral_regex=True,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    device_map={"": "cpu"},
)
model = torch.compile(model)

app = FastAPI()

# -------------------------------------------------------------------
# METRICS
# -------------------------------------------------------------------

# Last observed latency (Gauge)
span_latency_gauge = meter.create_gauge(
    name="llm_span_latency_ms",
    description="Last observed latency per span",
    unit="ms",
)

# Cumulative latency (Counter)
span_latency_sum = meter.create_counter(
    name="llm_span_latency_ms_total",
    description="Total latency accumulated per span",
    unit="ms",
)

# Invocation count (Counter)
span_latency_count = meter.create_counter(
    name="llm_span_latency_count",
    description="Number of span executions",
)

# Tokens
input_token_counter = meter.create_counter(
    "llm_input_tokens_total",
    description="Total number of LLM input tokens",
    unit="tokens",
)

output_token_counter = meter.create_counter(
    "llm_output_tokens_total",
    description="Total number of LLM output tokens generated",
    unit="tokens",
)

# Errors
error_counter = meter.create_counter(
    "llm_errors_total",
    description="Total number of errors",
    unit="errors",
)

# -------------------------------------------------------------------
# REQUEST SCHEMA
# -------------------------------------------------------------------

class Request(BaseModel):
    prompt: str
    max_tokens: int = 250


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------

def record_span_latency(span_name: str, duration_ms: float):
    attrs = {
        "span_name": span_name,
        "model": MODEL_NAME,
    }

    span_latency_gauge.set(duration_ms, attributes=attrs)
    span_latency_sum.add(duration_ms, attributes=attrs)
    span_latency_count.add(1, attributes=attrs)


# -------------------------------------------------------------------
# ENDPOINT
# -------------------------------------------------------------------

@app.post("/generate")
def generate(req: Request):
    request_id = str(uuid.uuid4())
    request_start = time.time()

    with tracer.start_as_current_span("llm.request"):
        try:
            # ---------------- Tokenize ----------------
            t0 = time.time()
            ids = tokenizer(req.prompt, return_tensors="pt").input_ids
            input_tokens = ids.shape[1]
            t1 = time.time()

            tokenize_ms = (t1 - t0) * 1000
            record_span_latency("tokenize", tokenize_ms)

            input_token_counter.add(
                input_tokens,
                attributes={"model": MODEL_NAME},
            )

            # ---------------- Forward ----------------
            
            t0 = time.time()
            out = model.generate(
                ids,
                max_new_tokens=req.max_tokens,
                do_sample=False,
                num_beams=1,
            )
            t1 = time.time()

            forward_ms = (t1 - t0) * 1000
            record_span_latency("forward", forward_ms)

            total_tokens = out.shape[1]
            output_tokens = total_tokens - input_tokens

            output_token_counter.add(
                output_tokens,
                attributes={"model": MODEL_NAME},
            )

            # ---------------- Decode ----------------
            t0 = time.time()
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            t1 = time.time()

            decode_ms = (t1 - t0) * 1000
            record_span_latency("decode", decode_ms)

            # ---------------- Total Request ----------------
            total_request_ms = (time.time() - request_start) * 1000
            record_span_latency("total_request", total_request_ms)

            return {
                "id": request_id,
                "text": text,
            }

        except Exception as e:
            error_counter.add(
                1,
                attributes={"error_type": type(e).__name__},
            )
            raise
