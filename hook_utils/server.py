# server.py
import os, re, io, json, tempfile, uuid, gc
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

import torch
from torch import Tensor
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from transformers import AutoModelForCausalLM, AutoTokenizer
from record_utils import record_activations, convert_to_hooked_model


MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#TEMP_DIR = "/tmp/ajyl"
TEMP_DIR = "/n/netscratch/wattenberg_lab/Lab/ajyl/tmp_acts"
os.makedirs(TEMP_DIR, exist_ok=True)

# ---------- Model Load (once) ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    attn_implementation="eager",
    device_map="auto" if DEVICE == "cuda" else None,
).eval()
convert_to_hooked_model(model)


class PydanticModel(BaseModel):
    tensor_field: torch.Tensor
    model_config = ConfigDict(arbitrary_types_allowed=True)


class RunRequest(BaseModel):
    prompts: Optional[List[str]] = None
    record_module_names: List[str]
    seq_idx: List[int] | int | None = None
    save_logits: bool = False


class RunResponse(BaseModel):
    run_id: str
    activations_path: Optional[str] = None


def _save_npz(
    cache: Dict[str, torch.Tensor], run_id: str, record_module_names: List[str]
) -> Tuple[str, int]:
    path = os.path.join(TEMP_DIR, f"{run_id}_{'_'.join(record_module_names)}.npz")
    with io.BytesIO() as buf:
        # compressed to keep sizes reasonable
        np.savez_compressed(buf, **cache)
        payload = buf.getvalue()
    with open(path, "wb") as f:
        f.write(payload)
    return path, len(payload)


# ---------- FastAPI ----------
app = FastAPI(title="LM + Activation Server", version="0.1")


@app.get("/modules")
def list_modules(pattern: Optional[str] = None):
    items = []
    r = re.compile(pattern) if pattern else None
    for name, module in model.named_modules():
        if name == "":
            continue
        if r is None or r.search(name):
            items.append(name)
    return {"count": len(items), "modules": items}


@app.post("/query_string", response_model=RunResponse)
def query_string(req: RunRequest):
    run_id = str(uuid.uuid4())[:8]
    # prepare inputs
    print(req.prompts)
    enc = tokenizer(req.prompts, return_tensors="pt", padding=True)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    record_module_names = req.record_module_names
    with record_activations(model, record_module_names) as cache:
        with torch.no_grad():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

    if req.seq_idx is not None:
        cache = {k: v[0][:, req.seq_idx].cpu() for k, v in cache.items()}
    else:
        cache = {k: v[0].cpu() for k, v in cache.items()}

    if req.save_logits:
        cache["logits"] = out.logits
    # write activations
    npz_path, size_bytes = _save_npz(cache, run_id, record_module_names)

    return RunResponse(run_id=run_id, activations_path=npz_path)


@app.delete("/purge/{run_id}_{record_module_names}")
def purge(run_id: str, record_module_names: str):
    path = os.path.join(TEMP_DIR, f"{run_id}_{record_module_names}.npz")
    if os.path.exists(path):
        os.remove(path)
        gc.collect()
        return {"ok": True}
    raise HTTPException(404, f"Run {run_id} not found.")
