# client.py
import json, requests, numpy as np

BASE = "http://127.0.0.1:8989"


def query_model(prompts, record_module_names):

    payload = {
        "prompts": prompts,
        "record_module_names": record_module_names,
    }
    res = requests.post(f"{BASE}/query_string", json=payload).json()

    with np.load(res["activations_path"], allow_pickle=True) as file_p:
        npz = dict(file_p)

    requests.post(f"{BASE}/purge/{res['run_id']}_model.layers.18.hook_resid_post")
    return npz
