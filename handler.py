import runpod
import json
import base64
import uuid
import time
import requests
import os
import glob
import random

COMFYUI_URL = "http://127.0.0.1:8188"
WORKFLOW_PATH = "/workflow_api.json"
COMFYUI_INPUT_DIR = "/comfyui/input"
COMFYUI_OUTPUT_DIR = "/comfyui/output"

NODE_LOAD_IMAGE = "62"
NODE_POSITIVE_PROMPT = "562:552"
NODE_SEED = "562:550"
NODE_NEGATIVE_PROMPT = "562:553"

# Lightning LoRA output nodes (HIGH and LOW noise paths)
NODE_LIGHTNING_HIGH = "368:359"
NODE_LIGHTNING_LOW = "368:365"

# LoRA paths relative to ComfyUI models/loras/
# Files are at /workspace/ComfyUI/models/loras/wiikoo/wan2.2/NEW/
# LoRA paths are passed directly in the request payload (lora_high, lora_low)
# No hardcoded mapping here — user-api decides which LoRA to use.


def bypass_svi_pro(workflow):
    """Rewire ALL nodes that depend on SVI Pro LoRA nodes to skip them.

    SVI Pro LoRAs (368:366 HIGH, 368:356 LOW) have diff_m keys incompatible
    with standard WAN 2.2 checkpoints -> artifacts.
    Replace every reference to these nodes with their clean upstream inputs:
      368:366 (SVI Pro HIGH) -> replaced by 368:364 (ModelSamplingSD3)
      368:356 (SVI Pro LOW)  -> replaced by 56 (UNETLoader LOW)
    """
    replacements = {
        "368:366": "368:364",
        "368:356": "56",
    }
    for node in workflow.values():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and val[0] in replacements:
                val[0] = replacements[val[0]]


def inject_lora(workflow, upstream_node_id, lora_name, strength, new_node_id):
    """Insert LoraLoaderModelOnly after upstream_node_id, rewiring all consumers."""
    consumers = []
    for nid, node in workflow.items():
        for key, val in node.get("inputs", {}).items():
            if isinstance(val, list) and len(val) == 2 and val[0] == upstream_node_id and val[1] == 0:
                consumers.append((nid, key))

    if not consumers:
        return  # nothing downstream to rewire

    workflow[new_node_id] = {
        "inputs": {
            "lora_name": lora_name,
            "strength_model": strength,
            "model": [upstream_node_id, 0],
        },
        "class_type": "LoraLoaderModelOnly",
        "_meta": {"title": f"Action LoRA [{new_node_id}]"},
    }

    for nid, key in consumers:
        workflow[nid]["inputs"][key] = [new_node_id, 0]


def free_comfy_memory():
    """Release cached VRAM between jobs (prevents OOM from fragmented PyTorch cache)."""
    try:
        requests.post(
            f"{COMFYUI_URL}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=10,
        )
    except Exception:
        pass


def wait_for_comfyui(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def save_input_image(image_base64):
    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(COMFYUI_INPUT_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(image_base64))
    return filename


def queue_prompt(workflow):
    payload = {"prompt": workflow}
    r = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_completion(prompt_id, timeout=600):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            if r.status_code == 200:
                history = r.json()
                if prompt_id in history:
                    status = history[prompt_id].get("status", {})
                    if status.get("status_str") == "success":
                        return history[prompt_id]
                    if status.get("status_str") == "error":
                        msgs = status.get("messages", [])
                        raise Exception(f"ComfyUI error: {msgs}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)
    raise TimeoutError(f"Generation timed out after {timeout}s")


def get_video_output(history):
    outputs = history.get("outputs", {})
    for node_id, node_output in outputs.items():
        if "gifs" in node_output:
            for gif in node_output["gifs"]:
                fullpath = gif.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    return fullpath
                subfolder = gif.get("subfolder", "")
                filepath = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, gif["filename"])
                if os.path.exists(filepath):
                    return filepath
        if "videos" in node_output:
            for vid in node_output["videos"]:
                fullpath = vid.get("fullpath")
                if fullpath and os.path.exists(fullpath):
                    return fullpath
                subfolder = vid.get("subfolder", "")
                filepath = os.path.join(COMFYUI_OUTPUT_DIR, subfolder, vid["filename"])
                if os.path.exists(filepath):
                    return filepath
    videos = glob.glob(os.path.join(COMFYUI_OUTPUT_DIR, "**/*.mp4"), recursive=True)
    if videos:
        return max(videos, key=os.path.getmtime)
    raise FileNotFoundError("No video output found")


def handler(job):
    job_input = job["input"]
    image_base64 = job_input.get("image")
    prompt = job_input.get("prompt", "natural body movement, smooth motion, warm lighting")
    seed = job_input.get("seed", -1)
    negative_prompt = job_input.get("negative_prompt")
    lora_high = job_input.get("lora_high")   # e.g. "wiikoo/wan2.2/NEW/BounceHighWan2_2.safetensors"
    lora_low = job_input.get("lora_low")     # e.g. "wiikoo/wan2.2/NEW/BounceLowWan2_2.safetensors"
    lora_strength = float(job_input.get("lora_strength", 0.85))

    if not image_base64:
        return {"error": "Missing 'image' (base64 encoded)"}

    if not wait_for_comfyui(timeout=30):
        return {"error": "ComfyUI not ready"}

    free_comfy_memory()
    image_filename = save_input_image(image_base64)

    with open(WORKFLOW_PATH, "r") as f:
        workflow = json.load(f)

    # Bypass incompatible SVI Pro LoRA nodes (diff_m keys cause artifacts with standard WAN2.2)
    bypass_svi_pro(workflow)

    # Patch standard inputs
    workflow[NODE_LOAD_IMAGE]["inputs"]["image"] = image_filename
    workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = prompt

    if seed == -1:
        seed = random.randint(0, 2**50)
    workflow[NODE_SEED]["inputs"]["seed"] = seed

    if negative_prompt:
        workflow[NODE_NEGATIVE_PROMPT]["inputs"]["text"] = negative_prompt

    # Inject action LoRA if paths provided
    if lora_high:
        inject_lora(workflow, NODE_LIGHTNING_HIGH, lora_high, lora_strength, "action_lora_high")
    if lora_low:
        inject_lora(workflow, NODE_LIGHTNING_LOW, lora_low, lora_strength, "action_lora_low")

    # Clear old outputs
    for f in glob.glob(os.path.join(COMFYUI_OUTPUT_DIR, "*.mp4")):
        os.remove(f)

    try:
        prompt_id = queue_prompt(workflow)
    except Exception as e:
        return {"error": f"Failed to queue: {str(e)}"}

    try:
        history = wait_for_completion(prompt_id, timeout=600)
    except (TimeoutError, Exception) as e:
        return {"error": str(e)}

    try:
        video_path = get_video_output(history)
        with open(video_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        ext = os.path.splitext(video_path)[1].lstrip(".")
        return {
            "video": video_base64,
            "format": ext,
            "seed": seed,
            "prompt": prompt,
            "lora_high": lora_high,
        }
    except FileNotFoundError as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
