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


def wait_for_comfyui(timeout=120):
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                return True
        except:
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
    prompt = job_input.get("prompt", "She slowly moves, natural body movement, warm soft lighting")
    seed = job_input.get("seed", -1)
    negative_prompt = job_input.get("negative_prompt", None)

    if not image_base64:
        return {"error": "Missing 'image' (base64 encoded)"}

    if not wait_for_comfyui(timeout=30):
        return {"error": "ComfyUI not ready"}

    image_filename = save_input_image(image_base64)

    with open(WORKFLOW_PATH, "r") as f:
        workflow = json.load(f)

    workflow[NODE_LOAD_IMAGE]["inputs"]["image"] = image_filename
    workflow[NODE_POSITIVE_PROMPT]["inputs"]["text"] = prompt

    if seed == -1:
        seed = random.randint(0, 2**50)
    workflow[NODE_SEED]["inputs"]["seed"] = seed

    if negative_prompt:
        workflow["562:553"]["inputs"]["text"] = negative_prompt

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
            "prompt": prompt
        }
    except FileNotFoundError as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
