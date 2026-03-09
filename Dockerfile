FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git wget ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install --no-cache-dir runpod requests

RUN pip3 install --no-cache-dir sageattention triton

RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui
RUN cd /comfyui && pip3 install --no-cache-dir -r requirements.txt

RUN cd /comfyui/custom_nodes && \
    git clone https://github.com/kijai/ComfyUI-KJNodes && \
    git clone https://github.com/rgthree/rgthree-comfy && \
    git clone https://github.com/ashtar1984/comfyui-find-perfect-resolution && \
    git clone https://github.com/city96/ComfyUI-GGUF && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/Fannovel16/ComfyUI-Frame-Interpolation

RUN cd /comfyui/custom_nodes/ComfyUI-KJNodes && \
    pip3 install --no-cache-dir -r requirements.txt || true
RUN pip3 install --no-cache-dir gguf imageio-ffmpeg

COPY handler.py /handler.py
COPY workflow_api.json /workflow_api.json
COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
