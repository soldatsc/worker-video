#!/bin/bash
set -e

VOLUME="/runpod-volume"
MODELS="/comfyui/models"

echo "=== SVI Video Worker ==="

# Link models from Network Volume
if [ -d "$VOLUME/ComfyUI/models" ]; then
    echo "Linking models from Network Volume..."
    for dir in unet clip vae loras upscale_models; do
        if [ -d "$VOLUME/ComfyUI/models/$dir" ]; then
            rm -rf "$MODELS/$dir"
            ln -sf "$VOLUME/ComfyUI/models/$dir" "$MODELS/$dir"
            echo "  $dir -> volume"
        fi
    done
else
    echo "ERROR: No models at $VOLUME/ComfyUI/models/"
    ls -la $VOLUME/ 2>/dev/null || echo "Volume not mounted!"
fi

# Verify models
echo "=== Model check ==="
ls -lh $MODELS/unet/wan2.2/ 2>/dev/null | head -3 || echo "NO UNET!"
ls -lh $MODELS/clip/*.safetensors 2>/dev/null | head -3 || echo "NO CLIP!"
ls -lh $MODELS/vae/*.safetensors 2>/dev/null | head -3 || echo "NO VAE!"

# Start ComfyUI
echo "=== Starting ComfyUI ==="
SAGE_FLAG=""
if [ "${DISABLE_SAGE_ATTENTION}" != "1" ]; then
    SAGE_FLAG="--use-sage-attention"
    echo "SageAttention: ON"
else
    echo "SageAttention: OFF"
fi

cd /comfyui
python3 main.py --listen 127.0.0.1 --port 8188 $SAGE_FLAG &
COMFY_PID=$!

# Wait for ComfyUI
echo "Waiting for ComfyUI..."
for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
        echo "ComfyUI ready! (${i}s)"
        break
    fi
    sleep 2
done

# Start handler
echo "=== Starting RunPod Handler ==="
python3 /handler.py
