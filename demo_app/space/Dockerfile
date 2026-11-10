# Hugging Face Space (Docker SDK) for the GeoGuessr model demo.
# deploy_space.py uploads this as the Space's root Dockerfile, next to the app files.
FROM python:3.11-slim

# Spaces run the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HOST=0.0.0.0 \
    PORT=7860 \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

# CPU-only torch keeps the image small; free Spaces have no GPU.
RUN pip install --no-cache-dir --user torch torchvision --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir --user flask transformers pillow numpy

# Bake the pretrained CLIP backbone into the image so the Space starts fast.
RUN python -c "from transformers import CLIPModel; CLIPModel.from_pretrained('openai/clip-vit-base-patch32')"
# Same for the ResNet-101 backbone under the ResNet head.
RUN python -c "import torchvision.models as m; m.resnet101(weights=m.ResNet101_Weights.DEFAULT)"

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "demo_app/app.py"]
