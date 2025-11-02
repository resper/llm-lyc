# CUDA Runtime (Ubuntu 22.04) – stabiler als latest
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Systempakete für OCR & PDF -> Bild
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv \
    tesseract-ocr poppler-utils libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Python-Abhängigkeiten
# - torch/cu121 + transformers + tokenizers
# - pypdf: Text aus "echten" PDFs
# - pdf2image + pytesseract: OCR für bildbasierte PDFs (braucht poppler-utils)
# - pillow, numpy: Hilfs libs
RUN pip3 install --no-cache-dir \
    "torch==2.4.1+cu121" "torchvision==0.19.1+cu121" \
      --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install --no-cache-dir \
      transformers==4.45.2 accelerate==1.0.1 safetensors==0.4.5 \
      pypdf==5.1.0 pdf2image==1.17.0 pytesseract==0.3.13 \
      pillow==10.4.0 numpy==2.1.3

# App kopieren
WORKDIR /app
COPY main.py /app/main.py

# Standard: auf GPU 0 zugreifen; Lyceum weist idR eine GPU zu
ENV CUDA_VISIBLE_DEVICES=0
# Entry: pdf + prompt werden als Argumente übergeben
ENTRYPOINT ["python3", "/app/main.py"]
