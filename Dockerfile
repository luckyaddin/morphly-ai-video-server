FROM runpod/base:0.7.0-ubuntu2004-cuda1241

# Python 3.11
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install uv

# Install PyTorch with CUDA 12.4
RUN pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install the application dependencies together
RUN pip install --no-cache-dir -r requirements.txt

# Install the LTX packages
RUN cd /app/ltx && \
    uv pip install --system \
      -e packages/ltx-core \
      -e packages/ltx-pipelines

ENTRYPOINT []

CMD ["bash", "-lc", "echo 'Starting Morphly worker'; python --version; python -c 'import torch, scipy, safetensors, einops; print(\"Dependencies OK\"); print(\"CUDA:\", torch.cuda.is_available())'; exec python -u /app/handler.py"]