FROM runpod/base:0.7.0-ubuntu2004-cuda1241

# Python 3.11
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install uv
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu124
RUN pip install scipy
RUN cd ltx && uv pip install --system -e .

# Install RunPod
RUN pip install runpod~=1.7.9

ENTRYPOINT []

CMD ["bash", "-lc", "echo 'Starting Morphly worker'; python --version; which python; ls -la /app; exec python -u handler.py"]