FROM runpod/base:0.6.3-cuda12.8.0

# Python 3.11
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install uv

# Install the whole LTX workspace
RUN cd ltx && uv pip install --system -e .

# Install RunPod
RUN pip install runpod~=1.7.9

CMD ["python", "-u", "handler.py"]