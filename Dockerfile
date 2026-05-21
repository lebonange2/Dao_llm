FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set workspace as the working directory (RunPod persistent volume)
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY main.py /workspace/main.py
COPY app.py  /workspace/app.py
COPY start.sh /workspace/start.sh

RUN chmod +x /workspace/start.sh

# Gradio web UI
EXPOSE 7860

CMD ["/workspace/start.sh"]
