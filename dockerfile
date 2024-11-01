# Use a base image with Python and CUDA support
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        python3-pip \
        python3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*


# Set the working directory to the cloned repo
WORKDIR /workspace/audio_to_audio

COPY . .

# Install Python dependencies from requirements.txt
RUN pip3 install --upgrade pip && \
    pip3 install -r requirements.txt

# Install the nightly version of PyTorch with CUDA support (choose cu121 or cu118 as needed)
RUN pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# Entry point for the container
CMD ["bash"]
