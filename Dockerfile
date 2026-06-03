FROM ghcr.io/lambdalabsml/vllm-builder:v0.8.1-cu126-arm64

RUN apt-get update && apt-get install -y psmisc

ENV PATH="$HOME/.elan/bin:$PATH"


# Install your desired python packages
RUN pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    tqdm \
    openai \
    anthropic \
    together \
    google-genai \
    pytest \
    dotenv \
    numpy \
    pandas \
    matplotlib \
    pytz \
    termcolor \
    easydict \
    tabulate \
    pexpect \
    jload \
    loguru
RUN pip install --upgrade transformers