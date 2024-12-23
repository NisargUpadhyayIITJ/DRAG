# Use a slim base image for Python 3.10
FROM python:3.10.15-slim

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . .

# Define arguments and environment variables for API keys
ARG GUARDRAILS_API_KEY
ENV GUARDRAILS_API_KEY=${GUARDRAILS_API_KEY}

ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

ARG OPENAI_API_KEY_GHAR
ENV OPENAI_API_KEY_GHAR=${OPENAI_API_KEY_GHAR}

ARG JINA_API_KEY
ENV JINA_API_KEY=${JINA_API_KEY}

# Update system and install required utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install virtual environment tool and create a virtual environment
RUN python3 -m pip install --no-cache-dir uv virtualenv \
    && virtualenv /opt/backend_env --python=3.10

# Activate virtual environment and install dependencies
ENV PATH="/opt/backend_env/bin:$PATH"
COPY requirements.txt .

RUN pip install uv
RUN uv pip sync requirements.txt

# Log environment variable values for verification (optional)
RUN echo "Your Guardrails token: $GUARDRAILS_API_KEY" && \
    echo "OPENAI_API_KEY: $OPENAI_API_KEY" && \
    echo "OPENAI_API_KEY_GHAR: $OPENAI_API_KEY_GHAR" && \
    echo "JINA_API_KEY: $JINA_API_KEY"

# Configure Guardrails
RUN guardrails configure --enable-metrics --enable-remote-inferencing --token $GUARDRAILS_API_KEY

# Clone and move necessary repositories for Guardrails hub functionality
RUN git clone https://github.com/guardrails-ai/toxic_language.git && \
    mkdir -p /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails && \
    mv toxic_language/ /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails/

RUN git clone https://github.com/guardrails-ai/regex_match.git && \
    mv regex_match/ /opt/backend_env/lib/python3.10/site-packages/guardrails/hub/guardrails/

# Ensure CUDA dependencies are properly handled
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcuda1 libcudnn8 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Expose necessary ports
EXPOSE 8001
EXPOSE 7865
EXPOSE 7866

# Run the application using Uvicorn
CMD ["uvicorn", "fast_api_serverPathway:app", "--port", "8001", "--host", "127.0.0.1"]
