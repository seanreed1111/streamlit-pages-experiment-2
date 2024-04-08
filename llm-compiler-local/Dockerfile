# For more information, please refer to https://aka.ms/vscode-docker-python
# For Streamlit see https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM ubuntu:22.04
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
WORKDIR /app
RUN --mount=type=cache, target=/var/cache/apt \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    python3-all \
    python3-pip \
    python-is-python3 \
    unixodbc-dev

COPY . .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf ~/.cache/pip && \
    ACCEPT_EULA=Y apt-get install -y /app/scripts/msodbcsql18_18.3.2.1-1_amd64.deb && \
    ACCEPT_EULA=Y apt-get install -y /app/scripts/mssql-tools18_18.2.1.1-1_amd64.deb && \
    echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc && \
    . ~/.bashrc
