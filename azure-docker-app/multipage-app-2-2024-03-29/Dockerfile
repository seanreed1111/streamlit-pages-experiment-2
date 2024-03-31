# For more information, please refer to https://aka.ms/vscode-docker-python
# For Streamlit see https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
FROM ubuntu:22.04

# EXPOSE 5000

# deb path https://packages.microsoft.com/ubuntu/22.04/prod/pool/main/m/msodbcsql18/msodbcsql18_18.3.2.1-1_arm64.deb
# # Keeps Python from generating .pyc files in the container
# ENV PYTHONDONTWRITEBYTECODE=1

# # Turns off buffering for easier container logging
# ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    python3-all \
    python3-pip \
    python-is-python3 \
    unixodbc-dev
    
# RUN curl https://packages.microsoft.com/keys/microsoft.asc | tee /etc/apt/trusted.gpg.d/microsoft.asc

# RUN curl https://packages.microsoft.com/config/ubuntu/$(lsb_release -rs)/prod.list | tee /etc/apt/sources.list.d/mssql-release.list


# RUN curl https://packages.microsoft.com/ubuntu/22.04/prod/pool/main/m/msodbcsql18/msodbcsql18_18.3.2.1-1_arm64.deb | tee /debir/msodbcsql18_18.3.2.1-1_arm64.deb
WORKDIR /app
COPY  . .
RUN ACCEPT_EULA=Y apt-get install -y /app/scripts/msodbcsql18_18.3.2.1-1_amd64.deb
RUN ACCEPT_EULA=Y apt-get install -y /app/scripts/mssql-tools18_18.2.1.1-1_amd64.deb

RUN echo 'export PATH="$PATH:/opt/mssql-tools18/bin"' >> ~/.bashrc && \
    . ~/.bashrc

WORKDIR /app
COPY  . .
RUN pip3 install -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Creates a non-root user with an explicit UID and adds permission to access the /app folder
# For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app
# USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
