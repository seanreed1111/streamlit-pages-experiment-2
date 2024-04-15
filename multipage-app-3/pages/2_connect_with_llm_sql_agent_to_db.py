# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

# https://python.langchain.com/docs/expression_language/cookbook/sql_db
import json
import os
import sys

# import sqlalchemy
# from sqlalchemy import create_engine
# import urllib
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent
# from langchain_core.callbacks import Callbacks
# from langchain.agents.agent_types import AgentType
# from langchain.schema import ChatMessage
# from langchain.storage import InMemoryStore
# from langchain.storage import LocalFileStore
# from langchain.storage import LocalFileStore
# # Instantiate the LocalFileStore with the root path
# file_store = LocalFileStore("/path/to/root")
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler

if "multipage-app-3" not in sys.path:
    sys.path.append("../src")  # needed to get the  src imports to run

from src.sql_agent_prompt import NEW_SQL_PREFIX
# from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from loguru import logger


def logger_setup():
    log_dir = Path.home() / "PythonProjects" / "logs" / "multipage-app-3" 
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file_name = Path(__file__).stem + ".log"
    log_file_path = log_dir / log_file_name
    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        log_file_path,
        level=log_level,
        format=log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )


logger_setup()



LANGCHAIN_PROJECT = "Multipage App Chat With SQL Agent WAB DB"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")


with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"
        )
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["agent_model_name"] = os.getenv("MODEL_NAME")
        st.session_state["agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME"
        )
    st.info(f"Now using {llm_choice_radio} as the underlying agent llm.")

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio}"

with st.spinner("Setting up agent...please wait"):
    CONFIG_DIR_PATH = st.session_state["config_dir_path"]
    try:
        db = st.session_state["db"]
    except Exception as e:
        st.error(e)
        st.error("Please go back to main app page and connect to the WAB database")
        st.stop()

    TEMPERATURE = 0.05

    llm_config = {
        "llm-temperature": TEMPERATURE,
        "request_timeout": 120,
        "verbose": True,
        "model_name": st.session_state["agent_model_name"],
    }

    logger.info(f"\nllm-config = {json.dumps(llm_config)}")
    llm = AzureChatOpenAI(
        temperature=TEMPERATURE,
        streaming=True,
        azure_deployment=st.session_state["agent_deployment_name"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
        model_name=st.session_state["agent_model_name"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        request_timeout=120,
        verbose=True,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        prefix=NEW_SQL_PREFIX,
        suffix=None,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=15,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    st.success("Agent setup done!")

if (
    "llm_sql_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_sql_agent_messages
):
    st.session_state["llm_sql_agent_messages"] = [
        {"role": "assistant", "content": "How can I help you?"}
    ]

for msg in st.session_state.llm_sql_agent_messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_sql_agent_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    logger.info(f"{prompt=}")

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])
        st.session_state.llm_sql_agent_messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
        logger.info("response: " + str(response))
