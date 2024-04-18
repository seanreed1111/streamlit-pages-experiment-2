import json
import os
import urllib
from io import StringIO
from pathlib import Path
from pprint import pprint

import folium
import geopandas
import geopy

# import ipython
import matplotlib
import pandas as pd
import plotly
import pyodbc
import seaborn
import streamlit as st
import streamlit_folium
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    Tool,
    create_openai_functions_agent,
    create_react_agent,
)
from langchain.agents.agent_types import AgentType
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.output_parsers import OutputFixingParser


from langchain.schema import ChatMessage
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI
from loguru import logger
from mpl_toolkits.basemap import Basemap

LANGCHAIN_PROJECT = "Multipage App LLM CSV Agent"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

with st.sidebar:
    llm_choice_radio_python_agent = st.radio(
        "Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"]
    )
    if llm_choice_radio_python_agent == "GPT-3.5-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["python_agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"
        )
    elif llm_choice_radio_python_agent == "GPT-4-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME")
        st.session_state["python_agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME"
        )
    st.info(
        f"Now using {llm_choice_radio_python_agent} as the underlying python agent llm."
    )

os.environ["LANGCHAIN_PROJECT"] = (
    f"{LANGCHAIN_PROJECT} with {llm_choice_radio_python_agent}"
)

if "config_dir_path" not in st.session_state:
    st.info("please ensure that you've completed loading the main (app) page")
    st.stop()

# upload file
# see also langchain.storage for storage https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.storage
with st.sidebar:
    st.markdown("### UPLOAD FILE")
    file_uploader_radio = st.radio("Choose one", ["no file needed", "upload file"])
    if file_uploader_radio == "upload file":
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file:
            st.session_state["filename"] = uploaded_file.name
            st.session_state["uploaded_file"] = uploaded_file

            st.session_state["uploaded_file_exists"] = True
            if uploaded_file.name.endswith("csv"):
                st.session_state["csv_exists"] = True
            st.success(f"file {uploaded_file.name} uploaded successfully")

    elif file_uploader_radio == "no file needed":
        st.session_state["filename"] = None
        st.session_state["uploaded_file"] = None
        st.session_state["uploaded_file_exists"] = False
        st.session_state["csv_exists"] = False

with st.spinner("Setting up python agent...please wait"):
    llm = AzureChatOpenAI(
        temperature=0.05,
        streaming=True,
        azure_deployment=st.session_state["python_agent_deployment_name"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
        model_name=st.session_state["python_agent_model_name"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        request_timeout=120,
        verbose=False,
    )

    def get_python_agent_executor(llm):
        tools = [PythonREPLTool()]
        instructions = """You are an agent designed to write and execute python code to answer questions.
        You have access to a python REPL, which you can use to execute python code.
        If you get an error, debug your code and try again.
        Only use the output of your code to answer the question. 
        You might know the answer without running any code, but you should still run the code to get the answer.
        If you are asked to write code, return only the code with no additional text or tests.
        If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
        """
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        logger.info(f"{base_prompt=}")
        prompt = base_prompt.partial(instructions=instructions)
        logger.info("prompt from hub is:" + str(prompt))
        agent = create_openai_functions_agent(llm, tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            max_execution_time=100,
            max_iterations=10,
            return_intermediate_steps=True,
            handle_parsing_errors=True,
            memory=None,  # setup memory
            verbose=True,
        )

    def get_csv_agent_executor(llm, file):
        return create_csv_agent(
            llm,
            file,
            verbose=True,
            agent_type="openai-tools",
            max_iterations=10,
            return_intermediate_steps=True,
            # handle_parsing_errors=True,
            max_execution_time=100,
        )

    if (
        st.session_state["uploaded_file"] and st.session_state["csv_exists"]
    ):  # ignore non-csvs for now. parse later
        agent_executor = get_csv_agent_executor(llm, st.session_state["uploaded_file"])
    else:
        agent_executor = get_python_agent_executor(llm)

    st.success("Agent setup done!")

if (
    "llm_csv_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_csv_agent_messages
):
    st.session_state["llm_csv_agent_messages"] = [
        {
            "role": "assistant",
            "content": "I am an agent designed to write and execute python code. How can I help you?",
        }
    ]

for msg in st.session_state.llm_csv_agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_csv_agent_messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])

        st.session_state.llm_csv_agent_messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
        logger.info(str(response))
