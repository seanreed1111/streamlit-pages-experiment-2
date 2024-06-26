import datetime
import json
import os
import tempfile
import urllib
from pathlib import Path
import sys
import folium
import geopandas
import geopy
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
)
from langchain.agents.agent_types import AgentType
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import ChatMessage
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_experimental.tools import PythonREPLTool
from langchain_openai import AzureChatOpenAI
from loguru import logger
from mpl_toolkits.basemap import Basemap

LANGCHAIN_PROJECT = "Multipage App #3 Visualize With CSV Python Agent"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

now = str(datetime.date.today())
temp_dir_path = tempfile.mkdtemp(prefix=now)
log_file_path = Path(temp_dir_path) / "csv_python_agent.log" #appends automatically if file exists
st.session_state["temp_dir_path"] = Path(temp_dir_path)
logger.info(f"created {temp_dir_path=}")
log_level = "DEBUG"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS zz}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
logger.add(sys.stderr, level=log_level, format=log_format, colorize=True, backtrace=True, diagnose=True)
logger.add(log_file_path, level=log_level, format=log_format, colorize=False, backtrace=True, diagnose=True)

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


# TTD
# agent still does not know where to find the csv file. it is currently looking for it in the 
# temp_dir_path. I did not want to feed the entire file into the system message
# as that is not sustainable.
# maybe the answer is to drop down from file uploader and go
# back to stringifying and saving the file to tempdir manually using BytesIO library


# tell agent where the file is located via system message
# 

# upload file
# see also langchain.storage for storage https://api.python.langchain.com/en/latest/langchain_api_reference.html#module-langchain.storage

with st.sidebar:
    st.markdown("### UPLOAD FILE")
    file_uploader_radio = st.radio("Choose one", ["no file needed", "upload file"])
    if file_uploader_radio == "upload file":
        uploaded_file = st.file_uploader("Choose a file")
        logger.info(f"\ntype of uploaded file is {type(uploaded_file)}\n")
        if uploaded_file:
            st.session_state["filename"] = uploaded_file.name
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["uploaded_file_exists"] = True

            st.success(f"file {uploaded_file.name} uploaded successfully")

    elif file_uploader_radio == "no file needed":
        st.session_state["filename"] = None
        st.session_state["uploaded_file"] = None
        st.session_state["uploaded_file_exists"] = False
        st.session_state["csv_exists"] = False

with st.spinner("Setting up python agent...please wait"):
    llm = AzureChatOpenAI(
        temperature=0,
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
        The user will often refer to the file they have uploaded. This file is available as input for your python programs.
        in the variable <<< st.session_state["uploaded_file"] >>>
        If you get an error, debug your code and try again.
        Save your output, including code and visualizations, to the directory {temp_dir_path}. This is the only
        directory you should use for output.
        You might know the answer without running any code, but you should still run the code to check that
        you get the correct answer.
        If it does not seem like you can write code to answer the question, 
        just return "I don't know how to do that" as the answer.
        """.format(temp_dir_path=st.session_state["temp_dir_path"])
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

    agent_executor = get_python_agent_executor(llm)

    st.success("Agent setup done!")

if (
    "llm_python_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_python_agent_messages
):
    if st.session_state["uploaded_file_exists"]:
        initial_content = f"I am an agent designed to write and execute python code. I have access to \
            the file {st.session_state['filename']} in my memory store. How can I help you?"

    else:
        initial_content = "I am an agent designed to write and execute python code. How can I help you?"

    st.session_state["llm_python_agent_messages"] = [
        {
            "role": "system",
            "content": f"You are an again that can write and execute python code. you have access \
                to data in a file {st.session_state['filename']}",
        },
        {
            "role": "assistant",
            "content": initial_content,
        },
    ]

for msg in st.session_state.llm_python_agent_messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_python_agent_messages.append(
        {"role": "user", "content": prompt}
    )
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])

        st.session_state.llm_python_agent_messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
        logger.info(str(response))
