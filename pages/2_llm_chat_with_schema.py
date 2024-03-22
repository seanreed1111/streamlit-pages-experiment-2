from pathlib import Path
# import ast 
import os
import json
import streamlit as st
# import pandas as pd
from langchain_openai import AzureChatOpenAI

# from langchain_community.utilities.sql_database import SQLDatabase 
from io import StringIO
# import urllib
from langchain.schema import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "LLM chat with Schema - multipage app"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"
MAX_TOKENS = 1000

def set_system_message(schema):
    system_message = f"""
        You are an expert at writing Mircosoft SQL database queries and T-SQL code. 
        When asked to write SQL queries use the following schema
        \n\n\n
        {schema}
        \n\n\n
        After writing a query, score its estimated accuracy in answering the 
        user's question on a scale of 1-5, with 1 the lowest score and 5 the 
        highest possible. Respond with the query and the accuracy score. If you give
        an accuracy score of 1 or 2, briefly state your reason.
        """
    st.session_state["chat_with_schema_messages"].append(ChatMessage(role="system", content=system_message))   

with st.sidebar:
    uploaded_file = st.file_uploader("Upload your MSSQL schema file")
    if uploaded_file:
        try:
            #parse file into string
            st.session_state["uploaded_schema"] = StringIO(uploaded_file.getvalue().decode("utf-8")).read()  
            st.session_state["uploaded_file"] = True
            set_system_message(st.session_state["uploaded_schema"])
        except Exception as e:
            st.error(e)
            logger.error(e)

def run_azure_config(config_dir = config_dir_path):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

# class StreamHandler(BaseCallbackHandler):
#     def __init__(self, container, initial_text=""):
#         self.container = container
#         self.text = initial_text

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.text += token
#         self.container.markdown(self.text)

run_azure_config()

def reset_chat():
    st.session_state["chat_with_schema_messages"] = []
    st.session_state["uploaded_file"] = False
    st.session_state["uploaded_schema"] = None

reset_chat_button = st.button("Reset Chat", on_click=reset_chat)

if reset_chat_button or ("chat_with_schema_messages" not in st.session_state):
    st.session_state["chat_with_schema_messages"] = [ChatMessage(role="assistant", content="I am an expert at writing MSSQL database queries. How can I help you?")]

for msg in st.session_state.chat_with_schema_messages:
    if msg.role != "system":
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.chat_with_schema_messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        llm = AzureChatOpenAI(
            temperature=0,
            streaming=False,
            max_tokens=MAX_TOKENS,
            azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=os.environ["MODEL_NAME_GPT35"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
        )
        response = llm.invoke(st.session_state.chat_with_schema_messages)
        st.session_state.chat_with_schema_messages.append(ChatMessage(role="assistant", content=response.content))