from pathlib import Path
# import ast 
import os
import json
from functools import partial
import streamlit as st
# import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler

# from langchain_community.utilities.sql_database import SQLDatabase 
from io import StringIO
# import urllib
from langchain.schema import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "LLM chat with Schema - multipage app"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

MAX_TOKENS = 1000
config_dir_path = st.session_state["config_dir_path"]

if "default_schema_filename" not in st.session_state:
    st.session_state["default_schema_filename"] = "DDL_for_LLM_upload_sample.sql"

# @st.cache_resource
# def get_llm():
#     return AzureChatOpenAI(
#         temperature=0,
#         streaming=False,
#         max_tokens=MAX_TOKENS,
#         azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
#         azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
#         model_name=os.environ["MODEL_NAME_GPT35"],
#         openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
#         request_timeout=45,
#         verbose=True,
#     )
# llm = get_llm()

@st.cache_resource
def set_system_message(schema):
    system_message_string = f"""
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
    system_message = ChatMessage(role="user", content=system_message_string) 
    st.session_state["chat_with_schema_system_message"] = system_message
    logger.info(f"first 500 chars of system message is {system_message.content[:500]}")
    st.session_state["chat_with_schema_messages"].append(system_message)

# @st.cache_resource
# def load_schema_from_file(filename):
#     config_dir_path = st.session_state["config_dir_path"]
#     try:
#         schema_file_path = config_dir_path / filename
#         with open(schema_file_path, 'r') as f:
#             schema = f.read()
        
#         assert schema is not None
#         ##### TRUNCATE SCHEMA 
#         schema = schema[:30000]
#         ##### TRUNCATED SCHEMA
#         st.session_state["uploaded_schema"] = schema
#         return schema
    
#     except Exception as e:
#         logger.error(str(e))
#         st.error(e)

# schema = load_schema_from_file(st.session_state["default_schema_filename"])
# st.session_state["default_schema"] = schema
# logger.info(f"first 500 chars of default schema is {schema[:500]}")


with st.sidebar:
    uploaded_file = st.file_uploader("Upload your MSSQL schema file")
    if uploaded_file:
            #parse file into string and save
            st.session_state["uploaded_schema"] = StringIO(uploaded_file.getvalue().decode("utf-8")).read()  
            logger.info(f"first 500 chars of uploaded schema is {st.session_state['uploaded_schema'][:500]}")
            set_system_message(st.session_state["uploaded_schema"]) #happens after upload
            st.success("schema successfully uploaded")

  
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)


# def reset_chat():
#     # st.session_state["chat_with_schema_messages"] = []
#     st.session_state["uploaded_file"] = False
#     st.session_state["uploaded_schema"] = None
# reset_chat_button = st.button("Reset Chat", on_click=reset_chat)

reset_chat_button = st.button("Reset Chat")
if reset_chat_button or not st.session_state["chat_with_schema_messages"]:
    content = "I am an expert at writing MSSQL database queries. If you upload your schema, I write more specific queries for you."
    st.session_state["chat_with_schema_messages"] = [ChatMessage(role="assistant", content=content)]

for msg in st.session_state.chat_with_schema_messages:
    if msg is not st.session_state["chat_with_schema_system_message"]: #do not write system message over and over again
        st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.chat_with_schema_messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"): #I should make a partial and add the stream_handler into it
        stream_handler = StreamHandler(st.empty())
        llm = AzureChatOpenAI(
            temperature=0,
            streaming=True,
            max_tokens=MAX_TOKENS,
            azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=os.environ["MODEL_NAME_GPT35"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
            callbacks=[stream_handler]
        )
        response = llm.invoke(st.session_state.chat_with_schema_messages)
        st.session_state.chat_with_schema_messages.append(ChatMessage(role="assistant", content=response.content))