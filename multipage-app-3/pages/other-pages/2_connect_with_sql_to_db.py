import ast
import json
import os
from pathlib import Path

import pandas as pd

# from functools import partial
import streamlit as st

# from langchain_community.utilities.sql_database import SQLDatabase
# import urllib
from langchain.schema import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "Multipage App Connect With Pure SQL To DB"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

db = st.session_state["db"]


test_query = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'trg'
    ORDER BY TABLE_NAME;
    """


@logger.catch
@st.cache_data
def parse_repsonse(response: str):
    if response:
        python_obj_from_response = ast.literal_eval(response)
        if python_obj_from_response:
            logger.info(f"python_obj_from_response = {python_obj_from_response}")
            if isinstance(python_obj_from_response, list):
                return ("ok", python_obj_from_response)
    return ("error", response)


@st.cache_data
def get_dataframe_from_response(response):
    logger.info(f"response = {response}")
    status_code, parsed_response = parse_repsonse(response)
    if status_code == "ok":
        df = pd.DataFrame(parsed_response)
        if df is not None:
            return ("ok", df)

    return ("error", response)


reset_chat_button = st.button("Reset Chat")

if (
    "sql_messages" not in st.session_state
    or reset_chat_button
    or not st.session_state["sql_messages"]
):
    first_sql_message = ChatMessage(
        role="assistant", content="Enter your MSSQL Query to run against the db"
    )
    st.session_state["sql_messages"] = [first_sql_message]
    st.session_state["first_sql_message"] = first_sql_message

for msg in st.session_state.sql_messages:
    if msg.role == "user":
        st.chat_message(msg.role).write(msg.content)
    elif msg.role == "assistant" and msg != st.session_state["first_sql_message"]:
        status_code, response = get_dataframe_from_response(
            msg.content
        )  # msg.content is always text
        st.chat_message(msg.role).write(response)


with st.sidebar:
    st.sidebar.write("here is a sample query: ")
    st.sidebar.write(test_query)

if prompt := st.chat_input():
    with st.spinner("running query"):
        st.chat_message("user").write(prompt)
        st.session_state.sql_messages.append(ChatMessage(role="user", content=prompt))
        try:
            response = db.run(prompt)

        except Exception as e:
            response = str(e)

    with st.chat_message("assistant"):
        st.session_state.sql_messages.append(
            ChatMessage(role="assistant", content=response)
        )
        if response:
            st.write(response)
        else:
            st.chat_message(msg.role).write("error: no response")    