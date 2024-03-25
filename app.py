import os
import json
import streamlit as st
from pathlib import Path
from loguru import logger


LANGCHAIN_PROJECT = "Multipage App"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

st.sidebar.success("Select a app to use from above.")

if "config_dir_path" not in st.session_state:
    st.session_state["config_dir_path"] = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"



def run_azure_config(config_dir):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

if "run_azure_config" not in st.session_state:
    run_azure_config(st.session_state["config_dir_path"])
    st.session_state["run_azure_config"] = True

with st.spinner("loading llm"):
    if "llm" not in st.session_state:
        # cache the resource using st.cache_resource
        pass

if "sql_agent" not in st.session_state:
    # load agent
        # cache the resource using st.cache_resource
    pass

if "db" not in st.session_state:
    # load db
    # cache the resource using st.cache_resource
    pass

# establish the multiple chat message containers
if "chat_with_schema_messages" not in st.session_state:
    st.session_state["chat_with_schema_messages"] = []

if "sql_messages" not in st.session_state:
    st.session_state["sql_messages"] = []