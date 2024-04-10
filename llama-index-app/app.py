import json
import os
import urllib
from pathlib import Path

import streamlit as st
import tiktoken
from langchain_community.utilities.sql_database import SQLDatabase
from loguru import logger
from llama_index.llms.azure_openai import AzureOpenAI
LANGCHAIN_PROJECT = "Llama-index SQL Agent App #1"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

SCHEMA = {"schema": "trg"}
CONFIG_DIR_PATH = Path.cwd() / "config"
DB_CONFIG_FILE = "db_config_llama_index.json"
AZURE_CONFIG_FILE= "azure_config_llama_index.json"

def run_azure_config(config_dir_path, azure_config_file):
    azure_config_file_path = config_dir_path / azure_config_file
    config = {}
    with open(azure_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

if "run_azure_config" not in st.session_state:
    with st.spinner("performing Azure configuration... please wait"):
        if "config_dir_path" not in st.session_state:
            st.session_state["config_dir_path"] = CONFIG_DIR_PATH
        run_azure_config(CONFIG_DIR_PATH, AZURE_CONFIG_FILE)
        st.session_state["run_azure_config"] = True
        st.success("Azure configuration... done.")


# establish chat messages for each page and add to session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sql_messages" not in st.session_state:
    st.session_state["sql_messages"] = []
if "llm_sql_agent_messages" not in st.session_state:
    st.session_state["llm_sql_agent_messages"] = []
if "llm_python_agent_messages" not in st.session_state:
    st.session_state["llm_python_agent_messages"] = []



from llama_index.core import SQLDatabase
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
    column,
)

def get_connection_string(config_dir_path, db_config_file):
    driver = "{ODBC Driver 18 for SQL Server}"
    db_config_path = config_dir_path / db_config_file

    with open(db_config_path) as json_file:
        dbconfig = json.load(json_file)

    server = dbconfig["server"]
    database = dbconfig["database"]
    uid = dbconfig["username"]
    pwd = dbconfig["password"]
    port = int(dbconfig.get("port", 1433))
    pyodbc_connection_string = f"DRIVER={driver};SERVER={server};PORT={port};DATABASE={database};UID={uid};PWD={pwd};Encrypt=yes;Connection Timeout=30;READONLY=True;"
    params = urllib.parse.quote_plus(pyodbc_connection_string)
    sqlalchemy_connection_string = (
        f"mssql+pyodbc:///?odbc_connect={params}"
    )
    return sqlalchemy_connection_string

db_connection_radio = st.radio(
    "Choose one", ["No DB Connection Needed", "Connect to WAB DB"]
)
if db_connection_radio == "Connect to WAB DB" and "db" not in st.session_state:
    with st.spinner(
        "performing database configuration and connecting... please wait"
    ):

        test_query = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA;"
        try:
            sqlalchemy_connection_string = (
                get_connection_string(
                config_dir_path=st.session_state["config_dir_path"], 
                db_config_file=DB_CONFIG_FILE
                )
            )
            engine = create_engine(sqlalchemy_connection_string)
            db = SQLDatabase(engine, **SCHEMA ) #SCHEMA????
            # db.run_sql(test_query) ### HOW DO YOU RUN A QUERY???
            st.session_state["db"] = db
            st.success("Sucessfully created the database")
            st.info("Please select a app to use from the sidebar.")
            # logger.info("DB test query completed successfully")
        except Exception as e:
            st.warning("Database connection failed!")
            st.error(e)
            logger.error(str(e))

    
