import json
import os
import urllib
from pathlib import Path

import streamlit as st
import tiktoken
from langchain_community.utilities.sql_database import SQLDatabase
from loguru import logger

LANGCHAIN_PROJECT = "LLM-Compiler"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

USETRG = True
SCHEMA = {"schema": "trg"} if USETRG else {"schema": "sandbox"}

with st.spinner("performing Azure configuration... please wait"):
    if "config_dir_path" not in st.session_state:
        st.session_state["config_dir_path"] = Path.cwd() / "config"

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

    if st.session_state["run_azure_config"]:
        st.success("Azure Configuration... done.")


# establish chat messages for each page and add to session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "sql_messages" not in st.session_state:
    st.session_state["sql_messages"] = []
if "llm_sql_agent_messages" not in st.session_state:
    st.session_state["llm_sql_agent_messages"] = []
if "llm_python_agent_messages" not in st.session_state:
    st.session_state["llm_python_agent_messages"] = []

with st.sidebar:
    db_connection_radio = st.radio(
        "Choose one", ["No DB Connection Needed", "Connect to WAB DB"]
    )
    if db_connection_radio == "Connect to WAB DB" and "db" not in st.session_state:
        with st.spinner(
            "performing database configuration and connecting... please wait"
        ):

            @st.cache_resource(ttl="2h")
            def get_db_engine(db_config_file, config_dir_path, **kwargs):
                @st.cache_resource(ttl="2h")
                def get_wab_connection_string(db_config_file, config_dir_path):
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

                return SQLDatabase.from_uri(
                    database_uri=get_wab_connection_string(
                        db_config_file, config_dir_path
                    ),
                    **kwargs,
                )

            # connect and test
            test_query = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA;"
            try:
                db = get_db_engine(
                    db_config_file="dbconfig.json",
                    config_dir_path=st.session_state["config_dir_path"],
                    **SCHEMA,
                )
                logger.info("DB connection established! Testing connection with query")
                db.run(test_query)
                st.session_state["db"] = db
                st.success("Sucessfully connected to the database")
                logger.info("DB test query completed successfully")
            except Exception as e:
                st.warning("Database connection failed!")
                st.error(e)
                logger.error(str(e))

    st.info("Please select a app to use from the sidebar.")
