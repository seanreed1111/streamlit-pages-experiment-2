import os
import json
import urllib
import streamlit as st
from pathlib import Path
from loguru import logger
from langchain_community.utilities.sql_database import SQLDatabase

LANGCHAIN_PROJECT = "Multipage App"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
SCHEMA = {"schema":"trg"}
MAX_TOKENS = 4000

with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35")
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["model_name"] = os.getenv("MODEL_NAME")   
        st.session_state["deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    st.info(f"Now using {llm_choice_radio} as the underlying llm.")

with st.spinner("performing app setup script... please wait"):
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
    
    if st.session_state["run_azure_config"]:
        st.info("Azure Configuration... done.")

    if "max_tokens" not in st.session_state:
        st.session_state["max_tokens"] = MAX_TOKENS

    if "db" not in st.session_state:
        @st.cache_resource(ttl="4h")
        def get_db_engine(db_config_file, config_dir_path, **kwargs):
            
            @st.cache_resource(ttl="4h")    
            def get_wab_connection_string(db_config_file, config_dir_path):
                driver= '{ODBC Driver 18 for SQL Server}'
                db_config_path = config_dir_path / db_config_file

                with open(db_config_path) as json_file:
                    dbconfig = json.load(json_file)

                server = dbconfig['server']
                database = dbconfig['database']
                uid = dbconfig['username']
                pwd = dbconfig['password']
                port = int(dbconfig.get("port",1433))
                pyodbc_connection_string = f"DRIVER={driver};SERVER={server};PORT={port};DATABASE={database};UID={uid};PWD={pwd};Encrypt=yes;Connection Timeout=30;READONLY=True;"
                params = urllib.parse.quote_plus(pyodbc_connection_string)
                sqlalchemy_connection_string = f"mssql+pyodbc:///?odbc_connect={params}"
                return sqlalchemy_connection_string
            

            return SQLDatabase.from_uri(database_uri=get_wab_connection_string(db_config_file, config_dir_path), **kwargs
                                    )

        with st.spinner(".....connecting to database.."):
            test_query = "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA;"
            try: 
                db = get_db_engine(db_config_file ="dbconfig.json",config_dir_path = st.session_state["config_dir_path"],
                    **SCHEMA
                )
                db.run(test_query)
                st.session_state["db"] = db
                st.success("Sucessfully connected to the database")
            except Exception as e:
                st.error(e)
                logger.error(str(e))

    # establish the multiple chat message containers
    if "chat_with_schema_messages" not in st.session_state:
        st.session_state["chat_with_schema_messages"] = []

    if "sql_messages" not in st.session_state:
        st.session_state["sql_messages"] = []
    
    if "llm_python_agent_messages" not in st.session_state:
            st.session_state["llm_python_agent_messages"] = []
    
    st.success("Setup completed! Please select a app to use from the sidebar.")
    