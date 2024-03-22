# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

#https://python.langchain.com/docs/expression_language/cookbook/sql_db
import streamlit as st
from pathlib import Path
import os
import json
from langchain_community.chat_models.azure_openai import AzureChatOpenAI #deprecated class, fix later
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase #from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit 
from sqlalchemy import create_engine
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import urllib

# LANGCHAIN_PROJECT = "Experiment #4 Chat With SQL Agent WAB DB"
LANGCHAIN_PROJECT = Path(__file__).stem
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)

# RUN install_drivers.sh in bash
# pip install 

def run_azure_config():
    config_dir = Path.cwd()
    openai_config_file_path = config_dir / "allconfig.json"
    config_files = [openai_config_file_path]
    config = {}
    for file in config_files:
        with open(file) as json_config:
            config.update(json.load(json_config))
    for k in config:
        os.environ[k] = config[k]

    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

def get_wab_connection_string(db_config_file="dbconfig.json"):
    driver= '{ODBC Driver 18 for SQL Server}'
    db_config_path = Path.cwd() / db_config_file
    
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


run_azure_config()

LOCALDB = "CHINOOKDB"

# User inputs
radio_opt = ["Use sample Chinook database", "Connect to WAB Database"]
selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
if radio_opt.index(selected_opt) == 1:
    db_uri = st.sidebar.text_input(
        label="Database URI", placeholder="azsqldb-genai-dataanalytics-sb"
    )
else:
    db_uri = "CHINOOKDB"

# Setup agent
llm = AzureChatOpenAI(
            temperature=0,
            streaming=True,
            max_tokens=800,
            azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=os.environ["MODEL_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
        )


@st.cache_resource(ttl="2h")
def configure_db(db_uri):
    if db_uri == "CHINOOKDB":
        # Make the DB connection read-only to reduce risk of injection attacks
        # See: https://python.langchain.com/docs/security
        db_filepath = Path.cwd() / "Chinook.db"
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    else:
        return SQLDatabase.from_uri(database_uri=get_wab_connection_string())


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "llm_sql_agent_messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["llm_sql_agent_messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.llm_sql_agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.llm_sql_agent_messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.llm_sql_agent_messages.append({"role": "assistant", "content": response})
        st.write(response)
