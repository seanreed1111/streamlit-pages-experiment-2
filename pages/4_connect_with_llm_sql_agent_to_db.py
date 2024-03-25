# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

#https://python.langchain.com/docs/expression_language/cookbook/sql_db
import streamlit as st
from pathlib import Path
import os
import json
# from langchain_community.chat_models.azure_openai import AzureChatOpenAI #deprecated class, fix later
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.schema import ChatMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from loguru import logger
# import sqlalchemy
# from sqlalchemy import create_engine
import urllib

LANGCHAIN_PROJECT = "Experiment #4 Chat With SQL Agent WAB DB"
# LANGCHAIN_PROJECT = Path(__file__).stem
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

SCHEMA = {"schema":"trg"}
CONFIG_DIR_PATH = st.session_state["config_dir_path"]

@logger.catch
@st.cache_resource(ttl="4h")
def get_db_engine(db_config_file="dbconfig.json", config_dir_path = CONFIG_DIR_PATH, **kwargs):
    
    if not kwargs: 
        kwargs = {"schema":"sandbox"}
    
    @st.cache_resource(ttl="4h")    
    def get_wab_connection_string(db_config_file=db_config_file, config_dir_path=CONFIG_DIR_PATH ):
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
    

    return SQLDatabase.from_uri(database_uri=get_wab_connection_string(),
                                **kwargs
                               )

test_query = """
    SELECT TABLE_NAME
    FROM INFORMATION_SCHEMA.TABLES
    WHERE TABLE_SCHEMA = 'trg'
    ORDER BY TABLE_NAME;
    """  

with st.sidebar:
    st.spinner("connecting to database..")
    try: 
        db = get_db_engine(**SCHEMA)

        db.run(test_query)
        st.success("Sucessfully connected to the database")
    except Exception as e:
        st.error(e)
        logger.error(str(e))

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

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "llm_sql_agent_messages" not in st.session_state or st.button("Clear message history") or not st.session_state.llm_sql_agent_messages:
    st.session_state["llm_sql_agent_messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.llm_sql_agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_sql_agent_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(prompt, callbacks=[st_cb])
        st.session_state.llm_sql_agent_messages.append({"role": "assistant", "content": response})
        st.write(response)
