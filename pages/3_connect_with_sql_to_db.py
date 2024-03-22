from pathlib import Path
import ast 
import os
import json
import streamlit as st
import pandas as pd

from langchain_community.utilities.sql_database import SQLDatabase
import urllib
from langchain.schema import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "Connect With SQL Only - multipage app"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"

def run_azure_config(config_dir = config_dir_path):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

#### Need get_db_engine to run asynchronously so it doesn't block loading the rest of page
# https://docs.python.org/3/library/asyncio.html
## jason brownlee https://superfastpython.com/python-async-function/
# https://superfastpython.com/asyncio-run-program/
# https://medium.com/@danielwume/an-in-depth-guide-to-asyncio-and-await-in-python-059c3ecc9d96
# https://docs.python.org/3/library/asyncio-task.html
####

@logger.catch
@st.cache_resource(ttl="4h")
def get_db_engine(db_config_file="dbconfig.json", config_dir_path = config_dir_path, **kwargs):
    
    if not kwargs: 
        kwargs = {"schema":"sandbox"}
    
    @st.cache_resource(ttl="4h")    
    def get_wab_connection_string(db_config_file=db_config_file, config_dir_path=config_dir_path ):
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

run_azure_config()
db = get_db_engine()


def test_db_connection(q):
    try:
        db.run(q)
        st.sidebar.success('Sucessfully connected to the database')
        return True
    
    except Exception as e:
        st.error(e)
        return False

q = """
SELECT TABLE_NAME
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'sandbox'
ORDER BY TABLE_NAME;
"""   
test_db_connection(q)

@logger.catch
def convert_response_to_dataframe(response):
    try:
        python_obj_from_response = ast.literal_eval(response)
        # st.text(f"type of python_obj_from_results is:{type(python_obj_from_results)}") #type:list
        # write a function that lets you return the column names to put on top of results
        if isinstance(python_obj_from_response, list):
            df = pd.DataFrame(python_obj_from_response)
            return df
        return response

    except Exception as e:
        return str(e)


def reset_chat():
    st.session_state["sql_messages"] = []

reset_chat_button = st.button("Reset Chat", on_click=reset_chat) 

if reset_chat_button or ("sql_messages" not in st.session_state):
    st.session_state["sql_messages"] = [ChatMessage(role="assistant", content="Enter your MSSQL Query to run against the db")]

for msg in st.session_state.sql_messages:
    if msg.role == "user":
        st.chat_message(msg.role).write(msg.content)
    elif msg.role == "assistant":
        df = convert_response_to_dataframe(msg.content)
        if isinstance(df, pd.DataFrame):
            st.chat_message(msg.role).write("assistant response below")
            st.dataframe(convert_response_to_dataframe(msg.content), use_container_width=True)
        elif df is None:
            st.chat_message(msg.role).write("response is None")
        else:
            st.chat_message(msg.role).write(msg.content) #write the error as a string

with st.sidebar:
    st.sidebar.write("here is a sample query: ")
    st.sidebar.write(q)

if prompt := st.chat_input():
    st.session_state.sql_messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    try:
        response: str = db.run(prompt)
    except Exception as e:
        response = str(e)

    with st.chat_message("assistant"):
        st.session_state.sql_messages.append(ChatMessage(role="assistant", content=response))
