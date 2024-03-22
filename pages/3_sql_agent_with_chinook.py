# %%writefile streaming_app.py
# experiment-3-chat-with-local-db
import streamlit as st
from pathlib import Path
import os
import json
from langchain_community.chat_models.azure_openai import AzureChatOpenAI #deprecated class, fix later
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.agent_toolkits import SQLDatabaseToolkit #https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agent
from sqlalchemy import create_engine
import sqlite3
from loguru import logger

LANGCHAIN_PROJECT = Path(__file__).stem
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)

config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"
db_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB\scripts")
def run_config(path = config_dir_path):

    openai_config_file_path = config_dir_path / "allconfig.json"
    config_files = [openai_config_file_path]
    config = {}
    for file in config_files:
        with open(file) as json_config:
            config.update(json.load(json_config))
    for k in config:
        os.environ[k] = config[k]

    os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

run_config()
LOCALDB = "USE_LOCALDB"
db_uri = LOCALDB

# # User inputs
# radio_opt = ["Use sample database - Chinook.db", "Connect to your own SQL database"]
# selected_opt = st.sidebar.radio(label="Choose suitable option", options=radio_opt)
# if radio_opt.index(selected_opt) == 1:
#     db_uri = st.sidebar.text_input(
#         label="Database URI", placeholder="mysql://user:pass@hostname:port/db"
#     )
# else:
#     db_uri = LOCALDB

# Check user inputs
if not db_uri:
    st.info("Please enter database URI to connect to your database.")
    st.stop()

# Setup agent
llm = AzureChatOpenAI(
            temperature=0,
            streaming=True,
            max_tokens=900,
            azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=os.environ["MODEL_NAME"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
        )

@logger.catch
@st.cache_resource(ttl="2h")
def configure_db(uri = db_uri, db_dir_path = db_dir_path):
    if db_uri == LOCALDB:

        db_filepath = db_dir_path / "Chinook.db"
        creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite:///", creator=creator))
    return SQLDatabase.from_uri(uri)


db = configure_db(db_uri)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Ask me anything!")

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
