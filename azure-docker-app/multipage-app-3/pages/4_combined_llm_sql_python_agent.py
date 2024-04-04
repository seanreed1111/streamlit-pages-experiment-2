import streamlit as st
from pathlib import Path
import os
import json
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_sql_agent
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import ChatMessage
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from loguru import logger

import urllib

LANGCHAIN_PROJECT = f"Multipage App #3 Chat With SQL Agent WAB DB"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")
