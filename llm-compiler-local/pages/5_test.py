from langchain import hub
from loguru import logger
from pprint import pprint
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
base_prompt: ChatPromptTemplate = hub.pull("langchain-ai/openai-functions-template") 
logger.info(str(base_prompt))
st.write(f"{base_prompt=}")
