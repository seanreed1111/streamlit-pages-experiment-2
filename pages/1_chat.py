import os
import json
from pprint import pprint
from pathlib import Path
from loguru import logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from langchain_openai import AzureChatOpenAI
import streamlit as st

LANGCHAIN_PROJECT = "Experiment #1: Chat with LLM - multipage app"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

MAX_TOKENS = 1000

config_dir_path = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def run_azure_config(config_dir = config_dir_path):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

def clear_chat_history():
    st.session_state.chat_history = []

clear_chat_history = st.button('Clear Chat History', on_click=clear_chat_history)

run_azure_config(config_dir_path)


if "messages" not in st.session_state or clear_chat_history:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        llm = AzureChatOpenAI(
            temperature=0,
            streaming=True,
            max_tokens=MAX_TOKENS,
            azure_deployment=os.environ["AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=os.environ["MODEL_NAME_GPT35"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
            callbacks=[stream_handler]
        )
        
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
