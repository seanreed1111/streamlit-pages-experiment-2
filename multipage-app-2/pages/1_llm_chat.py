import os
import json
# from pprint import pprint
from pathlib import Path
from loguru import logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from langchain_openai import AzureChatOpenAI
import streamlit as st

LANGCHAIN_PROJECT = "Multipage App #2 Chat with LLM - multipage app"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

def clear_chat_history():
    st.session_state.chat_history = []

clear_chat_history = st.button('Clear Chat History', on_click=clear_chat_history)

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
            max_tokens=st.session_state["max_tokens"],
            azure_deployment=st.session_state["deployment_name"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=st.session_state["model_name"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
            callbacks=[stream_handler]
        )
        
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
