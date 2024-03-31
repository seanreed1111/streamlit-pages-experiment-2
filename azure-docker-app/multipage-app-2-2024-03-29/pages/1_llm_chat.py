import os
import json
# from pprint import pprint
from pathlib import Path
from loguru import logger
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import ChatMessage

from langchain_openai import AzureChatOpenAI
import streamlit as st

LANGCHAIN_PROJECT = "Multipage App #2 Chat with LLM"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["llm_chat_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35")
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_NAME")     
        st.session_state["llm_chat_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    st.info(f"Now using {llm_choice_radio} as the underlying llm for chat on this page.")

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio}"

# if "config_dir_path" not in st.session_state:
#     st.session_state["config_dir_path"] = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"

# def run_azure_config(config_dir):
#     all_config_file_path = config_dir / "allconfig.json"
#     config = {}
#     with open(all_config_file_path) as json_config:
#         config.update(json.load(json_config))
#         for k in config:
#             os.environ[k] = config[k]

# if "run_azure_config" not in st.session_state:
#     run_azure_config(st.session_state["config_dir_path"])
#     st.session_state["run_azure_config"] = True

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
            # max_tokens=st.session_state["max_tokens"],
            azure_deployment=st.session_state["llm_chat_deployment_name"],
            azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
            model_name=st.session_state["llm_chat_model_name"],
            openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
            request_timeout=45,
            verbose=True,
            callbacks=[stream_handler]
        )
        
        response = llm.invoke(st.session_state.messages)
        st.session_state.messages.append(ChatMessage(role="assistant", content=response.content))
