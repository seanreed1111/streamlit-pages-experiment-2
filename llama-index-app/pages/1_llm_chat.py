import json
import os
import sys
from pathlib import Path
import streamlit as st
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.core.llms import ChatMessage
from loguru import logger

LANGCHAIN_PROJECT = "Llama-index Chat with LLM"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_GPT35")
        st.session_state["llm_chat_engine"] = os.getenv(
            "ENGINE_GPT35"
        )
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_GPT4")
        st.session_state["llm_chat_engine"] = os.getenv(
            "ENGINE_GPT4"
        )
    st.info(
        f"Now using {llm_choice_radio} as the underlying llm for chat on this page."
    )

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio}"

def logger_setup():
    log_dir = Path.home() / "logs" / "llama-index-app"
    log_dir.mkdir(exist_ok=True, parents=True)
    log_file_name = Path(__file__).stem + ".log"
    log_file_path = log_dir / log_file_name
    log_level = "DEBUG"
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <yellow>Line {line: >4} ({file}):</yellow> <b>{message}</b>"
    logger.add(
        sys.stderr,
        level=log_level,
        format=log_format,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        log_file_path,
        level=log_level,
        format=log_format,
        colorize=False,
        backtrace=True,
        diagnose=True,
    )

logger_setup()
logger.info(f"Project:{os.environ['LANGCHAIN_PROJECT']}")

llm = AzureOpenAI(
    temperature=0,
    streaming=True,
    azure_deployment=st.session_state["llm_chat_engine"],
    model_name=st.session_state["llm_chat_model_name"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    request_timeout=45,
    verbose=True)

def clear_chat_history():
    st.session_state.chat_history = []

clear_chat_history = st.button("Clear Chat History")


if "messages" not in st.session_state or clear_chat_history:
    st.session_state["messages"] = [
        ChatMessage(role="assistant", content="How can I help you?")
    ]

for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    response = llm.chat(st.session_state.messages)
    # logger.debug(f"\n\n{response=}\n")
    st.session_state.messages.append(
        ChatMessage(role="assistant", content=response.message.content)
        )
    with st.chat_message("assistant"):
        st.write(response.message.content)
    # logger.info("response: " + str(response))
