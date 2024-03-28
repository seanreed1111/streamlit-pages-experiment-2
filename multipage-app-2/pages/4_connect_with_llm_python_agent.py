# reference docs https://python.langchain.com/docs/integrations/toolkits/python
import streamlit as st
from pathlib import Path
import os
import json
from langchain_openai import AzureChatOpenAI
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import ChatMessage
from loguru import logger
import urllib
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent

LANGCHAIN_PROJECT = f"Multipage App #2 Chat With Python Agent using create_openai_functions_agent and PythonREPLTool"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["python_agent_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35")
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME")     
        st.session_state["python_agent_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    st.info(f"Now using {llm_choice_radio} as the underlying python agent llm.")

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio}"

with st.spinner("Setting up agent...please wait"):
    CONFIG_DIR_PATH = st.session_state["config_dir_path"]

    llm = AzureChatOpenAI(
                temperature=0.05,
                streaming=True,
                # max_tokens=st.session_state["max_tokens"],
                azure_deployment=st.session_state["python_agent_deployment_name"],
                azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
                model_name=st.session_state["python_agent_model_name"],
                openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                request_timeout=120,
                verbose=False
            )

    tools = [PythonREPLTool()]
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        max_execution_time=200,
        max_iterations=5,
        verbose=True
    )

    st.success("Agent setup done!")

if "llm_python_agent_messages" not in st.session_state or st.button("Clear message history") or not st.session_state.llm_python_agent_messages:
    st.session_state["llm_python_agent_messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.llm_python_agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_python_agent_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    logger.info(f"the type of the prompt is {type(prompt)}")
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])
        st.session_state.llm_python_agent_messages.append({"role": "assistant", "content": response})
        st.write(response)
