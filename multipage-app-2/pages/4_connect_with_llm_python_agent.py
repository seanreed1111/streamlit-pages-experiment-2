# reference docs 
# - https://python.langchain.com/docs/integrations/toolkits/python
# - https://github.com/langchain-ai/langchain/issues/5611#issuecomment-1603700131

# - https://python.langchain.com/docs/integrations/tools/filesystem
# - https://python.langchain.com/docs/modules/callbacks/filecallbackhandler

import streamlit as st
from pathlib import Path
import os
import json
from langchain_openai import AzureChatOpenAI
# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.schema import ChatMessage
from langchain.memory import ConversationBufferMemory
from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_react_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent
from loguru import logger
import urllib

LANGCHAIN_PROJECT = f"Multipage App #2 Chat With Python Agent using create_openai_functions_agent and PythonREPLTool"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

with st.sidebar:
    llm_choice_radio_python_agent = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio_python_agent == "GPT-3.5-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["python_agent_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35")
    elif llm_choice_radio_python_agent == "GPT-4-turbo":
        st.session_state["python_agent_model_name"] = os.getenv("MODEL_NAME")     
        st.session_state["python_agent_deployment_name"] = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
    st.info(f"Now using {llm_choice_radio_python_agent} as the underlying python agent llm.")

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio_python_agent}"

if "config_dir_path" not in st.session_state:
    st.session_state["config_dir_path"] = Path(r"C:\Users\sreed\OneDrive - West Monroe Partners\BD-Folders\WAB") / "config"

def run_azure_config(config_dir):
    all_config_file_path = config_dir / "allconfig.json"
    config = {}
    with open(all_config_file_path) as json_config:
        config.update(json.load(json_config))
        for k in config:
            os.environ[k] = config[k]

if "run_azure_config" not in st.session_state:
    run_azure_config(st.session_state["config_dir_path"])
    st.session_state["run_azure_config"] = True

with st.spinner("Setting up python agent...please wait"):
    llm = AzureChatOpenAI(
                temperature=0.1,
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
        max_execution_time=500,
        max_iterations=20,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        memory=None,
        verbose=True
    )

    st.success("Agent setup done!")

if "llm_python_agent_messages" not in st.session_state or st.button("Clear message history") or not st.session_state.llm_python_agent_messages:
    st.session_state["llm_python_agent_messages"] = [{"role": "assistant", "content": "I am an agent designed to write and execute python code. How can I help you?"}]

for msg in st.session_state.llm_python_agent_messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_python_agent_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # logger.info(f"the type of the prompt is {type(prompt)}")
    # https://youtu.be/ynRpxQhCsfU?si=8GNYZQkt_O56Dbtz accessing intermediate steps
    # https://python.langchain.com/docs/modules/agents/how_to/intermediate_steps
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])

        # logger.debug(f"\ntype of response['intermediate_steps'] is {type(response['intermediate_steps'])}")
        # logger.debug(f"\nlength of response['intermediate_steps'] is {len(response['intermediate_steps'])}") 
        # logger.debug(f"\nthe final item in response['intermediate_steps'] is {response['intermediate_steps'][-1]}")
        # logger.debug(f"\nthe final response is {response['output']}")                    
        # for i, x in enumerate(response['intermediate_steps']):
        #     logger.debug(f"\nintermediate_step {i}:{str(x)}\n")
        st.session_state.llm_python_agent_messages.append({"role": "assistant", "content": response})
        st.write(response)
