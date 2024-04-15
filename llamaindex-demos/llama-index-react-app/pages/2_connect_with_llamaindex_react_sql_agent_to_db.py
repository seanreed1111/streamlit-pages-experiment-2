# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

# https://python.langchain.com/docs/expression_language/cookbook/sql_db
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List  #, Optional, Set, Tuple, cast

import streamlit as st
from llama_index.core.agent import (
    AgentChatResponse,
    AgentRunner,
    QueryPipelineAgentWorker,
    ReActChatFormatter,
    Task,
)
from llama_index.core.agent.react.output_parser import ReActOutputParser
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.query_pipeline import (
    AgentFnComponent,
    AgentInputComponent,
    # CustomAgentComponent,
    # InputComponent,
    # Link,
    # QueryComponent,
    ToolRunnerComponent,
)
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.settings import Settings
from llama_index.core.tools import BaseTool, QueryEngineTool
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from loguru import logger

LANGCHAIN_PROJECT = "Llama-index Chat with SQL Agent"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)

with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_GPT35")
        st.session_state["llm_chat_engine"] = os.getenv("ENGINE_GPT35")
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["llm_chat_model_name"] = os.getenv("MODEL_GPT4")
        st.session_state["llm_chat_engine"] = os.getenv("ENGINE_GPT4")
    st.info(
        f"Now using {llm_choice_radio} as the underlying llm on this page."
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
if "db" not in st.session_state:
    st.info("Please go back to the `app` page and connect to the WAB database")
    st.stop()

llm = AzureOpenAI(
    temperature=0,
    streaming=True,
    azure_deployment=st.session_state["llm_chat_engine"],
    model_name=st.session_state["llm_chat_model_name"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    request_timeout=100,
    verbose=True,
)
embed_model = OpenAIEmbedding(
    model=os.environ["OPENAI_EMBEDDING_MODEL_NAME"],
    deployment_name=os.environ["OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_base=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version="2022-12-01",
    api_type = "azure"
)
# define global callback setting
callback_manager = CallbackManager()
Settings.callback_manager = callback_manager
Settings.llm = llm
Settings.embed_model = embed_model

sql_database = st.session_state["db"]

sql_query_engine = NLSQLTableQueryEngine(
    sql_database=sql_database,
    tables=["party", "deposit", "account", "party_account"],
    verbose=True,
)
sql_tool = QueryEngineTool.from_defaults(
    query_engine=sql_query_engine,
    name="sql_tool",
    description=("Useful for translating a natural language query into a MSSQL query"),
)


## Agent Input Component
## This is the component that produces agent inputs to the rest of the components
## Can also put initialization logic here.


def agent_input_fn(task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
    """Agent input function.

    Returns:
        A Dictionary of output keys and values. If you are specifying
        src_key when defining links between this component and other
        components, make sure the src_key matches the specified output_key.

    """
    # initialize current_reasoning
    if "current_reasoning" not in state:
        state["current_reasoning"] = []
    reasoning_step = ObservationReasoningStep(observation=task.input)
    state["current_reasoning"].append(reasoning_step)
    return {"input": task.input}


## define prompt function
def react_prompt_fn(
    task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
) -> List[ChatMessage]:
    # Add input to reasoning
    chat_formatter = ReActChatFormatter()
    return chat_formatter.format(
        tools,
        chat_history=task.memory.get() + state["memory"].get_all(),
        current_reasoning=state["current_reasoning"],
    )


def parse_react_output_fn(
    task: Task, state: Dict[str, Any], chat_response: ChatResponse
):
    """Parse ReAct output into a reasoning step."""
    output_parser = ReActOutputParser()
    reasoning_step = output_parser.parse(chat_response.message.content)
    return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}


def run_tool_fn(task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep):
    """Run tool and process tool output."""
    tool_runner_component = ToolRunnerComponent(
        [sql_tool], callback_manager=task.callback_manager
    )
    tool_output = tool_runner_component.run_component(
        tool_name=reasoning_step.action,
        tool_input=reasoning_step.action_input,
    )
    observation_step = ObservationReasoningStep(observation=str(tool_output))
    state["current_reasoning"].append(observation_step)
    # TODO: get output

    return {"response_str": observation_step.get_content(), "is_done": False}


def process_response_fn(
    task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep
):
    """Process response."""
    state["current_reasoning"].append(response_step)
    response_str = response_step.response
    # Now that we're done with this step, put into memory
    state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
    state["memory"].put(ChatMessage(content=response_str, role=MessageRole.ASSISTANT))

    return {"response_str": response_str, "is_done": True}


def process_agent_response_fn(task: Task, state: Dict[str, Any], response_dict: dict):
    """Process agent response."""
    return (
        AgentChatResponse(response_dict["response_str"]),
        response_dict["is_done"],
    )



qp = QP(verbose=True)

agent_input_component = AgentInputComponent(fn=agent_input_fn)
react_prompt_component = AgentFnComponent(
    fn=react_prompt_fn, partial_dict={"tools": [sql_tool]}
)


parse_react_output = AgentFnComponent(fn=parse_react_output_fn)
run_tool = AgentFnComponent(fn=run_tool_fn)
process_response = AgentFnComponent(fn=process_response_fn)
process_agent_response = AgentFnComponent(fn=process_agent_response_fn)

qp.add_modules(
    {
        "agent_input": agent_input_component,
        "react_prompt": react_prompt_component,
        "llm": llm,
        "react_output_parser": parse_react_output,
        "run_tool": run_tool,
        "process_response": process_response,
        "process_agent_response": process_agent_response,
    }
)

# link input to react prompt to parsed out response (either tool action/input or observation)
qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

# add conditional link from react output to tool call (if not done)
qp.add_link(
    "react_output_parser",
    "run_tool",
    condition_fn=lambda x: not x["done"],
    input_fn=lambda x: x["reasoning_step"],
)
# add conditional link from react output to final response processing (if done)
qp.add_link(
    "react_output_parser",
    "process_response",
    condition_fn=lambda x: x["done"],
    input_fn=lambda x: x["reasoning_step"],
)

# whether response processing or tool output processing, add link to final agent response
qp.add_link("process_response", "process_agent_response")
qp.add_link("run_tool", "process_agent_response")


agent_worker = QueryPipelineAgentWorker(qp)
agent = AgentRunner(agent_worker, callback_manager=CallbackManager([]), verbose=True)

# start task#########################################################3
# task = agent.create_task("how many tables are in the schema?")

# task = agent.create_task("show me the account ids of accounts with the top ten deposit amounts")
# q = "calculate the total deposit amount by cost center. \
#     show the top ten deposit amounts along with the associated cost center"

# q = """
# Calculate the total deposit amount by cost center 
# and show a report of the top 10 deposit amounts along with the respective cost centers
# """

# q= """
# retrieve the top 10 cost centers along with the number of deposit accounts, 
# total deposit amount, average deposit account balance, and the deposit rank. 
# Use the PARTY, PARTY_ACCOUNT, ACCOUNT, and DEPOSIT to get your answer
# show deposit current balance in descending order.
# """
# task = agent.create_task(q)
# step_output = agent.run_step(task.task_id)

# logger.info(f"{step_output=}")

# response = agent.chat(q)
# print(response)
# logger.info(str(response))

if (
    "llm_sql_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_sql_agent_messages
):
    st.session_state["llm_sql_agent_messages"] = [
        ChatMessage(role="assistant", content="How can I help you?")
    ]
# for msg in st.session_state.messages:
#     st.chat_message(msg.role).write(msg.content)
for msg in st.session_state.llm_sql_agent_messages:
    role = msg.dict()['role']
    if role == MessageRole.ASSISTANT:
        st.chat_message("assistant").write(msg.dict()["content"])
    elif role == MessageRole.USER:
        st.chat_message("user").write(msg.dict()["content"])        

if prompt := st.chat_input():
    st.session_state.llm_sql_agent_messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    logger.info(f"{prompt=}")

    response: AgentChatResponse = agent.chat(prompt)
    # logger.info(f"type of response = {type(response)}")
    st.session_state.llm_sql_agent_messages.append(
        ChatMessage(role="assistant", content=response.response)
    )
    with st.chat_message("assistant"):
        st.write(response.response)
