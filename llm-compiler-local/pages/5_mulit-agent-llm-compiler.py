# # LLMCompiler
#
# This notebook shows how to implement [LLMCompiler, by Kim, et. al](https://arxiv.org/abs/2312.04511) in LangGraph.
#
#
# It has 3 main components:
#
# 1. Planner: stream a DAG of tasks.
# 2. Task Fetching Unit: schedules and executes the tasks as soon as they are executable
# 3. Joiner: Responds to the user or triggers a second plan
#
#
import datetime  # noqa: F401
import itertools
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import streamlit as st
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_openai_functions_agent,
)
from langchain.chains.openai_functions import create_structured_output_runnable
# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    # ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableBranch
from langchain_core.runnables import (
    chain as as_runnable,
)
from langchain_core.tools import BaseTool  # , tool
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool

# from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import AzureChatOpenAI
from langgraph.graph import END, MessageGraph
from loguru import logger

if "llm-compiler-local/src" not in sys.path:
    sys.path.append("../src")  # needed to get the azure config imports to run

from src.math_tools import get_math_tool
from src.output_parser import LLMCompilerPlanParser, Task
from typing_extensions import TypedDict

RECURSION_LIMIT = 20

if "config_dir_path" not in st.session_state:
    st.info("please ensure that you've completed loading the main (app) page")
    st.stop()


def logger_setup():
    log_dir = Path.home() / "PythonProjects" / "llm-compiler-local" / "logs"
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

with st.sidebar:
    st.markdown("## PICK LLM")
    llm_choice_radio_multi_agent = st.radio(
        "Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"]
    )
    if llm_choice_radio_multi_agent == "GPT-3.5-turbo":
        st.session_state["multi_agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["multi_agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"
        )
    elif llm_choice_radio_multi_agent == "GPT-4-turbo":
        st.session_state["multi_agent_model_name"] = os.getenv("MODEL_NAME")
        st.session_state["multi_agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME"
        )
    st.info(f"Now using {llm_choice_radio_multi_agent} as the underlying llm.")

os.environ["LANGCHAIN_PROJECT"] = (
    f"{Path(__file__).stem}v2-using-{st.session_state['multi_agent_model_name']}"
)

with st.sidebar:
    st.markdown("### UPLOAD FILE")
    file_uploader_radio = st.radio("Choose one", ["no file needed", "upload file"])
    if file_uploader_radio == "upload file":
        uploaded_file = st.file_uploader("Choose a file")
        logger.info(f"\ntype of uploaded file is {type(uploaded_file)}\n")
        if uploaded_file:
            st.session_state["filename"] = uploaded_file.name
            st.session_state["uploaded_file"] = uploaded_file
            st.session_state["uploaded_file_exists"] = True

            st.success(f"file {uploaded_file.name} uploaded successfully")

    elif file_uploader_radio == "no file needed":
        st.session_state["filename"] = None
        st.session_state["uploaded_file"] = None
        st.session_state["uploaded_file_exists"] = False
        st.session_state["csv_exists"] = False

with st.spinner("Setting up multi-agent...please wait"):
    llm = AzureChatOpenAI(
        temperature=0.05,
        streaming=True,
        model_name=st.session_state["multi_agent_model_name"],
        azure_deployment=st.session_state["multi_agent_deployment_name"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        request_timeout=120,
        verbose=True,
    )

# Initialize tools
calculate = get_math_tool(llm)


# search = TavilySearchResults(
#     max_results=8,
#     description='tavily_search_results_json(query="the search query") - a search engine.',
# )
def get_python_agent_executor(llm):
    tools = [PythonREPLTool()]
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question. 
    You might know the answer without running any code, but you should still run the code to get the answer.
    If you are asked to write code, return only the code with no additional text or tests.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    logger.info(f"{base_prompt=}")
    prompt = base_prompt.partial(instructions=instructions)
    logger.info("prompt from hub is:" + str(prompt))
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_execution_time=100,
        max_iterations=10,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        memory=None,  # setup memory
        verbose=True,
    )


def get_csv_agent_executor(llm, file):
    return create_csv_agent(
        llm,
        file,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=10,
        return_intermediate_steps=True,
        # handle_parsing_errors=True,
        max_execution_time=100,
    )


tools = [calculate, PythonREPLTool()]

# Planner
# The code below composes constructs the prompt template for the planner and composes it
# with LLM and output parser, defined in [output_parser.py](./output_parser.py).
# The output parser processes a task list in the following form:
#
# ```plaintext
# 1. tool_1(arg1="arg1", arg2=3.5, ...)
# Thought: I then want to find out Y by using tool_2
# 2. tool_2(arg1="", arg2="${1}")'
# 3. join()<END_OF_PLAN>"
# ```
#
# The "Thought" lines are optional. The `${#}` placeholders are variables.
# These are used to route tool (task) outputs to other tools.

prompt = hub.pull("wfh/llm-compiler")
# logger.info(f"\nHere is the llm-compiler prompt {str(prompt)}\n")


@logger.catch
def create_planner(
    llm: BaseChatModel, tools: Sequence[BaseTool], base_prompt: ChatPromptTemplate
):
    tool_descriptions = "\n".join(
        f"{i+1}. {included_tool.description}\n" for i, included_tool in enumerate(tools)
    )
    planner_prompt = base_prompt.partial(
        replan="",
        num_tools=len(tools) + 1,  # +1 is for the join() tool.
        tool_descriptions=tool_descriptions,
    )
    replanner_prompt = base_prompt.partial(
        replan=' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
        "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
        'You MUST use these information to create the next plan under "Current Plan".\n'
        ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
        " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
        " - You must continue the task index from the end of the previous one. Do not repeat task indices.",
        num_tools=len(tools) + 1,
        tool_descriptions=tool_descriptions,
    )

    def should_replan(state: list):
        # Context is passed as a system message
        return isinstance(state[-1], SystemMessage)

    def wrap_messages(state: list):
        return {"messages": state}

    def wrap_and_get_last_index(state: list):
        next_task = 0
        for message in state[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        state[-1].content = state[-1].content + f" - Begin counting at : {next_task}"
        return {"messages": state}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | llm
        | LLMCompilerPlanParser(tools=tools)
    )


# This is the primary "agent" in our application
planner = create_planner(llm, tools, prompt)

# example_question = "What's the temperature in Denver raised to the 2rd power?"

# for task in planner.stream([HumanMessage(content=example_question)]):
#     print(task["tool"], task["args"])
#     print("---")


@logger.catch
def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterable[Task]


@logger.catch
def _execute_task(task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {
                key: _resolve_arg(val, observations) for key, val in args.items()
            }
        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}.)"
            f" Args could not be resolved. Error: {repr(e)}"
        )
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f"ERROR(Failed to call {tool_to_use.name} with args {args}."
            + f" Args resolved to {resolved_args}. Error: {repr(e)})"
        )


@logger.catch
def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # $1 or ${1} -> 1
    ID_PATTERN = r"\$\{?(\d+)\}?"

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if isinstance(arg, str):
        return re.sub(ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
@logger.catch
def schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception:
        observation = traceback.format_exception()
    observations[task["idx"]] = observation


@logger.catch
def schedule_pending_task(
    task: Task, observations: Dict[int, Any], retry_after: float = 0.2
):
    while True:
        deps = task["dependencies"]
        if deps and (any([dep not in observations for dep in deps])):
            # Dependencies not yet satisfied
            time.sleep(retry_after)
            continue
        schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
@logger.catch
def schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""
    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}
    messages = scheduler_input["messages"]
    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    task_names = {}
    originals = set(observations)
    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            deps = task["dependencies"]
            task_names[task["idx"]] = (
                task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            )
            args_for_tasks[task["idx"]] = task["args"]
            if (
                # Depends on other tasks
                deps and (any([dep not in observations for dep in deps]))
            ):
                futures.append(
                    executor.submit(
                        schedule_pending_task, task, observations, retry_after
                    )
                )
            else:
                # No deps or all deps satisfied
                # can schedule now
                schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)
    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name, content=str(obs), additional_kwargs={"idx": k, "args": task_args}
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


@as_runnable
@logger.catch
def plan_and_schedule(messages: List[BaseMessage], config):
    tasks = planner.stream(messages, config)
    # Begin executing the planner immediately
    try:
        tasks = itertools.chain([next(tasks)], tasks)
    except StopIteration:
        # Handle the case where tasks is empty.
        tasks = iter([])
    scheduled_tasks = schedule_tasks.invoke(
        {
            "messages": messages,
            "tasks": tasks,
        },
        config,
    )
    return scheduled_tasks


# tool_messages = plan_and_schedule.invoke([HumanMessage(content=example_question)])

# ## "Joiner"
#
# So now we have the planning and initial execution done. We need a component to process these outputs and either:
#
# 1. Respond with the correct answer.
# 2. Loop with a new plan.
#
# The paper refers to this as the "joiner". It's another LLM call.
# We are using function calling to improve parsing reliability.


class FinalResponse(BaseModel):
    """The final response/answer."""

    response: str


class Replan(BaseModel):
    feedback: str = Field(
        description="Analysis of the previous attempts and recommendations on what needs to be fixed."
    )


class JoinOutputs(BaseModel):
    """Decide whether to replan or whether you can return the final response."""

    thought: str = Field(
        description="The chain of thought reasoning for the selected action"
    )
    action: Union[FinalResponse, Replan]


joiner_prompt = hub.pull("wfh/llm-compiler-joiner").partial(
    examples=""
)  # You can optionally add examples

# logger.info(f"\nHere is the llm-compiler joiner_prompt {str(joiner_prompt)}\n")
runnable = create_structured_output_runnable(JoinOutputs, llm, joiner_prompt)

# We will select only the most recent messages in the state, and format the output to be more useful for
# the planner, should the agent need to loop.


@logger.catch
def _parse_joiner_output(decision: JoinOutputs) -> List[BaseMessage]:
    response = [AIMessage(content=f"Thought: {decision.thought}")]
    if isinstance(decision.action, Replan):
        return response + [
            SystemMessage(
                content=f"Context from last attempt: {decision.action.feedback}"
            )
        ]
    else:
        return response + [AIMessage(content=decision.action.response)]


@logger.catch
def select_recent_messages(messages: list) -> dict:
    selected = []
    for msg in messages[::-1]:
        selected.append(msg)
        if isinstance(msg, HumanMessage):
            break
    return {"messages": selected[::-1]}


joiner = select_recent_messages | runnable | _parse_joiner_output

# input_messages = [HumanMessage(content=example_question)] + tool_messages


# joiner.invoke(input_messages)


# ## Compose using LangGraph

# We'll define the agent as a stateful graph, with the main nodes being:
# 1. Plan and execute (the DAG from the first step above)
# 2. Join: determine if we should finish or replan
# 3. Recontextualize: update the graph state based on the output from the joiner

graph_builder = MessageGraph()
graph_builder.add_node("plan_and_schedule", plan_and_schedule)
graph_builder.add_node("join", joiner)
graph_builder.add_edge("plan_and_schedule", "join")


def should_continue(state: List[BaseMessage]):
    if isinstance(state[-1], AIMessage):
        return END
    return "plan_and_schedule"


graph_builder.add_conditional_edges(
    start_key="join",
    # Next, we pass in the function that will determine which node is called next.
    condition=should_continue,
)
graph_builder.set_entry_point("plan_and_schedule")
chain = graph_builder.compile()


# #### Multi-hop question
# [
#     HumanMessage(
#         content="What was the GDP of Australia in 2020 in in Billions of US Dollars? \
#  What was the GDP of Iceland in 2020 in Billions of US Dollars? Which has the larger GDP? \
#     What is the sum of the GDPs, in Billions of US Dollars"
#     )
# ],
if (
    "llm_multi_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_multi_agent_messages
):
    if st.session_state["uploaded_file_exists"]:
        initial_content = "How can I help you with the file you uploaded?"

    else:
        initial_content = "How can I help you?"

    st.session_state["llm_multi_agent_messages"] = [
        # {
        #     "role": "system",
        #     "content": f"You are an again that can write and execute python code. you have access \
        #         to data in a file {st.session_state['filename']}",
        # },
        {
            "role": "assistant",
            "content": initial_content,
        },
    ]

for msg in st.session_state.llm_multi_agent_messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if user_input := st.chat_input():
    st.session_state.llm_multi_agent_messages.append(
        {"role": "user", "content": user_input}
    )
    st.chat_message("user").write(user_input)

    steps = chain.stream(
        [HumanMessage(content=user_input)],
        {
            "recursion_limit": RECURSION_LIMIT,
        },
    )
    output = {}
    for i, step in enumerate(steps):
        output[i] = step
    logger.info(f"\noutput={str(output)}")
    st.session_state.llm_multi_agent_messages.append(
        {"role": "assistant", "content": output}
    )
    st.chat_message("assistant").write(output)
