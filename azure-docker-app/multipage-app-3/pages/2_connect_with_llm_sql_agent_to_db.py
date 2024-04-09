# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

# https://python.langchain.com/docs/expression_language/cookbook/sql_db
# import json
import os
import sys

# import sqlalchemy
# from sqlalchemy import create_engine
# import urllib
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent

# from langchain.agents.agent_types import AgentType
# from langchain.schema import ChatMessage
# from langchain.storage import InMemoryStore
# from langchain.storage import LocalFileStore
# from langchain.storage import LocalFileStore
# # Instantiate the LocalFileStore with the root path
# file_store = LocalFileStore("/path/to/root")
from langchain_community.agent_toolkits import SQLDatabaseToolkit

# from langchain.callbacks import StreamlitCallbackHandler
from langchain_community.callbacks import StreamlitCallbackHandler

# from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import AzureChatOpenAI
from loguru import logger


def logger_setup():
    log_dir = Path.home() / "PythonProjects" / "multipage-app-3" / "logs"
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

LANGCHAIN_PROJECT = "Multipage App #3 Chat With SQL Agent WAB DB"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

sample_system_prompt = """
You are a MSSQL agent designed to interact with the user and generate SQL scripts based on their requirements.
 
INSTRUCTIONS:

- Request the user to provide some sample SQL scripts that are specific to their database or application, if none is included in the prompt. These will help guide the generation of SQL scripts that meet the user requirements.
- Ask the user to specify their question or requirement.
- Generate a SQL script that aligns with the user's question and the provided SQL schema.
- During the interaction with the user if it appears the requests are very complex and you are unable to generate appropriate SQL scripts, then you can request the user to provide additional sample queries that may help you understand the inter relationships of the tables better.
 
Remember, the database you are working with a schema named 'trg'. It's a snapshot of a real dataset and does not track changes over time. If a user's question implies tracking data over time, gently remind them of this limitation.
 
When crafting SQL queries, aim to include data elements that are human-readable. 
For instance, rather than returning only a customer ID, include the customer name as well. This enhances the understandability of the results.
 
In this schema, 'party' refers to the customer.
 
Here are two example SQL queries for your reference:
 
To find deposit values:
SELECT SUM(DP_CUR_BAL) AS Total_Deposit_Value
FROM trg.DEPOSIT;


To join three tables: PARTY, PARTY_ACCOUNT, and DEPOSIT:
SELECT 
    p.PRTY_ID AS Customer_ID,
    CASE 
        WHEN p.PRTY_NM IS NOT NULL THEN p.PRTY_NM
        ELSE p.FRST_NM + ' ' + ISNULL(p.MDLE_NM, '') + ' ' + p.LST_NM
    END AS Customer_Name,
    SUM(d.DP_CUR_BAL) AS Total_Deposit_Value
FROM 
    trg.PARTY p
JOIN 
    trg.PARTY_ACCOUNT pa ON p.PRTY_ID = pa.PRTY_ID
JOIN 
    trg.DEPOSIT d ON pa.ACCT_ID = d.ACCT_ID
GROUP BY 
    p.PRTY_ID,
    p.PRTY_NM,
    p.FRST_NM,
    p.MDLE_NM,
    p.LST_NM;
 

PROCESSING STEPS:
1.0 When asked to perform complex tasks, you must first plan and then reflect on the steps. 
2.0 If DISPLAY_INTERMEDIATE i set to YES, then display intermediate steps to keep the user informed about the actions being taken. If this flag is not set to YES, then DO NOT display the intermediate steps.
3.0 For queries that may yield large responses, ask the user to input the number of records to return. This could range from a few records to a larger sample, but not necessarily all records.
4.0 Ensure the user has the ability to stop the query generation and execution process if they wish. This could be based on a timeout value.
 
The basic interactions may be something like this:
Question: <User's question> 
Understand Question: <Your understanding of the question> 
Intermediate step 1: <Identify tables to be used or other steps> 
Intermediate step 2: <Identify columns in the tables to be used or additional, lower level steps> 
SQL Query: <Final SQL query>
 
Remember: At start up after providing all the info about how this works, clearly instruct the user what to do next. Even when there are lot of instructions, the next step should be very clear to the user
"""
with st.sidebar:
    llm_choice_radio = st.radio("Choose one", ["GPT-3.5-turbo", "GPT-4-turbo"])
    if llm_choice_radio == "GPT-3.5-turbo":
        st.session_state["agent_model_name"] = os.getenv("MODEL_NAME_GPT35")
        st.session_state["agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME_GPT35"
        )
    elif llm_choice_radio == "GPT-4-turbo":
        st.session_state["agent_model_name"] = os.getenv("MODEL_NAME")
        st.session_state["agent_deployment_name"] = os.getenv(
            "AZURE_OPENAI_API_DEPLOYMENT_NAME"
        )
    st.info(f"Now using {llm_choice_radio} as the underlying agent llm.")

os.environ["LANGCHAIN_PROJECT"] = f"{LANGCHAIN_PROJECT} with {llm_choice_radio}"

with st.spinner("Setting up agent...please wait"):
    CONFIG_DIR_PATH = st.session_state["config_dir_path"]
    try:
        db = st.session_state["db"]
    except Exception as e:
        st.error(e)
        st.error("Please go back to main app page and connect to the WAB database")
        st.stop()

    llm = AzureChatOpenAI(
        temperature=0.2,
        streaming=True,
        azure_deployment=st.session_state["agent_deployment_name"],
        azure_endpoint=os.environ["AZURE_OPENAI_API_ENDPOINT"],
        model_name=st.session_state["agent_model_name"],
        openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        request_timeout=120,
        verbose=True,
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type="openai-tools",
        max_iterations=15,
        handle_parsing_errors=True,
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    st.success("Agent setup done!")

if (
    "llm_sql_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_sql_agent_messages
):
    st.session_state["llm_sql_agent_messages"] = [
        {"role": "system", "content": sample_system_prompt},
        {"role": "assistant", "content": "How can I help you?"},
    ]

for msg in st.session_state.llm_sql_agent_messages:
    if msg["role"] != "system":
        st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.llm_sql_agent_messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    logger.info(f"{prompt=}")

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container())
        response = agent_executor.invoke({"input": prompt}, callbacks=[st_cb])
        st.session_state.llm_sql_agent_messages.append(
            {"role": "assistant", "content": response}
        )
        st.write(response)
        logger.info("response: " + str(response))
