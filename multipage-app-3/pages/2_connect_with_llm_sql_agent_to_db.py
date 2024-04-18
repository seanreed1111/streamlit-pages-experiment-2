# reference docs https://python.langchain.com/docs/integrations/toolkits/sql_database#use-sqldatabasetoolkit-within-an-agenthttps://python.langchain.com/docs/expression_language/cookbook/sql_db

# https://python.langchain.com/docs/expression_language/cookbook/sql_db
import json
import os
import sys

# import sqlalchemy
# from sqlalchemy import create_engine
# import urllib
from pathlib import Path

import streamlit as st
from langchain.agents import create_sql_agent
from langchain_core.callbacks import Callbacks
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

LANGCHAIN_PROJECT = "Multipage App Chat With SQL Agent WAB DB"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.markdown(f"### {LANGCHAIN_PROJECT}")

# sample_system_prompt = """
# You are a MSSQL agent designed to interact with the user.
# When crafting SQL queries, aim to include data elements that are human-readable.
# For instance, rather than returning only a customer ID, include the customer name as well. This enhances the understandability of the results.

# In this schema, 'party' refers to the customer.

# Here are two example SQL queries for your reference:

# Example #1
# If the user asks you to find deposit values:

# Perform the query
# <<<SELECT SUM(DP_CUR_BAL) AS Total_Deposit_Value
# FROM trg.DEPOSIT;>>>

# Example #2
# If you need to join three tables: trg.PARTY, trg.PARTY_ACCOUNT, and trg.DEPOSIT:
# SELECT
#     p.PRTY_ID AS Customer_ID,
#     CASE
#         WHEN p.PRTY_NM IS NOT NULL THEN p.PRTY_NM
#         ELSE p.FRST_NM + ' ' + ISNULL(p.MDLE_NM, '') + ' ' + p.LST_NM
#     END AS Customer_Name,
#     SUM(d.DP_CUR_BAL) AS Total_Deposit_Value
# FROM
#     trg.PARTY p
# JOIN
#     trg.PARTY_ACCOUNT pa ON p.PRTY_ID = pa.PRTY_ID
# JOIN
#     trg.DEPOSIT d ON pa.ACCT_ID = d.ACCT_ID
# GROUP BY
#     p.PRTY_ID,
#     p.PRTY_NM,
#     p.FRST_NM,
#     p.MDLE_NM,
#     p.LST_NM;


# When asked to perform multistep or complex tasks, you must first plan and then reflect on the steps.
# """
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

    TEMPERATURE = 0

    llm_config = {
        "llm-temperature": TEMPERATURE,
        "request_timeout": 120,
        "verbose": True,
        "model_name": st.session_state["agent_model_name"],
    }

    logger.info(f"\nllm-config = {json.dumps(llm_config)}")
    llm = AzureChatOpenAI(
        temperature=TEMPERATURE,
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
        agent_executor_kwargs={"return_intermediate_steps": True},
    )
    st.success("Agent setup done!")

    system_prompt_content = """ 
    If you get the question: 

    Show me the number of loan accounts and total loan amount by cost center. 
    Show me the top 20 cost centers with the highest loan amounts. 
    Order the output by descending loan amount


    Your sql query should be:


    SELECT TOP 20
        a.COST_CENTR_ID AS Cost_Center,
        COUNT(DISTINCT l.ACCT_ID) AS Number_of_Loan_Accounts,
        FORMAT(SUM(l.LN_ACCT_TOTAL_OWE), 'C', 'en-US') AS Total_Loan_Amount
    FROM 
        trg.ACCOUNT a
    JOIN 
        trg.LOAN l ON a.ACCT_ID = l.ACCT_ID
    GROUP BY 
        a.COST_CENTR_ID
    ORDER BY 
        SUM(l.LN_ACCT_TOTAL_OWE) DESC;
    """

if (
    "llm_sql_agent_messages" not in st.session_state
    or st.button("Clear message history")
    or not st.session_state.llm_sql_agent_messages
):
    st.session_state["llm_sql_agent_messages"] = [
        {"role":"system", "content": system_content},
        {"role": "assistant", "content": "How can I help you?"}
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
