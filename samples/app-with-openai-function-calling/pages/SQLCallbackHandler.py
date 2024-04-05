from langchain.callbacks.base import BaseCallbackHandler
from langchain.agents import create_sql_agent
...
class SQLCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.sql_result = None

    def on_agent_action(self, action, **kwargs):
        """Run on agent action. if the tool being used is sql_db_query,
         it means we're submitting the sql and we can 
         record it as the final sql"""

        if action.tool == "sql_db_query":
            self.sql_result = action.tool_input
...
db_agent = create_sql_agent(llm=llm,  
                  toolkit=toolkit,agent_type=AgentType.OPENAI_FUNCTIONS,
                  top_k=100,extra_tools=custom_tool_list,
                  prefix=custom_prefix,
                  functions_suffix=custom_functions_suffix,
                  suffix=custom_suffix,verbose=True,
                  max_execution_time=30000,max_iterations=1000,
                  handle_parsing_errors=True,
                  handle_sql_errors=True,
            )
...
result = db_agent.invoke({"input": question},
                        {"callbacks": [handler]}
            )
print(">>Handler SQL Result: ", handler.sql_result)