#!/usr/bin/env python
# coding: utf-8

# # Async
# 
# In this example we will build a chat executor with native async implementations of the core logic. This enables taking advantage of Chat Models which have async clients, removing the need for calling the model in a separate thread.

# ## Setup
# 
# First we need to install the packages required

# In[1]:


get_ipython().system('pip install --quiet -U langchain langchain_openai tavily-python')


# Next, we need to set API keys for OpenAI (the LLM we will use) and Tavily (the search tool we will use)

# In[2]:


import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API Key:")


# Optionally, we can set API key for [LangSmith tracing](https://smith.langchain.com/), which will give us best-in-class observability.

# In[ ]:


os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")


# ## Set up the tools
# 
# We will first define the tools we want to use.
# For this simple example, we will use a built-in search tool via Tavily.
# However, it is really easy to create your own tools - see documentation [here](https://python.langchain.com/docs/modules/agents/tools/custom_tools) on how to do that.
# 

# In[3]:


from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=1)]


# We can now wrap these tools in a simple ToolExecutor.
# This is a real simple class that takes in a ToolInvocation and calls that tool, returning the output.
# A ToolInvocation is any class with `tool` and `tool_input` attribute.
# 

# In[4]:


from langgraph.prebuilt import ToolExecutor

tool_executor = ToolExecutor(tools)


# ## Set up the model
# 
# Now we need to load the chat model we want to use.
# Importantly, this should satisfy two criteria:
# 
# 1. It should work with messages. We will represent all agent state in the form of messages, so it needs to be able to work well with them.
# 2. It should work with OpenAI function calling. This means it should either be an OpenAI model or a model that exposes a similar interface.
# 
# Note: these model requirements are not requirements for using LangGraph - they are just requirements for this one example.
# 

# In[5]:


from langchain_openai import ChatOpenAI

# We will set streaming=True so that we can stream tokens
# See the streaming section for more information on this.
model = ChatOpenAI(temperature=0, streaming=True)


# 
# After we've done this, we should make sure the model knows that it has these tools available to call.
# We can do this by converting the LangChain tools into the format for OpenAI function calling, and then bind them to the model class.
# 

# In[6]:


from langchain.tools.render import format_tool_to_openai_function

functions = [format_tool_to_openai_function(t) for t in tools]
model = model.bind_functions(functions)


# ## Define the agent state
# 
# The main type of graph in `langgraph` is the `StatefulGraph`.
# This graph is parameterized by a state object that it passes around to each node.
# Each node then returns operations to update that state.
# These operations can either SET specific attributes on the state (e.g. overwrite the existing values) or ADD to the existing attribute.
# Whether to set or add is denoted by annotating the state object you construct the graph with.
# 
# For this example, the state we will track will just be a list of messages.
# We want each node to just add messages to that list.
# Therefore, we will use a `TypedDict` with one key (`messages`) and annotate it so that the `messages` attribute is always added to.
# 

# In[7]:


from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# ## Define the nodes
# 
# We now need to define a few different nodes in our graph.
# In `langgraph`, a node can be either a function or a [runnable](https://python.langchain.com/docs/expression_language/).
# There are two main nodes we need for this:
# 
# 1. The agent: responsible for deciding what (if any) actions to take.
# 2. A function to invoke tools: if the agent decides to take an action, this node will then execute that action.
# 
# We will also need to define some edges.
# Some of these edges may be conditional.
# The reason they are conditional is that based on the output of a node, one of several paths may be taken.
# The path that is taken is not known until that node is run (the LLM decides).
# 
# 1. Conditional Edge: after the agent is called, we should either:
#    a. If the agent said to take an action, then the function to invoke tools should be called
#    b. If the agent said that it was finished, then it should finish
# 2. Normal Edge: after the tools are invoked, it should always go back to the agent to decide what to do next
# 
# Let's define the nodes, as well as a function to decide how what conditional edge to take.
# 
# **MODIFICATION**
# 
# We define each node as an async function.

# In[8]:


from langgraph.prebuilt import ToolInvocation
import json
from langchain_core.messages import FunctionMessage


# Define the function that determines whether to continue or not
def should_continue(state):
    messages = state["messages"]
    last_message = messages[-1]
    # If there is no function call, then we finish
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "continue"


# Define the function that calls the model
async def call_model(state):
    messages = state["messages"]
    response = await model.ainvoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# Define the function to execute tools
async def call_tool(state):
    messages = state["messages"]
    # Based on the continue condition
    # we know the last message involves a function call
    last_message = messages[-1]
    # We construct an ToolInvocation from the function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=json.loads(
            last_message.additional_kwargs["function_call"]["arguments"]
        ),
    )
    # We call the tool_executor and get back a response
    response = await tool_executor.ainvoke(action)
    # We use the response to create a FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # We return a list, because this will get added to the existing list
    return {"messages": [function_message]}


# ## Define the graph
# 
# We can now put it all together and define the graph!

# In[9]:


from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.set_entry_point("agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
    # Finally we pass in a mapping.
    # The keys are strings, and the values are other nodes.
    # END is a special node marking that the graph should finish.
    # What will happen is we will call `should_continue`, and then the output of that
    # will be matched against the keys in this mapping.
    # Based on which one it matches, that node will then be called.
    {
        # If `tools`, then we call the tool node.
        "continue": "action",
        # Otherwise we finish.
        "end": END,
    },
)

# We now add a normal edge from `tools` to `agent`.
# This means that after `tools` is called, `agent` node is called next.
workflow.add_edge("action", "agent")

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()


# ## Use it!
# 
# We can now use it!
# This now exposes the [same interface](https://python.langchain.com/docs/expression_language/) as all other LangChain runnables.

# In[10]:


from langchain_core.messages import HumanMessage

inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
await app.ainvoke(inputs)


# This may take a little bit - it's making a few calls behind the scenes.
# In order to start seeing some intermediate results as they happen, we can use streaming - see below for more information on that.
# 
# ## Streaming
# 
# LangGraph has support for several different types of streaming.
# 
# ### Streaming Node Output
# 
# One of the benefits of using LangGraph is that it is easy to stream output as it's produced by each node.
# 

# In[11]:


inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
async for output in app.astream(inputs):
    # stream() yields dictionaries with output keyed by node name
    for key, value in output.items():
        print(f"Output from node '{key}':")
        print("---")
        print(value)
    print("\n---\n")


# ### Streaming LLM Tokens
# 
# You can also access the LLM tokens as they are produced by each node. 
# In this case only the "agent" node produces LLM tokens.
# In order for this to work properly, you must be using an LLM that supports streaming as well as have set it when constructing the LLM (e.g. `ChatOpenAI(model="gpt-3.5-turbo-1106", streaming=True)`)
# 

# In[12]:


inputs = {"messages": [HumanMessage(content="what is the weather in sf")]}
async for output in app.astream_log(inputs, include_types=["llm"]):
    # astream_log() yields the requested logs (here LLMs) in JSONPatch format
    for op in output.ops:
        if op["path"] == "/streamed_output/-":
            # this is the output from .stream()
            ...
        elif op["path"].startswith("/logs/") and op["path"].endswith(
            "/streamed_output/-"
        ):
            # because we chose to only include LLMs, these are LLM tokens
            print(op["value"])


# In[ ]:




