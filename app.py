import os
import streamlit as st


LANGCHAIN_PROJECT = "Multipage App"
st.set_page_config(page_title=LANGCHAIN_PROJECT, page_icon="")
st.title(LANGCHAIN_PROJECT)
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT

st.sidebar.success("Select a app to use from above.",)