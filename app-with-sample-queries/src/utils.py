import streamlit as st
from loguru import logger

def load_schema_from_file(filename):
    config_dir_path = st.session_state["config_dir_path"]
    try:
        schema_file_path = config_dir_path / filename
        with open(schema_file_path, 'r') as f:
            schema = f.read()
        
        assert schema is not None
        ##### TRUNCATE SCHEMA 
        schema = schema[:30000]
        ##### TRUNCATED SCHEMA
        st.session_state["uploaded_schema"] = schema
        return schema
    
    except Exception as e:
        logger.error(str(e))
        st.error(e)

schema = load_schema_from_file(st.session_state["default_schema_filename"])
st.session_state["default_schema"] = schema
logger.info(f"first 500 chars of default schema is {schema[:500]}")
