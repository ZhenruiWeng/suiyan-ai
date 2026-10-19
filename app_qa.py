import streamlit as st
from streamlit import divider
import time
import config_data as config
from rag import RagService

st.title("智能助手")
st.divider()

if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助"}]

if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    with st.spinner("AI思考中..."):
        res = st.session_state["rag"].chain.invoke({"input": prompt},config.session_config)
        ans=res
        st.chat_message("assistant").write(ans)
        st.session_state["message"].append({"role": "assistant", "content": ans})