import os
import streamlit as st
from template import ReActAgent

st.set_page_config(
    page_title="Vingroup ReAct Agent Demo",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Vingroup ReAct Agent")
st.caption("Lab #3 — Baseline Chatbot vs ReAct Agent")

with st.sidebar:
    st.header("⚙️ Settings")
    max_iterations = st.slider("Max iterations", 1, 10, 5)
    show_trace = st.checkbox("Show ReAct trace", value=True)

    if os.getenv("OPENAI_API_KEY"):
        st.success("OPENAI_API_KEY detected")
    else:
        st.warning("OPENAI_API_KEY chưa được cấu hình")

default_query = (
    "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, "
    "rồi cho biết thời tiết SGN nên mặc gì?"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_trace" not in st.session_state:
    st.session_state.last_trace = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Nhập câu hỏi cho ReAct Agent...")

if not st.session_state.messages:
    if st.button("✈️ Chạy câu hỏi mẫu", use_container_width=True):
        prompt = default_query

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Agent đang suy luận và gọi tools..."):
                agent = ReActAgent(max_iterations=max_iterations)
                result = agent.run(prompt)
                st.session_state.last_trace = agent.trace

            st.markdown(result)

        except Exception as e:
            result = f"Lỗi: {e}"
            st.error(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })

if show_trace and st.session_state.last_trace:
    st.divider()
    st.subheader("🔍 ReAct Trace")

    for item in st.session_state.last_trace:
        iteration = item.get("iteration", "?")

        with st.expander(f"Iteration {iteration}", expanded=False):
            if "thought" in item:
                st.markdown("**💭 Thought**")
                st.write(item["thought"])

            if "action" in item:
                st.markdown("**🔧 Action**")
                st.json(item["action"])

            if "observation" in item:
                st.markdown("**📊 Observation**")
                st.json(item["observation"])

            if "final_answer" in item:
                st.markdown("**✅ Final Answer**")
                st.write(item["final_answer"])

st.divider()
st.subheader("💡 Câu hỏi mẫu")

examples = [
    "Tìm chuyến bay từ HAN đi SGN dưới 2 triệu.",
    "Thời tiết SGN hôm nay nên mặc gì?",
    default_query,
]

cols = st.columns(3)

for col, example in zip(cols, examples):
    with col:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()
