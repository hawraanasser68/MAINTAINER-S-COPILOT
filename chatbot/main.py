"""Streamlit app entry point — handles auth gate and page routing."""

import streamlit as st

from chatbot.pages.admin import admin_page
from chatbot.pages.chat import chat_page
from chatbot.pages.login import login_page
from chatbot.pages.memory import memory_page

st.set_page_config(
    page_title="Maintainer's Copilot",
    page_icon="🔧",
    layout="wide",
)

# Auth gate — redirect to login if not authenticated
if "jwt" not in st.session_state:
    login_page()
    st.stop()

# Sidebar navigation
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Chat", "Memory Inspector", "Admin"],
        label_visibility="collapsed",
    )

if page == "Chat":
    chat_page()
elif page == "Memory Inspector":
    memory_page()
elif page == "Admin":
    admin_page()
