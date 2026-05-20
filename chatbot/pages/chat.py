"""Chat interface — main conversation page with tool-call badges."""

import os
import uuid

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

TOOL_BADGE_COLORS = {
    "classify": "#1f77b4",
    "extract_entities": "#ff7f0e",
    "summarize": "#2ca02c",
    "rag_search": "#9467bd",
    "write_memory": "#d62728",
}


def _auth_headers() -> dict[str, str]:
    token = st.session_state.get("jwt", "")
    return {"Authorization": f"Bearer {token}"}


def _render_tool_badge(tool_name: str) -> str:
    color = TOOL_BADGE_COLORS.get(tool_name, "#666")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75em;margin-right:4px">{tool_name}</span>'


def chat_page() -> None:
    st.title("Maintainer's Copilot")

    if "conversation_id" not in st.session_state:
        st.session_state["conversation_id"] = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Sidebar controls
    with st.sidebar:
        st.subheader("Session")
        st.write(f"**Conversation:** `{st.session_state['conversation_id'][:8]}...`")
        if st.button("New conversation"):
            st.session_state["conversation_id"] = str(uuid.uuid4())
            st.session_state["messages"] = []
            st.rerun()
        st.divider()
        st.write(f"Logged in as **{st.session_state.get('email', '?')}**")
        if st.button("Log out"):
            for key in ["jwt", "email", "conversation_id", "messages"]:
                st.session_state.pop(key, None)
            st.rerun()

    # Render message history
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                badges = "".join(_render_tool_badge(tc["tool"]) for tc in msg["tool_calls"])
                st.markdown(f"**Tools called:** {badges}", unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Ask about a scikit-learn issue…")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = httpx.post(
                        f"{API_URL}/chat",
                        json={
                            "message": user_input,
                            "conversation_id": st.session_state["conversation_id"],
                        },
                        headers=_auth_headers(),
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        reply = data["reply"]
                        tool_calls = data.get("tool_calls_made", [])
                        st.markdown(reply)
                        if tool_calls:
                            badges = "".join(_render_tool_badge(tc["tool"]) for tc in tool_calls)
                            st.markdown(f"**Tools called:** {badges}", unsafe_allow_html=True)
                        st.session_state["messages"].append(
                            {"role": "assistant", "content": reply, "tool_calls": tool_calls}
                        )
                        # Update conversation_id in case the server created a new one
                        st.session_state["conversation_id"] = str(data["conversation_id"])
                    elif resp.status_code == 401:
                        st.error("Session expired — please log in again.")
                        st.session_state.pop("jwt", None)
                        st.rerun()
                    else:
                        st.error(f"API error (HTTP {resp.status_code}): {resp.text[:200]}")
                except httpx.ConnectError:
                    st.error("Cannot reach the API server. Is it running?")
                except httpx.TimeoutException:
                    st.error("Request timed out (the model is taking too long).")
