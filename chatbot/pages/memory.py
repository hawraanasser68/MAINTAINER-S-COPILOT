"""Memory inspector — shows short-term (Redis) and long-term (pgvector) memories."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('jwt', '')}"}


def memory_page() -> None:
    st.title("Memory Inspector")

    tab_short, tab_long = st.tabs(["Short-term (Redis)", "Long-term (pgvector)"])

    with tab_short:
        st.subheader("Current Conversation History")
        convo_id = st.session_state.get("conversation_id", "")
        if not convo_id:
            st.info("No active conversation.")
        else:
            try:
                resp = httpx.get(
                    f"{API_URL}/chat/history/{convo_id}",
                    headers=_auth_headers(),
                    timeout=10,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    messages = data.get("messages", [])
                    if not messages:
                        st.info("No messages in this conversation yet.")
                    else:
                        for msg in messages:
                            role_icon = "🧑" if msg["role"] == "user" else "🤖"
                            st.markdown(f"**{role_icon} {msg['role'].capitalize()}:** {msg['content']}")
                            st.divider()
                        st.caption("TTL: 3600 seconds — resets on each message")
                elif resp.status_code == 401:
                    st.error("Not authenticated.")
                else:
                    st.error(f"Error fetching history: HTTP {resp.status_code}")
            except httpx.ConnectError:
                st.error("Cannot reach API server.")

    with tab_long:
        st.subheader("Long-term Semantic Memories")
        st.caption("Memories saved across sessions via the `write_memory` tool.")

        search_query = st.text_input("Search memories by semantic similarity", placeholder="e.g. sklearn version compatibility")
        if st.button("Search") and search_query:
            try:
                resp = httpx.get(
                    f"{API_URL}/memory/search",
                    params={"query": search_query, "top_k": 10},
                    headers=_auth_headers(),
                    timeout=15,
                )
                if resp.status_code == 200:
                    memories = resp.json().get("memories", [])
                    if not memories:
                        st.info("No matching memories found.")
                    else:
                        for mem in memories:
                            score = mem.get("score", 0)
                            st.markdown(f"**{mem['memory_type']}** (score: {score:.3f})")
                            st.text(mem["content"])
                            st.caption(f"Created: {mem.get('created_at', '?')}")
                            st.divider()
                elif resp.status_code == 401:
                    st.error("Not authenticated.")
                else:
                    st.error(f"Search failed: HTTP {resp.status_code}")
            except httpx.ConnectError:
                st.error("Cannot reach API server.")

        st.subheader("Recent Memories")
        try:
            resp = httpx.get(
                f"{API_URL}/memory",
                params={"limit": 20},
                headers=_auth_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                memories = resp.json().get("memories", [])
                if not memories:
                    st.info("No long-term memories stored yet.")
                else:
                    for mem in memories:
                        st.markdown(f"**{mem['memory_type']}** — {mem['content']}")
                        st.caption(f"Saved: {mem.get('created_at', '?')}")
                        st.divider()
            elif resp.status_code == 401:
                st.error("Not authenticated.")
        except httpx.ConnectError:
            st.error("Cannot reach API server.")
