"""Admin page — invite users, create widget configs. Admins only."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {st.session_state.get('jwt', '')}"}


def admin_page() -> None:
    st.title("Admin Panel")
    st.caption("This page is only visible to admin users.")

    tab_users, tab_widgets = st.tabs(["Users", "Widget Configs"])

    # -----------------------------------------------------------------------
    with tab_users:
        st.subheader("Invite / Register a New User")
        with st.form("register_form"):
            new_email = st.text_input("Email")
            new_password = st.text_input("Password", type="password")
            is_superuser = st.checkbox("Grant admin role")
            submitted = st.form_submit_button("Create user")

        if submitted:
            if not new_email or not new_password:
                st.warning("Email and password are required.")
            else:
                try:
                    resp = httpx.post(
                        f"{API_URL}/auth/register",
                        json={
                            "email": new_email,
                            "password": new_password,
                            "is_superuser": is_superuser,
                        },
                        headers=_auth_headers(),
                        timeout=10,
                    )
                    if resp.status_code in (200, 201):
                        st.success(f"User {new_email} created.")
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", "Unknown error")
                        st.error(f"Registration failed: {detail}")
                    elif resp.status_code == 401:
                        st.error("Not authenticated.")
                    else:
                        st.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
                except httpx.ConnectError:
                    st.error("Cannot reach API server.")

    # -----------------------------------------------------------------------
    with tab_widgets:
        st.subheader("Create Widget Configuration")
        st.caption(
            "Widgets let you embed the copilot on external pages with a per-origin allow-list."
        )

        with st.form("widget_form"):
            allowed_origins_raw = st.text_area(
                "Allowed origins (one per line)",
                placeholder="https://example.com\nhttps://staging.example.com",
            )
            greeting = st.text_input(
                "Greeting message", value="Hi! How can I help you with scikit-learn?"
            )
            enabled_tools_raw = st.multiselect(
                "Enabled tools",
                ["classify", "extract_entities", "summarize", "rag_search", "write_memory"],
                default=["classify", "rag_search", "summarize"],
            )
            theme_primary = st.color_picker("Primary colour", "#1f77b4")
            submitted_widget = st.form_submit_button("Create widget")

        if submitted_widget:
            allowed_origins = [o.strip() for o in allowed_origins_raw.splitlines() if o.strip()]
            try:
                resp = httpx.post(
                    f"{API_URL}/widgets",
                    json={
                        "allowed_origins": allowed_origins,
                        "greeting": greeting,
                        "enabled_tools": enabled_tools_raw,
                        "theme": {"primary": theme_primary},
                    },
                    headers=_auth_headers(),
                    timeout=10,
                )
                if resp.status_code in (200, 201):
                    data = resp.json()
                    st.success(f"Widget created. ID: `{data.get('widget_id', '?')}`")
                elif resp.status_code == 401:
                    st.error("Not authenticated or not an admin.")
                else:
                    st.error(f"HTTP {resp.status_code}: {resp.text[:200]}")
            except httpx.ConnectError:
                st.error("Cannot reach API server.")
