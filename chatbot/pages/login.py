"""Login page — email/password → JWT stored in st.session_state."""

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")


def login_page() -> None:
    st.title("Maintainer's Copilot — Login")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Log in")

    if submitted:
        if not email or not password:
            st.error("Please enter your email and password.")
            return

        try:
            resp = httpx.post(
                f"{API_URL}/auth/jwt/login",
                data={"username": email, "password": password},
                timeout=10,
            )
            if resp.status_code == 200:
                token = resp.json().get("access_token", "")
                st.session_state["jwt"] = token
                st.session_state["email"] = email
                st.success("Logged in successfully!")
                st.rerun()
            elif resp.status_code == 400:
                st.error("Invalid email or password.")
            else:
                st.error(f"Login failed (HTTP {resp.status_code}).")
        except httpx.ConnectError:
            st.error("Cannot reach the API server. Is it running?")

    st.divider()
    st.caption("Don't have an account? Ask an admin to create one for you.")
