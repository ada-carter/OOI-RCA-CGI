import urllib.parse
import requests
import streamlit as st
import logging
from core.config import settings

logger = logging.getLogger(__name__)

# Google OAuth2 endpoints
AUTHORIZATION_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"

def get_redirect_uri():
    """Dynamically determine the redirect URI based on the host."""
    return settings.OAUTH_REDIRECT_URI

def get_login_url():
    """Generate the Google OAuth2 login URL."""
    params = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "redirect_uri": get_redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "access_type": "online",
        "prompt": "select_account"
    }
    url = f"{AUTHORIZATION_URL}?{urllib.parse.urlencode(params)}"
    return url

def exchange_code(code: str):
    """Exchange the authorization code for an access token."""
    data = {
        "client_id": settings.GOOGLE_CLIENT_ID,
        "client_secret": settings.GOOGLE_CLIENT_SECRET,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": get_redirect_uri()
    }
    response = requests.post(TOKEN_URL, data=data)
    response.raise_for_status()
    return response.json()

def get_user_info(access_token: str):
    """Fetch user profile information from Google."""
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(USERINFO_URL, headers=headers)
    response.raise_for_status()
    return response.json()

def require_auth():
    """
    Call this at the top of app.py to enforce login.
    Returns True if authenticated, False otherwise.
    """
    if "user_id" in st.session_state:
        return True
        
    # Check if we are returning from OAuth flow
    if "code" in st.query_params:
        code = st.query_params["code"]
        try:
            tokens = exchange_code(code)
            user_info = get_user_info(tokens["access_token"])
            
            # Upsert user into DB
            import sys
            import os
            _backend = os.path.abspath('backend')
            if _backend not in sys.path:
                sys.path.insert(0, _backend)
            from db.database import SessionLocal
            from db.models import User
            
            with SessionLocal() as db:
                user = db.query(User).filter(User.id == user_info["id"]).first()
                if not user:
                    user = User(
                        id=user_info["id"],
                        email=user_info["email"],
                        name=user_info.get("name", ""),
                        picture=user_info.get("picture", "")
                    )
                    db.add(user)
                else:
                    user.name = user_info.get("name", "")
                    user.picture = user_info.get("picture", "")
                db.commit()
            
            st.session_state.user_id = user_info["id"]
            st.session_state.user_email = user_info["email"]
            st.session_state.user_name = user_info.get("name", "User")
            st.session_state.user_picture = user_info.get("picture", "")
            
            # Clear query params
            st.query_params.clear()
            st.rerun()
        except Exception as e:
            logger.error(f"OAuth exchange failed: {e}")
            st.error("Authentication failed. Please try again.")
            
    return False

def render_login_page():
    """Render the login UI."""
    st.markdown("## OOI RCA Copilot")
    st.markdown("Please log in with your Google account to access the Copilot.")
    
    if not settings.GOOGLE_CLIENT_ID or not settings.GOOGLE_CLIENT_SECRET:
        st.error("⚠️ Google OAuth credentials are not configured in `secrets.toml`.")
        st.stop()
        
    login_url = get_login_url()
    st.markdown(f'<a href="{login_url}" target="_self"><button style="padding: 10px 20px; font-size: 16px; background-color: #4285F4; color: white; border: none; border-radius: 4px; cursor: pointer;">Sign in with Google</button></a>', unsafe_allow_html=True)
