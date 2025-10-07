import streamlit as st
import sys
from pathlib import Path
from datetime import datetime, timedelta
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from auth.authenticator import AuthManager

st.set_page_config(
    page_title="Password Reset - QuantumBCI",
    page_icon="🔑",
    layout="centered"
)

# Initialize database
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

db_manager = st.session_state.db_manager

# Hide sidebar on password reset page
st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

st.title("🔑 Password Reset")

tab1, tab2 = st.tabs(["Request Reset", "Reset with Token"])

with tab1:
    st.subheader("Request Password Reset")
    st.write("Enter your registered email address to receive a password reset link.")
    
    email = st.text_input("Email Address", key="reset_email")
    
    if st.button("Send Reset Link", type="primary", use_container_width=True):
        if not email:
            st.error("Please enter your email address")
        else:
            # Check if user exists
            user = db_manager.get_user_by_email(email)
            
            if user:
                # Generate secure token
                token = secrets.token_urlsafe(32)
                expires_at = datetime.now() + timedelta(hours=1)
                
                # Save token to database
                success = db_manager.create_password_reset_token(
                    user['id'], token, expires_at.strftime('%Y-%m-%d %H:%M:%S')
                )
                
                if success:
                    # Get the Replit app URL
                    replit_url = "http://0.0.0.0:5000"
                    reset_link = f"{replit_url}/?reset_token={token}"
                    
                    st.success("✅ Password reset instructions sent!")
                    st.info(f"""
                    **Password Reset Link:**
                    
                    Copy this link to reset your password:
                    ```
                    {reset_link}
                    ```
                    
                    This link will expire in 1 hour.
                    
                    **Note:** In a production environment, this link would be sent to your email ({email}).
                    For this demo, please copy the link above and use it in the "Reset with Token" tab.
                    """)
                    
                    # In production, send email here
                    # send_reset_email(email, reset_link, user['full_name'])
                else:
                    st.error("Failed to generate reset token. Please try again.")
            else:
                # Don't reveal if email exists for security
                st.success("✅ If this email is registered, you will receive reset instructions.")
    
    st.markdown("---")
    if st.button("← Back to Login"):
        st.switch_page("app.py")

with tab2:
    st.subheader("Reset Password with Token")
    st.write("Enter the token from your reset link and choose a new password.")
    
    token_input = st.text_input("Reset Token", key="token_input", help="Copy from the reset link")
    new_password = st.text_input("New Password", type="password", key="new_pass")
    confirm_password = st.text_input("Confirm Password", type="password", key="confirm_pass")
    
    if st.button("Reset Password", type="primary", use_container_width=True):
        if not token_input or not new_password or not confirm_password:
            st.error("All fields are required")
        elif new_password != confirm_password:
            st.error("Passwords do not match")
        elif len(new_password) < 6:
            st.error("Password must be at least 6 characters")
        else:
            # Verify token
            token_data = db_manager.get_password_reset_token(token_input)
            
            if not token_data:
                st.error("Invalid or expired reset token")
            elif token_data['used'] == 1:
                st.error("This reset token has already been used")
            else:
                # Update password
                auth_manager = AuthManager(db_manager)
                import bcrypt
                password_hash = bcrypt.hashpw(
                    new_password.encode('utf-8'), 
                    bcrypt.gensalt()
                ).decode('utf-8')
                
                success = db_manager.update_user_password(
                    token_data['user_id'], 
                    password_hash
                )
                
                if success:
                    # Mark token as used
                    db_manager.mark_token_used(token_data['id'])
                    
                    st.success("✅ Password reset successful!")
                    st.info("You can now login with your new password.")
                    
                    if st.button("Go to Login"):
                        st.switch_page("app.py")
                else:
                    st.error("Failed to update password. Please try again.")
    
    st.markdown("---")
    if st.button("← Back to Login", key="back2"):
        st.switch_page("app.py")

st.markdown("---")
st.caption("🔒 Secure password reset system powered by QuantumBCI")
