import streamlit as st
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from auth.authenticator import AuthManager
from database.db_manager import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="QuantumBCI - Signal Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user' not in st.session_state:
    st.session_state.user = None
if 'db_manager' not in st.session_state:
    st.session_state.db_manager = DatabaseManager()

# Immediately hide sidebar on login page - inject BEFORE any rendering
if not st.session_state.authenticated:
    st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        min-width: 0 !important;
        opacity: 0 !important;
    }
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    </style>
    """, unsafe_allow_html=True)
def main():
    """Main application entry point"""
    
    # Initialize database ONCE - not on every rerun!
    if 'db_initialized' not in st.session_state:
        st.session_state.db_manager.initialize_database()
        st.session_state.db_initialized = True
    
    # Check authentication
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_main_app()

def show_login_page():
    """Display login/registration page"""
    
    # Display background image
    import base64
    from pathlib import Path
    
    # Styling for text with better visibility against background
    st.markdown("""
    <style>
    .main-title {
        color: #FFFFFF !important;
        text-align: center;
        text-shadow: 3px 3px 8px rgba(0,0,0,0.9), 0 0 15px rgba(138,43,226,0.8), 0 0 25px rgba(138,43,226,0.6);
        margin-bottom: 10px;
        font-size: 3.5em;
        font-weight: 700;
        background: linear-gradient(135deg, #FFFFFF, #FFB6F0, #E6A4FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        filter: drop-shadow(2px 2px 6px rgba(0,0,0,0.95)) drop-shadow(0 0 10px rgba(138,43,226,0.7));
    }
    .subtitle {
        color: #FFFFFF !important;
        text-align: center;
        text-shadow: 2px 2px 6px rgba(0,0,0,0.95), 0 0 10px rgba(138,43,226,0.6);
        font-size: 1.5em;
        font-weight: 500;
    }
    
    /* Login/Register container styling */
    .stTabs {
        background: rgba(10, 10, 30, 0.85) !important;
        backdrop-filter: blur(10px);
        border-radius: 15px !important;
        padding: 25px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6), 0 0 20px rgba(138, 43, 226, 0.3) !important;
        border: 1px solid rgba(138, 43, 226, 0.3) !important;
    }
    
    /* Tab buttons styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(20, 20, 40, 0.6) !important;
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #CCCCCC !important;
        background: rgba(30, 30, 50, 0.6) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #8B5CF6, #6366F1) !important;
        color: #FFFFFF !important;
    }
    
    /* Subheader styling */
    h3 {
        color: #FFFFFF !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
        font-weight: 600 !important;
    }
    
    /* Input fields styling */
    .stTextInput input, .stSelectbox select {
        background: rgba(30, 30, 60, 0.8) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(138, 43, 226, 0.4) !important;
        border-radius: 8px !important;
    }
    
    .stTextInput input:focus, .stSelectbox select:focus {
        border: 1px solid rgba(138, 43, 226, 0.8) !important;
        box-shadow: 0 0 10px rgba(138, 43, 226, 0.4) !important;
    }
    
    /* Label styling */
    .stTextInput label, .stSelectbox label {
        color: #E0E0E0 !important;
        font-weight: 500 !important;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.8);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #8B5CF6, #6366F1) !important;
        color: #FFFFFF !important;
        border: none !important;
        font-weight: 600 !important;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #7C3AED, #4F46E5) !important;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Add brain with electrical signals background if available
    bg_image_path = Path("attached_assets/Gemini_Generated_Image_iwhzm0iwhzm0iwhz_1759748395151.png")
    if bg_image_path.exists():
        with open(bg_image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode()
            st.markdown(f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(15, 0, 30, 0.4), rgba(25, 0, 50, 0.5)), url(data:image/jpeg;base64,{encoded});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
            """, unsafe_allow_html=True)
    else:
        # Fallback to gradient
        st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-title">🧠 QuantumBCI Signal Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<h3 class="subtitle">Quantum Machine Learning for Brain-Computer Interface Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("---")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to Your Account")
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.button("Login", type="primary", use_container_width=True):
                auth_manager = AuthManager(st.session_state.db_manager)
                user = auth_manager.authenticate_user(username, password)
                
                if user:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    
                    st.session_state.db_manager.log_activity(
                        user['id'], 
                        'login', 
                        f"User {username} logged in"
                    )
                    st.success(f"Welcome back, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            
            # Password reset link
            st.markdown("---")
            col_a, col_b, col_c = st.columns([1, 2, 1])
            with col_b:
                if st.button("🔑 Forgot Password?", use_container_width=True):
                    st.switch_page("pages/6_Password_Reset.py")
        
        with tab2:
            st.subheader("Create New Account")
            reg_username = st.text_input("Username", key="reg_username")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
            reg_fullname = st.text_input("Full Name", key="reg_fullname")
            reg_email = st.text_input("Email", key="reg_email")
            reg_role = st.selectbox("Role", ["Researcher", "Doctor"], key="reg_role")
            
            if st.button("Register", type="primary", use_container_width=True):
                if reg_password != reg_confirm:
                    st.error("Passwords do not match")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not reg_username or not reg_fullname or not reg_email:
                    st.error("All fields are required")
                else:
                    auth_manager = AuthManager(st.session_state.db_manager)
                    success, message = auth_manager.register_user(
                        reg_username, reg_password, reg_fullname, 
                        reg_email, reg_role.lower()
                    )
                    
                    if success:
                        st.success(message)
                        st.info("Please login with your credentials")
                    else:
                        st.error(message)

def show_main_app():
    """Display main application interface"""
    
    # Sidebar
    with st.sidebar:
        st.title("🧠 QuantumBCI")
        st.markdown(f"**User:** {st.session_state.user['full_name']}")
        st.markdown(f"**Role:** {st.session_state.user['role'].title()}")
        st.markdown("---")
        
        # Navigation info
        st.markdown("### Navigation")
        st.info("Use the sidebar pages to navigate through the application")
        
        st.markdown("---")
        
        if st.button("Logout", type="primary", use_container_width=True):
            st.session_state.db_manager.log_activity(
                st.session_state.user['id'],
                'logout',
                f"User {st.session_state.user['username']} logged out"
            )
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()
    
    # Main content
    st.title("Welcome to QuantumBCI Analysis Platform")
    
    st.markdown("""
    ### 🎯 System Overview
    
    This advanced Brain-Computer Interface (BCI) signal prediction system leverages **Quantum Machine Learning** 
    to analyze and predict EEG signals with unprecedented accuracy and efficiency.
    
    #### Key Features:
    - **20-Channel EEG Analysis**: Optimized channel selection from 64-channel data
    - **Quantum ML Models**: QSVM and Variational Quantum Classifiers using PennyLane
    - **Classical Comparison**: Benchmark against SVM and Random Forest models
    - **Real-time Processing**: Upload EDF files for instant analysis
    - **Brain Metrics**: Alpha, Beta, Theta, Delta band analysis
    - **PDF Reports**: Comprehensive session reports with visualizations
    - **Secure Storage**: Encrypted data handling and role-based access
    
    #### Quick Start:
    1. Navigate to **Upload & Analysis** to process EEG data
    2. View results in **Results** page
    3. Download PDF reports for documentation
    4. Manage users and view logs (Admin only)
    """)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_sessions = st.session_state.db_manager.get_user_session_count(
            st.session_state.user['id']
        )
        st.metric("Total Sessions", total_sessions)
    
    with col2:
        st.metric("Active Channels", "20")
    
    with col3:
        st.metric("ML Models", "5")
    
    with col4:
        st.metric("Quantum Qubits", "4-8")
    
    st.markdown("---")
    
    # System information
    st.markdown("### 📊 System Capabilities")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Quantum Models
        - Quantum Support Vector Machine (QSVM)
        - Variational Quantum Classifier (VQC)
        - Quantum Kernel Estimation
        - Amplitude Encoding
        """)
    
    with col2:
        st.markdown("""
        #### Classical Models
        - Support Vector Machine (SVM)
        - Random Forest Classifier
        - Feature Extraction (PCA, FFT)
        - Statistical Analysis
        """)
    
    st.markdown("---")
    
    # Recent activity
    st.markdown("### 📈 Recent Activity")
    recent_logs = st.session_state.db_manager.get_recent_activity(
        st.session_state.user['id'], limit=5
    )
    
    if recent_logs:
        for log in recent_logs:
            st.text(f"{log['timestamp']} - {log['action']}: {log['details']}")
    else:
        st.info("No recent activity")

if __name__ == "__main__":
    main()
