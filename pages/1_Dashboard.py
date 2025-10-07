import streamlit as st
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from database.db_manager import DatabaseManager
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

st.title("📊 Dashboard")

user = st.session_state.user
db_manager = st.session_state.db_manager

# Pagination settings
SESSIONS_PER_PAGE = 5

# Initialize pagination state
if 'dashboard_page' not in st.session_state:
    st.session_state.dashboard_page = 1

# Cache database queries
@st.cache_data(ttl=180, show_spinner=False)
def get_session_count_cached(user_id):
    return db_manager.get_user_session_count(user_id)

@st.cache_data(ttl=180, show_spinner=False)
def get_sessions_cached(user_id):
    return db_manager.get_user_sessions(user_id)

# Module-level cache for predictions and metrics
@st.cache_data(ttl=300, show_spinner=False)
def get_session_predictions_cached(session_id):
    return db_manager.get_session_predictions(session_id)

@st.cache_data(ttl=300, show_spinner=False)
def get_session_metrics_cached(session_id):
    return db_manager.get_session_metrics(session_id)

# User statistics
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sessions = get_session_count_cached(user['id'])
    st.metric("Total Sessions", total_sessions)

with col2:
    st.metric("Active Models", "5")

with col3:
    st.metric("Channels", "20")

with col4:
    st.metric("Quantum Qubits", "4-8")

st.markdown("---")

# Recent sessions with pagination
st.subheader("📁 Recent Sessions")

sessions = get_sessions_cached(user['id'])

if sessions:
    total_sessions_count = len(sessions)
    total_pages = math.ceil(total_sessions_count / SESSIONS_PER_PAGE)
    
    # Pagination controls
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", key="dash_prev", disabled=(st.session_state.dashboard_page == 1)):
            st.session_state.dashboard_page -= 1
            st.rerun()
    
    with col2:
        st.write(f"Page {st.session_state.dashboard_page} of {total_pages}")
    
    with col3:
        st.write(f"Total: {total_sessions_count} sessions")
    
    with col4:
        if st.button("Next ➡️", key="dash_next", disabled=(st.session_state.dashboard_page >= total_pages)):
            st.session_state.dashboard_page += 1
            st.rerun()
    
    # Get paginated sessions
    start_idx = (st.session_state.dashboard_page - 1) * SESSIONS_PER_PAGE
    end_idx = min(start_idx + SESSIONS_PER_PAGE, total_sessions_count)
    paginated_sessions = sessions[start_idx:end_idx]
    
    # Display sessions
    for session in paginated_sessions:
        with st.expander(f"{session['filename']} - {Helpers.format_timestamp(session['upload_time'])}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Status:** {session['processing_status']}")
                st.write(f"**Original Channels:** {session['channels_original']}")
                st.write(f"**Selected Channels:** {session['channels_selected']}")
            
            with col2:
                # Get predictions for this session (with module-level caching)
                predictions = get_session_predictions_cached(session['id'])
                if predictions:
                    st.write(f"**Models Run:** {len(predictions)}")
                    best_acc = max([p['accuracy'] for p in predictions if p['accuracy']])
                    st.write(f"**Best Accuracy:** {best_acc:.2%}")
                
                # Get metrics (with module-level caching)
                metrics = get_session_metrics_cached(session['id'])
                if metrics:
                    st.write(f"**Brain State:** {metrics.get('brain_state', 'N/A')}")
else:
    st.info("No sessions found. Upload an EDF file to get started!")

st.markdown("---")

# Activity summary
st.subheader("📈 Recent Activity")

recent_activity = db_manager.get_recent_activity(user['id'], limit=10)

if recent_activity:
    for activity in recent_activity:
        st.text(f"{Helpers.format_timestamp(activity['timestamp'])} - {activity['action']}: {activity['details']}")
else:
    st.info("No recent activity")
