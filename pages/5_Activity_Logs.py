import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from auth.authenticator import AuthManager
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

user = st.session_state.user
db_manager = st.session_state.db_manager
auth_manager = AuthManager(db_manager)

st.title("📜 Activity Logs")

# Pagination settings
LOGS_PER_PAGE = 20

# Initialize pagination state
if 'logs_page' not in st.session_state:
    st.session_state.logs_page = 1

# Check permissions
is_admin = auth_manager.is_admin(user)

# Cache database queries
@st.cache_data(ttl=120, show_spinner=False)
def get_all_logs_cached(limit):
    return db_manager.get_all_activity_logs(limit=limit)

@st.cache_data(ttl=120, show_spinner=False)
def get_user_logs_cached(user_id, limit):
    return db_manager.get_recent_activity(user_id, limit=limit)

if is_admin:
    st.subheader("All System Activity (Admin View)")
    
    # Get all activity logs
    all_logs = get_all_logs_cached(500)
    
    if all_logs:
        total_logs = len(all_logs)
        total_pages = math.ceil(total_logs / LOGS_PER_PAGE)
        
        # Pagination controls
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            if st.button("⬅️ Prev", key="admin_prev", disabled=(st.session_state.logs_page == 1)):
                st.session_state.logs_page -= 1
                st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.logs_page} of {total_pages}")
        
        with col3:
            st.write(f"Total: {total_logs} logs")
        
        with col4:
            if st.button("Next ➡️", key="admin_next", disabled=(st.session_state.logs_page >= total_pages)):
                st.session_state.logs_page += 1
                st.rerun()
        
        # Get paginated logs
        start_idx = (st.session_state.logs_page - 1) * LOGS_PER_PAGE
        end_idx = min(start_idx + LOGS_PER_PAGE, total_logs)
        paginated_logs = all_logs[start_idx:end_idx]
        
        # Convert to DataFrame for better display
        df_data = []
        for log in paginated_logs:
            df_data.append({
                'Timestamp': Helpers.format_timestamp(log['timestamp']),
                'User': f"{log['full_name']} ({log['username']})",
                'Action': log['action'],
                'Details': log['details']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No activity logs found")

else:
    st.subheader("Your Activity")
    
    # Get user's activity logs
    user_logs = get_user_logs_cached(user['id'], 200)
    
    if user_logs:
        total_logs = len(user_logs)
        total_pages = math.ceil(total_logs / LOGS_PER_PAGE)
        
        # Pagination controls
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        
        with col1:
            if st.button("⬅️ Prev", key="user_prev", disabled=(st.session_state.logs_page == 1)):
                st.session_state.logs_page -= 1
                st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.logs_page} of {total_pages}")
        
        with col3:
            st.write(f"Total: {total_logs} logs")
        
        with col4:
            if st.button("Next ➡️", key="user_next", disabled=(st.session_state.logs_page >= total_pages)):
                st.session_state.logs_page += 1
                st.rerun()
        
        # Get paginated logs
        start_idx = (st.session_state.logs_page - 1) * LOGS_PER_PAGE
        end_idx = min(start_idx + LOGS_PER_PAGE, total_logs)
        paginated_logs = user_logs[start_idx:end_idx]
        
        # Convert to DataFrame
        df_data = []
        for log in paginated_logs:
            df_data.append({
                'Timestamp': Helpers.format_timestamp(log['timestamp']),
                'Action': log['action'],
                'Details': log['details']
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True, height=600)
    else:
        st.info("No activity logs found")

# Activity statistics
st.markdown("---")
st.subheader("📊 Activity Summary")

if is_admin:
    all_logs = db_manager.get_all_activity_logs(limit=1000)
    
    if all_logs:
        # Action type distribution
        action_counts = {}
        for log in all_logs:
            action = log['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Actions", len(all_logs))
        
        with col2:
            st.metric("Unique Actions", len(action_counts))
        
        with col3:
            most_common = max(action_counts.items(), key=lambda x: x[1])[0]
            st.metric("Most Common", most_common)
        
        # Action breakdown
        st.markdown("#### Action Breakdown")
        
        action_df = pd.DataFrame([
            {'Action': k, 'Count': v} 
            for k, v in sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.bar_chart(action_df.set_index('Action'))

else:
    user_logs = db_manager.get_recent_activity(user['id'], limit=1000)
    
    if user_logs:
        # Action type distribution
        action_counts = {}
        for log in user_logs:
            action = log['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Your Total Actions", len(user_logs))
        
        with col2:
            st.metric("Action Types", len(action_counts))
        
        # Action breakdown
        st.markdown("#### Your Activity Breakdown")
        
        for action, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{action.title()}:** {count}")
