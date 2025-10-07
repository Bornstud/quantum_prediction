import streamlit as st
import sys
from pathlib import Path
import os
import math

sys.path.insert(0, str(Path(__file__).parent.parent))

from reports_output.reports.pdf_generator import PDFReportGenerator
from utils.helpers import Helpers

# Check authentication
if 'authenticated' not in st.session_state or not st.session_state.authenticated:
    st.warning("Please login first")
    st.stop()

st.title("📋 Results & Reports")

user = st.session_state.user
db_manager = st.session_state.db_manager

# Pagination settings
ITEMS_PER_PAGE = 10

# Initialize pagination state
if 'results_page' not in st.session_state:
    st.session_state.results_page = 1
if 'predictions_page' not in st.session_state:
    st.session_state.predictions_page = 1

# Get user sessions with caching
@st.cache_data(ttl=300, show_spinner=False)
def get_cached_sessions(user_id):
    return db_manager.get_user_sessions(user_id)

sessions = get_cached_sessions(user['id'])

if not sessions:
    st.info("No analysis sessions found. Upload an EDF file to get started!")
    st.stop()

# Session selector
st.subheader("Select Session")

session_options = {
    f"{s['filename']} - {Helpers.format_timestamp(s['upload_time'])}": s['id'] 
    for s in sessions
}

selected_session_name = st.selectbox("Choose a session", list(session_options.keys()))
selected_session_id = session_options[selected_session_name]

# Get session details
selected_session = next(s for s in sessions if s['id'] == selected_session_id)

st.markdown("---")

# Session information
st.subheader("📊 Session Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Status", selected_session['processing_status'])

with col2:
    st.metric("Original Channels", selected_session['channels_original'])

with col3:
    st.metric("Selected Channels", selected_session['channels_selected'])

# Brain metrics
st.markdown("---")
st.subheader("🧠 Brain Metrics")

metrics = db_manager.get_session_metrics(selected_session_id)

if metrics:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Alpha Power", f"{metrics['alpha_power']:.4f} μV²")
    
    with col2:
        st.metric("Beta Power", f"{metrics['beta_power']:.4f} μV²")
    
    with col3:
        st.metric("Theta Power", f"{metrics['theta_power']:.4f} μV²")
    
    with col4:
        st.metric("Delta Power", f"{metrics['delta_power']:.4f} μV²")
    
    # Band power distribution
    st.markdown("#### Relative Band Powers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Get metrics JSON
        import json
        metrics_dict = json.loads(metrics['metrics_json']) if metrics.get('metrics_json') else {}
        
        if metrics_dict:
            st.write("**Alpha:**", f"{metrics_dict.get('alpha_relative', 0):.2f}%")
            st.write("**Beta:**", f"{metrics_dict.get('beta_relative', 0):.2f}%")
            st.write("**Theta:**", f"{metrics_dict.get('theta_relative', 0):.2f}%")
    
    with col2:
        if metrics_dict:
            st.write("**Delta:**", f"{metrics_dict.get('delta_relative', 0):.2f}%")
            st.write("**Gamma:**", f"{metrics_dict.get('gamma_relative', 0):.2f}%")
            st.write("**Brain State:**", metrics_dict.get('brain_state', 'N/A'))

else:
    st.info("No brain metrics available for this session")

# Predictions with pagination
st.markdown("---")
st.subheader("🤖 Model Predictions")

@st.cache_data(ttl=300, show_spinner=False)
def get_cached_predictions(session_id):
    return db_manager.get_session_predictions(session_id)

predictions = get_cached_predictions(selected_session_id)

if predictions:
    total_predictions = len(predictions)
    total_pages = math.ceil(total_predictions / ITEMS_PER_PAGE)
    
    # Pagination controls
    col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
    
    with col1:
        if st.button("⬅️ Previous", disabled=(st.session_state.predictions_page == 1)):
            st.session_state.predictions_page -= 1
            st.rerun()
    
    with col2:
        st.write(f"Page {st.session_state.predictions_page} of {total_pages}")
    
    with col3:
        st.write(f"Total: {total_predictions} predictions")
    
    with col4:
        if st.button("Next ➡️", disabled=(st.session_state.predictions_page >= total_pages)):
            st.session_state.predictions_page += 1
            st.rerun()
    
    # Get paginated predictions
    start_idx = (st.session_state.predictions_page - 1) * ITEMS_PER_PAGE
    end_idx = min(start_idx + ITEMS_PER_PAGE, total_predictions)
    paginated_predictions = predictions[start_idx:end_idx]
    
    # Create predictions table
    pred_data = []
    for pred in paginated_predictions:
        pred_data.append({
            'Model': pred['model_name'],
            'Prediction': pred.get('prediction_result', 'Unknown'),
            'Accuracy': f"{pred['accuracy']:.2%}" if pred['accuracy'] else 'N/A',
            'Processing Time': f"{pred['processing_time']:.3f}s" if pred['processing_time'] else 'N/A',
            'Timestamp': Helpers.format_timestamp(pred['created_at'])
        })
    
    st.table(pred_data)
    
    # Best model
    best_pred = max(predictions, key=lambda x: x['accuracy'] if x['accuracy'] else 0)
    st.success(f"🏆 Best Model: {best_pred['model_name']} with {best_pred['accuracy']:.2%} accuracy")
    
else:
    st.info("No predictions available for this session")

# PDF Report Generation
st.markdown("---")
st.subheader("📄 Generate PDF Report")

if st.button("Generate PDF Report", type="primary"):
    
    if not predictions or not metrics:
        st.warning("Both brain metrics and predictions are required for PDF generation")
    else:
        with st.spinner("Generating PDF report..."):
            try:
                # Prepare data
                session_data = {
                    'filename': selected_session['filename'],
                    'upload_time': Helpers.format_timestamp(selected_session['upload_time']),
                    'channels_original': selected_session['channels_original'],
                    'channels_selected': selected_session['channels_selected'],
                    'processing_status': selected_session['processing_status']
                }
                
                import json
                metrics_dict = json.loads(metrics['metrics_json']) if metrics.get('metrics_json') else {}
                
                prediction_list = []
                for pred in predictions:
                    prediction_list.append({
                        'model_name': pred['model_name'],
                        'model_type': pred['model_type'],
                        'prediction_result': pred.get('prediction_result', 'Unknown'),
                        'accuracy': pred['accuracy'],
                        'processing_time': pred['processing_time']
                    })
                
                # Generate PDF
                pdf_gen = PDFReportGenerator()
                
                # Ensure reports directory exists
                Helpers.ensure_directory('reports_output')
                
                output_path = f"reports_output/report_{selected_session_id}_{user['id']}.pdf"
                pdf_path = pdf_gen.create_report(session_data, metrics_dict, prediction_list, output_path)
                
                # Log activity
                db_manager.log_activity(
                    user['id'],
                    'report',
                    f"Generated PDF report for session {selected_session_id}"
                )
                
                st.success("✓ PDF report generated successfully!")
                
                # Download button
                with open(pdf_path, 'rb') as f:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=f,
                        file_name=f"QuantumBCI_Report_{selected_session['filename']}.pdf",
                        mime="application/pdf"
                    )
            
            except Exception as e:
                st.error(f"Error generating PDF: {str(e)}")
                db_manager.log_activity(user['id'], 'error', f"PDF generation failed: {str(e)}")
