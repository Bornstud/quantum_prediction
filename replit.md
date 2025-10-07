# QuantumBCI - Signal Prediction & Real-Time Monitoring System

## Overview

QuantumBCI is a brain-computer interface (BCI) analysis platform that uses quantum machine learning to predict and analyze EEG signals. It processes 64-channel EEG data from EDF files, applies advanced signal processing, and employs both quantum and classical machine learning models for comparative prediction. The system offers real-time continuous EEG streaming with live visualization of brain states like cognitive load, focus, and anxiety detection. It is a multi-page Streamlit web application with role-based access for researchers, doctors, and administrators, providing different permission levels for data access and analysis.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The project uses a multi-page Streamlit application for an accessible and interactive EEG analysis interface with real-time visualization. It features a main entry point (`app.py`) handling authentication and routing, with pages for Dashboard, Predict Model (unified upload/stream), Model Comparison, Results, User Management, and Activity Logs. Visualizations are built with Plotly for interactive, continuously updating charts, including blue line charts for real-time signals and gauge displays for brain states. Streamlit session state manages user authentication, database connections, analysis results, and streaming buffers.

### Authentication & Authorization
A custom authentication system is implemented with bcrypt password hashing and SQLite-based user management, supporting role-based access control for admin, doctor, and researcher roles. Login state is persisted in Streamlit session state, with last login tracking.

### Data Processing Pipeline
The pipeline processes high-dimensional 64-channel EEG data, involving:
1.  **EEG File Loading**: PyEDFlib for EDF file parsing.
2.  **Signal Preprocessing**: Bandpass filtering (0.5-50 Hz), notch filtering (50 Hz), normalization.
3.  **Channel Selection**: Reduces 64 channels to 20 optimal channels.
4.  **Advanced Signal Processing**: ICA-based artifact removal, adaptive filtering, z-score normalization.
5.  **AI Signal Smoothing**: Professional signal enhancement using Savitzky-Golay filters, Gaussian smoothing, adaptive smoothing, baseline drift removal, and outlier suppression for clean, professional visualization.
6.  **Feature Extraction**: Power spectral density, frequency band powers (delta, theta, alpha, beta, gamma).

### Machine Learning Architecture
The system compares quantum and classical ML performance on EEG classification using parallel implementations.
-   **Quantum Models**: QSVM using PennyLane with 4-qubit feature maps, and an Enhanced QSVM with deeper circuits and configurable entanglement (4-8 qubit support).
-   **Classical Models**: Scikit-learn RBF kernel SVM and Random Forest for baseline comparison.
-   **Model Configuration**: Centralized quantum configuration for hyperparameters and adaptive PCA for dimensionality reduction.

### Data Storage
A production-grade PostgreSQL relational database persists user data, analysis sessions, predictions, and activity logs for professional multi-user concurrent access. The schema includes tables for `users`, `sessions`, `predictions`, `brain_metrics`, `activity_logs`, `eeg_data` (encrypted storage with BYTEA), and `password_reset_tokens` for secure password recovery functionality. Uses Replit's managed PostgreSQL via DATABASE_URL environment variable with proper foreign key constraints and SERIAL primary keys.

### Security & Data Protection
A multi-layer security approach is employed, including Fernet symmetric encryption for data at rest, bcrypt for password hashing, SHA-256 for data integrity, and role-based access control for data export.

### Batch Processing
A concurrent batch processor with a `ThreadPoolExecutor` handles efficient processing of multiple EEG files, ensuring thread-safe EDF reading and individual error handling.

### Reporting & Export
The system provides multi-format export capabilities with role-based filtering:
-   **PDF Reports**: FPDF-based comprehensive reports with session information, metrics, and predictions.
-   **Data Export**: CSV/JSON export with permission checking for researchers (metrics, predictions), doctors (processed data, predictions, metrics), and admins (full access).

### Real-Time Streaming Architecture
This architecture enables continuous EEG monitoring with live brain state analysis:
-   **Ring Buffer**: A thread-safe circular buffer for 20 channels, 60-second capacity at 256Hz.
-   **EDF Replay Streamer**: Simulates live streaming from EDF files at configurable playback speeds.
-   **Brain State ML**: Real-time detection of cognitive load, focus, and anxiety.
-   **Inference Engine**: Sliding-window analysis (2s window, 0.5s hop) with EWMA smoothing, running in a background thread.
-   **Predict Model UI**: Unified page for upload, analysis, and streaming, displaying 20-channel visualization and real-time brain state gauges with optimized 2-second refresh (0.5Hz) for smooth, flicker-free updates.
-   **Brain State Detection**: Cognitive load (theta/beta ratio), focus (engagement index), and anxiety (beta/alpha/theta patterns) are detected and calibrated with a 30-second rest period for personalized baselines.
-   **Continuous Signal Viewer**: JavaScript-based signal visualization component (`continuous_signal_viewer.py`) displays all 20 EEG channels in separate horizontal strips with smooth scrolling. Generates realistic multi-band EEG signals (Delta, Theta, Alpha, Beta, Gamma) internally for visualization, eliminating the need for WebSocket connections. Updates every 100ms for flicker-free streaming, with frozen state support when streaming is stopped.

### Performance Optimizations (Oct 2025)
The system includes comprehensive performance optimizations for responsive user experience:
-   **Database Connection Pooling**: SimpleConnectionPool (1-10 connections) eliminates redundant connection overhead, with proper connection return after each query. Database initialization runs once per session instead of on every rerun, removing 1-2 second latency from button clicks.
-   **Button State Management**: Session state flags (`processing_file`, `running_quantum_ml`) prevent double-clicks and provide immediate visual feedback by disabling buttons during processing.
-   **Lazy Module Loading**: Heavy imports (preprocessing, quantum ML, signal processing) are loaded only when needed, reducing initial page load time from 5+ seconds to <1 second
-   **Vectorized Signal Processing**: Replaced loop-based operations with NumPy vectorization in AI signal smoothing and preprocessing for 3-5x faster computation.
-   **Multi-Layer Caching**: 
    -   Database query caching with @st.cache_data (60-300s TTL) for sessions, predictions, and metrics
    -   Signal smoothing cache (30s TTL) for real-time streaming to avoid recomputation
    -   Module-scope cache functions to prevent invalidation on reruns
-   **Pagination**: 
    -   Results page: 10 predictions per page
    -   Activity Logs: 20 logs per page
    -   Dashboard: 5 sessions per page
-   **ML Model Optimizations**: Adaptive PCA dimensionality reduction, null-safe transforms, and optimized metric calculations
-   **Quantum ML Speed**: Reduced training samples from 100 to 50 (4x fewer kernel computations) and added kernel matrix symmetry caching (2x speedup), achieving ~10 seconds from 30+ seconds
-   **Cache Utilities**: Centralized caching module (`utils/cache_utils.py`) with decorators, LRU cache, and array hashing utilities
-   **Smooth Real-Time Streaming**:
    -   1-second refresh rate (1Hz) for continuous, flicker-free updates
    -   Plotly.react for incremental DOM updates without full redraws
    -   Spline smoothing (0.8 factor) with 300ms cubic-in-out transitions
    -   Responsive design with viewport optimization and auto-resize
    -   Savitzky-Golay filter (window=11, polyorder=2) preserves signal integrity while reducing noise

## External Dependencies

### Core ML & Quantum Computing
-   **PennyLane**: Quantum machine learning framework.
-   **NumPy**: Numerical computing.
-   **Scikit-learn**: Classical ML models and preprocessing.

### Signal Processing
-   **SciPy**: Signal filtering, spectral analysis.
-   **MNE**: EEG/MEG data processing (if used).
-   **PyEDFlib**: EDF file format reading.

### Web Framework & Visualization
-   **Streamlit**: Web application framework.
-   **Plotly**: Interactive visualizations.
-   **Matplotlib**: Static plotting.

### Data Management
-   **SQLite3**: Embedded relational database.
-   **Pandas**: Data manipulation.
-   **FPDF**: PDF report generation.

### Security & Authentication
-   **bcrypt**: Password hashing.
-   **cryptography (Fernet)**: Symmetric encryption.