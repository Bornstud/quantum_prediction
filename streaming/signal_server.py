"""
Lightweight HTTP server for serving EEG signal data.
Runs in background thread, serves JSON data from RingBuffer.
"""
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from streaming.ring_buffer import RingBuffer

_ring_buffer = None
_server_thread = None
_httpd = None

class SignalHandler(BaseHTTPRequestHandler):
    """Simple HTTP handler to serve signal data as JSON"""
    
    def do_GET(self):
        """Handle GET requests for signal data"""
        if self.path == '/signals':
            try:
                if _ring_buffer is None:
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps({'signals': [], 'channels': []}).encode())
                    return
                
                # Get latest 5-second window
                window_data, is_full = _ring_buffer.get_window(window_seconds=5.0)
                
                if window_data is not None and len(window_data) > 0:
                    # window_data shape: (n_channels, n_samples)
                    # Convert to list: [[ch1_samples], [ch2_samples], ...]
                    signals_list = window_data.tolist()
                    channel_names = [f"Ch{i+1}" for i in range(window_data.shape[0])]
                    
                    response_data = {
                        'signals': signals_list,
                        'channels': channel_names,
                        'sample_rate': 256
                    }
                else:
                    response_data = {'signals': [], 'channels': []}
                
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode())
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'error': str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress server logs"""
        pass

def set_ring_buffer(ring_buffer: RingBuffer):
    """Set the ring buffer to serve data from"""
    global _ring_buffer
    _ring_buffer = ring_buffer

def start_server(port=8001):
    """Start the signal server in a background thread"""
    global _server_thread, _httpd
    
    if _server_thread is not None and _server_thread.is_alive():
        print(f"Signal server already running on port {port}")
        return
    
    try:
        _httpd = HTTPServer(('0.0.0.0', port), SignalHandler)
        _server_thread = threading.Thread(target=_httpd.serve_forever, daemon=True)
        _server_thread.start()
        print(f"✅ Signal server started successfully on port {port}")
    except Exception as e:
        print(f"❌ Failed to start signal server: {e}")

def stop_server():
    """Stop the signal server"""
    global _httpd, _server_thread
    if _httpd is not None:
        _httpd.shutdown()
        _httpd.server_close()  # Release the socket
        _httpd = None
    if _server_thread is not None:
        _server_thread.join(timeout=2.0)
        if _server_thread.is_alive():
            print("Warning: Signal server thread did not stop cleanly")
        _server_thread = None
    print("Signal server stopped")
