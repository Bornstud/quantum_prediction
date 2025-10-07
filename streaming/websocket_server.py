"""
WebSocket server for ultra-smooth real-time EEG signal streaming
Streams data from ring buffer to connected clients at high frequency
"""
import asyncio
import websockets
import json
import pickle
import threading
from typing import Optional, Set
import logging

logger = logging.getLogger(__name__)

class EEGWebSocketServer:
    """WebSocket server for streaming EEG signals from ring buffer"""
    
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set = set()
        self.running = False
        self.server_task = None
        self.ring_buffer = None
        self.ai_smoother = None
        self._server_thread = None
        
    def set_ring_buffer(self, ring_buffer, ai_smoother):
        """Set the ring buffer and smoother to stream from"""
        self.ring_buffer = ring_buffer
        self.ai_smoother = ai_smoother
        
    async def register_client(self, websocket):
        """Register a new client connection"""
        self.clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister_client(self, websocket):
        """Unregister a client connection"""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def send_signal_data(self, websocket):
        """Send signal data to a specific client"""
        try:
            while True:
                if self.ring_buffer and self.ai_smoother:
                    # Get latest 5-second window
                    window_data, is_full = self.ring_buffer.get_window(window_seconds=5.0)
                    
                    if is_full:
                        # Apply AI smoothing
                        smoothed_data = self.ai_smoother.real_time_smooth(window_data)
                        
                        # Convert to JSON-serializable format
                        message = {
                            'type': 'signal_data',
                            'data': smoothed_data.tolist(),
                            'timestamp': asyncio.get_event_loop().time()
                        }
                        
                        # Send to client
                        await websocket.send(json.dumps(message))
                    else:
                        # Send buffering status
                        buffer_stats = self.ring_buffer.get_stats()
                        current_seconds = buffer_stats['current_samples'] / buffer_stats['sampling_rate']
                        fill_percent = buffer_stats['fill_percentage']
                        
                        message = {
                            'type': 'buffering',
                            'current_seconds': current_seconds,
                            'fill_percent': fill_percent
                        }
                        await websocket.send(json.dumps(message))
                
                # Update at 30Hz for ultra-smooth streaming
                await asyncio.sleep(1/30)
                
        except websockets.exceptions.ConnectionClosed:
            pass
            
    async def handle_client(self, websocket):
        """Handle individual client connection"""
        await self.register_client(websocket)
        try:
            await self.send_signal_data(websocket)
        finally:
            await self.unregister_client(websocket)
            
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        async with websockets.serve(self.handle_client, "0.0.0.0", self.port):
            logger.info(f"WebSocket server started on port {self.port}")
            await asyncio.Future()  # Run forever
            
    def start_in_background(self):
        """Start server in background thread (idempotent - won't double-bind)"""
        if self._server_thread and self._server_thread.is_alive():
            logger.info("WebSocket server already running, skipping restart")
            return
            
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server())
            
        self._server_thread = threading.Thread(target=run_server, daemon=True)
        self._server_thread.start()
        logger.info("WebSocket server started in background thread")
        
    def stop(self):
        """Stop the WebSocket server"""
        self.running = False
        if self._server_thread:
            logger.info("WebSocket server stopped")
