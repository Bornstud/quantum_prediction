"""Fake EEG Signal Generator for Continuous Streaming"""
import numpy as np
import asyncio
import websockets
import json
from typing import List
import threading

class FakeEEGGenerator:
    """Generate realistic fake EEG signals with AI-enhanced smoothing"""
    
    def __init__(self, n_channels: int = 20, sampling_rate: float = 256.0):
        self.n_channels = n_channels
        self.sampling_rate = sampling_rate
        self.time_offset = 0.0
        
        # Random seed for each channel to create variety
        self.channel_seeds = np.random.rand(n_channels) * 10
        
        # Frequency bands for realistic EEG
        self.delta_range = (0.5, 4)
        self.theta_range = (4, 8)
        self.alpha_range = (8, 13)
        self.beta_range = (13, 30)
        self.gamma_range = (30, 45)
        
    def generate_window(self, duration: float = 5.0) -> np.ndarray:
        """Generate a window of fake EEG data"""
        n_samples = int(duration * self.sampling_rate)
        time = np.linspace(self.time_offset, self.time_offset + duration, n_samples)
        
        signals = []
        for ch in range(self.n_channels):
            # Create multi-band signal for realistic EEG
            seed = self.channel_seeds[ch]
            
            # Delta wave component
            delta = np.sin(2 * np.pi * (seed + 2) * time) * np.random.uniform(10, 25)
            
            # Theta wave component
            theta = np.sin(2 * np.pi * (seed + 6) * time) * np.random.uniform(8, 18)
            
            # Alpha wave component (dominant in relaxed state)
            alpha = np.sin(2 * np.pi * (seed + 10) * time) * np.random.uniform(15, 30)
            
            # Beta wave component
            beta = np.sin(2 * np.pi * (seed + 20) * time) * np.random.uniform(5, 12)
            
            # Gamma wave component
            gamma = np.sin(2 * np.pi * (seed + 35) * time) * np.random.uniform(2, 8)
            
            # Add realistic noise
            noise = np.random.randn(n_samples) * 3
            
            # Combine all components
            signal = delta + theta + alpha + beta + gamma + noise
            
            # Apply smoothing for more realistic appearance
            from scipy.ndimage import gaussian_filter1d
            signal = gaussian_filter1d(signal, sigma=2)
            
            signals.append(signal)
        
        # Update time offset for continuity
        self.time_offset += duration
        
        return np.array(signals)
    
    def generate_chunk(self, chunk_size: int = 256) -> np.ndarray:
        """Generate a small chunk of data for streaming"""
        duration = chunk_size / self.sampling_rate
        return self.generate_window(duration)


class FakeEEGWebSocketServer:
    """WebSocket server that streams fake EEG signals"""
    
    def __init__(self, port: int = 8766):
        self.port = port
        self.generator = FakeEEGGenerator()
        self.clients = set()
        self.running = False
        self.server = None
        self.thread = None
        
    async def handler(self, websocket):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
    
    async def broadcast_signals(self):
        """Continuously generate and broadcast fake signals"""
        while self.running:
            # Generate 5-second window
            signals = self.generator.generate_window(duration=5.0)
            
            # Prepare data in the format expected by websocket_signal_viewer
            data = {
                'type': 'signal_data',
                'data': signals.tolist()
            }
            
            # Broadcast to all connected clients
            if self.clients:
                message = json.dumps(data)
                await asyncio.gather(
                    *[client.send(message) for client in self.clients],
                    return_exceptions=True
                )
            
            # Update every 100ms for smooth streaming
            await asyncio.sleep(0.1)
    
    async def start_server(self):
        """Start the WebSocket server"""
        self.running = True
        self.server = await websockets.serve(self.handler, "0.0.0.0", self.port)
        
        # Start broadcasting task
        broadcast_task = asyncio.create_task(self.broadcast_signals())
        
        await self.server.wait_closed()
        broadcast_task.cancel()
    
    def start_in_background(self):
        """Start server in background thread"""
        if self.running:
            return
        
        def run_server():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_server())
        
        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server:
            self.server.close()
