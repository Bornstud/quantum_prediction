"""
WebSocket-based signal viewer for ultra-smooth real-time EEG streaming
Identical appearance to simple_signal_viewer but uses WebSocket for data updates
"""
import streamlit.components.v1 as components
import json

def websocket_signal_viewer(websocket_url: str = "ws://localhost:8765", height: int = 800):
    """
    Display EEG signals with WebSocket-based real-time streaming.
    
    Args:
        websocket_url: WebSocket server URL
        height: Component height in pixels
    """
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                margin: 0;
                padding: 0;
                background-color: #0a0a0a;
                overflow: hidden;
            }}
            #chart {{
                width: 100%;
                height: 100%;
            }}
            #status {{
                position: absolute;
                top: 10px;
                right: 10px;
                color: #888;
                font-family: Arial;
                font-size: 12px;
                background: rgba(0,0,0,0.7);
                padding: 5px 10px;
                border-radius: 4px;
            }}
            .connected {{ color: #4CAF50; }}
            .disconnected {{ color: #FF5252; }}
            /* Responsive adjustments */
            @media (max-width: 768px) {{
                #chart {{
                    font-size: 12px;
                }}
            }}
        </style>
    </head>
    <body>
        <div id="chart"></div>
        <div id="status">⏳ Connecting...</div>
        
        <script>
            const SAMPLE_RATE = 256;
            const SPACING = 5;
            const WS_URL = "{websocket_url}";
            
            let ws = null;
            let chartInitialized = false;
            let reconnectTimer = null;
            
            const statusDiv = document.getElementById('status');
            
            // Initialize empty chart
            const layout = {{
                height: {height},
                margin: {{ t: 20, b: 40, l: 80, r: 20 }},
                plot_bgcolor: '#0a0a0a',
                paper_bgcolor: '#0a0a0a',
                showlegend: false,
                xaxis: {{
                    title: 'Time (seconds)',
                    color: '#888',
                    gridcolor: '#222',
                    dtick: window.innerWidth < 768 ? 1 : 0.5
                }},
                yaxis: {{
                    title: 'Channels',
                    color: '#888',
                    gridcolor: '#222',
                    showticklabels: false
                }},
                hovermode: 'closest',
                transition: {{
                    duration: 300,
                    easing: 'cubic-in-out'
                }}
            }};
            
            const config = {{
                displayModeBar: false,
                responsive: true
            }};
            
            // Initialize empty chart
            Plotly.newPlot('chart', [], layout, config);
            
            function updateChart(signalsData) {{
                const n_channels = signalsData.length;
                const n_samples = signalsData[0].length;
                const timePoints = Array.from({{length: n_samples}}, (_, i) => i / SAMPLE_RATE);
                
                const traces = signalsData.map((channelData, idx) => {{
                    const offset = (n_channels - idx - 1) * SPACING;
                    const yData = channelData.map(v => v + offset);
                    
                    return {{
                        x: timePoints,
                        y: yData,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ 
                            color: '#00D9FF', 
                            width: 1.2,
                            shape: 'spline',
                            smoothing: 0.8
                        }},
                        name: `Ch${{idx+1}}`,
                        hovertemplate: `Ch${{idx+1}}: %{{y:.2f}}<extra></extra>`
                    }};
                }});
                
                // Use Plotly.react for ultra-smooth incremental updates
                Plotly.react('chart', traces, layout, config);
                chartInitialized = true;
            }}
            
            function connectWebSocket() {{
                ws = new WebSocket(WS_URL);
                
                ws.onopen = () => {{
                    console.log('WebSocket connected');
                    statusDiv.innerHTML = '🟢 Live Streaming';
                    statusDiv.className = 'connected';
                }};
                
                ws.onmessage = (event) => {{
                    try {{
                        const message = JSON.parse(event.data);
                        
                        if (message.type === 'signal_data') {{
                            updateChart(message.data);
                        }} else if (message.type === 'buffering') {{
                            statusDiv.innerHTML = `🔄 Buffering ${{message.current_seconds.toFixed(1)}}s / 5.0s (${{message.fill_percent.toFixed(0)}}%)`;
                        }}
                    }} catch (error) {{
                        console.error('Error processing message:', error);
                    }}
                }};
                
                ws.onerror = (error) => {{
                    console.error('WebSocket error:', error);
                    statusDiv.innerHTML = '⚠️ Connection Error';
                    statusDiv.className = 'disconnected';
                }};
                
                ws.onclose = () => {{
                    console.log('WebSocket disconnected');
                    statusDiv.innerHTML = '🔴 Disconnected - Reconnecting...';
                    statusDiv.className = 'disconnected';
                    
                    // Auto-reconnect after 2 seconds
                    reconnectTimer = setTimeout(() => {{
                        connectWebSocket();
                    }}, 2000);
                }};
            }}
            
            // Start connection
            connectWebSocket();
            
            // Handle window resize
            window.addEventListener('resize', () => {{
                Plotly.Plots.resize('chart');
            }});
            
            // Cleanup on page unload
            window.addEventListener('beforeunload', () => {{
                if (ws) ws.close();
                if (reconnectTimer) clearTimeout(reconnectTimer);
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)
