"""
Flicker-free EEG Signal Viewer using Async Data Fetch
Custom Streamlit component with internal chart updates
"""
import streamlit.components.v1 as components

def async_signal_viewer(height=600, endpoint="http://localhost:8001/signals"):
    """
    Display EEG signals with flicker-free async updates.
    Component fetches data internally without Streamlit re-renders.
    
    Args:
        height: Component height in pixels
        endpoint: REST endpoint to fetch signal data from
    """
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #0a0a0a;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }}
            #chart {{
                width: 100%;
                height: 100%;
            }}
            #status {{
                position: absolute;
                top: 10px;
                right: 10px;
                color: #00D9FF;
                font-size: 12px;
                background: rgba(0,0,0,0.5);
                padding: 5px 10px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <div id="status">Connecting...</div>
        <div id="chart"></div>
        
        <script>
            // Construct endpoint URL for Replit environment
            // Try to use same origin with port 8001
            const isReplit = window.location.hostname.includes('repl');
            let ENDPOINT;
            if (isReplit) {{
                // In Replit, use hostname with -8001 pattern or same origin
                const hostname = window.location.hostname;
                const protocol = window.location.protocol;
                // Try replacing port in URL
                ENDPOINT = protocol + '//' + hostname.replace(/\\-5000/, '-8001') + '/signals';
                console.log('Replit detected, endpoint:', ENDPOINT);
            }} else {{
                // Local development
                ENDPOINT = 'http://localhost:8001/signals';
            }}
            const FETCH_INTERVAL = 200; // ms (5Hz)
            const CHANNEL_COUNT = 20;
            const WINDOW_DURATION = 5.0; // seconds
            const SAMPLE_RATE = 256;
            
            let isRunning = true;
            let chartInitialized = false;
            
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
                    range: [0, WINDOW_DURATION]
                }},
                yaxis: {{
                    title: 'Channels',
                    color: '#888',
                    gridcolor: '#222',
                    showticklabels: false
                }},
                hovermode: 'closest'
            }};
            
            // Create initial traces (one per channel)
            const traces = [];
            for (let i = 0; i < CHANNEL_COUNT; i++) {{
                traces.push({{
                    y: [],
                    type: 'scatter',
                    mode: 'lines',
                    line: {{ color: '#00D9FF', width: 1 }},
                    name: `Ch${{i+1}}`,
                    hovertemplate: `Ch${{i+1}}: %{{y:.2f}}<extra></extra>`
                }});
            }}
            
            Plotly.newPlot('chart', traces, layout, {{displayModeBar: false}});
            chartInitialized = true;
            
            // Fetch and update data
            async function fetchAndUpdate() {{
                if (!isRunning) return;
                
                try {{
                    const response = await fetch(ENDPOINT);
                    const data = await response.json();
                    
                    if (data.signals && data.signals.length > 0) {{
                        document.getElementById('status').textContent = 'Streaming ✓';
                        document.getElementById('status').style.color = '#00FF00';
                        
                        // Prepare updates for all channels
                        const updates = [];
                        const n_samples = data.signals[0].length;
                        const timePoints = Array.from({{length: n_samples}}, (_, i) => i / SAMPLE_RATE);
                        
                        // Stack channels vertically with spacing
                        const spacing = 5;
                        for (let ch = 0; ch < CHANNEL_COUNT; ch++) {{
                            const offset = (CHANNEL_COUNT - ch - 1) * spacing;
                            const yData = data.signals[ch].map(v => v + offset);
                            updates.push({{
                                x: [timePoints],
                                y: [yData]
                            }});
                        }}
                        
                        // Update all traces at once (no flicker!)
                        Plotly.update('chart', updates, {{}}, Array.from({{length: CHANNEL_COUNT}}, (_, i) => i));
                    }} else {{
                        document.getElementById('status').textContent = 'No Data';
                        document.getElementById('status').style.color = '#FF6B00';
                    }}
                }} catch (error) {{
                    console.error('Fetch error:', error);
                    document.getElementById('status').textContent = 'Error';
                    document.getElementById('status').style.color = '#FF0000';
                }}
            }}
            
            // Start polling
            setInterval(fetchAndUpdate, FETCH_INTERVAL);
            
            // Initial fetch
            fetchAndUpdate();
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)
