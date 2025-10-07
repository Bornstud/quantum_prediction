"""
Optimized signal viewer with smooth streaming and zero flickering
"""
import streamlit.components.v1 as components
import json
import numpy as np

def simple_signal_viewer(signals_data, height=800):
    """
    Display EEG signals with smooth streaming and incremental updates.
    
    Args:
        signals_data: numpy array of shape (n_channels, n_samples)
        height: Component height in pixels
    """
    
    if signals_data is None or len(signals_data) == 0:
        html_code = """
        <div style="background: #0a0a0a; color: #888; text-align: center; padding: 50px; font-family: Arial;">
            <h3>⏳ Waiting for signal data...</h3>
            <p>Upload a file and start streaming to see live EEG signals</p>
        </div>
        """
        components.html(html_code, height=200)
        return
    
    # Convert numpy array to JSON-serializable list
    if isinstance(signals_data, np.ndarray):
        signals_list = signals_data.tolist()
    else:
        signals_list = signals_data
    
    n_channels = len(signals_list)
    n_samples = len(signals_list[0]) if n_channels > 0 else 0
    
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
        
        <script>
            const signalsData = {json.dumps(signals_list)};
            const SAMPLE_RATE = 256;
            const SPACING = 5;
            
            // Create time array
            const n_samples = signalsData[0].length;
            const timePoints = Array.from({{length: n_samples}}, (_, i) => i / SAMPLE_RATE);
            
            // Create traces for each channel
            const traces = signalsData.map((channelData, idx) => {{
                const offset = (signalsData.length - idx - 1) * SPACING;
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
            
            // Use Plotly.react for smoother updates instead of newPlot
            const chartDiv = document.getElementById('chart');
            if (chartDiv._fullLayout) {{
                Plotly.react('chart', traces, layout, config);
            }} else {{
                Plotly.newPlot('chart', traces, layout, config);
            }}
            
            // Handle window resize for responsive design
            window.addEventListener('resize', () => {{
                Plotly.Plots.resize('chart');
            }});
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height)
