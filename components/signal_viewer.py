"""
Signal viewer component displaying all 20 EEG channels in separate scrollable strips
Professional visualization component for continuous signal monitoring
"""
import streamlit.components.v1 as components

def signal_viewer(height=800, is_streaming=True):
    """
    Display EEG signals for all 20 channels in separate horizontal strips with scroll.
    Generates smooth continuous signals for visualization purposes.
    
    Args:
        height: Component height in pixels
        is_streaming: If True, signals animate. If False, shows stopped state
    """
    
    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            * {{ box-sizing: border-box; }}
            body {{
                margin: 0;
                padding: 0;
                background-color: #0e1117;
                overflow-y: auto;
            }}
            #container {{ width: 100%; height: 100%; }}
            .channel-plot {{
                width: 100%;
                height: 100px;
                margin-bottom: 2px;
            }}
            #status {{
                position: fixed;
                top: 10px;
                right: 10px;
                color: {'#4CAF50' if is_streaming else '#FF5252'};
                font-family: Arial;
                font-size: 12px;
                background: rgba(0,0,0,0.9);
                padding: 5px 10px;
                border-radius: 4px;
                z-index: 1000;
            }}
        </style>
    </head>
    <body>
        <div id="status">{'🟢 Live Streaming' if is_streaming else '⏸️ Stopped'}</div>
        <div id="container"></div>
        
        <script>
            const SAMPLE_RATE = 256;
            const N_CHANNELS = 20;
            const WINDOW_DURATION = 6.0;
            const N_SAMPLES = Math.floor(WINDOW_DURATION * SAMPLE_RATE);
            const UPDATE_INTERVAL = 100;
            const IS_STREAMING = {'true' if is_streaming else 'false'};
            
            const CHANNEL_NAMES = [
                'Fc5', 'Fc3', 'Fc1', 'Fcz', 'Fc2', 'Fc4', 'Fc6',
                'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
                'Cp5', 'Cp3', 'Cp1', 'Cpz', 'Cp2', 'Cp4'
            ];
            
            let timeOffset = 0;
            let updateTimer = null;
            const channelSeeds = Array.from({{length: N_CHANNELS}}, () => Math.random() * 10);
            
            function generateChannelSignal(channelIndex, nSamples, timeStart) {{
                const seed = channelSeeds[channelIndex];
                const signal = [];
                
                for (let i = 0; i < nSamples; i++) {{
                    const t = timeStart + (i / SAMPLE_RATE);
                    const delta = Math.sin(2 * Math.PI * (seed * 0.5 + 2) * t) * 15;
                    const theta = Math.sin(2 * Math.PI * (seed * 0.8 + 6) * t) * 12;
                    const alpha = Math.sin(2 * Math.PI * (seed * 1.2 + 10) * t) * 25;
                    const beta = Math.sin(2 * Math.PI * (seed * 2 + 20) * t) * 8;
                    const gamma = Math.sin(2 * Math.PI * (seed * 3 + 35) * t) * 5;
                    const noise = (Math.random() - 0.5) * 10;
                    signal.push(delta + theta + alpha + beta + gamma + noise);
                }}
                return signal;
            }}
            
            function initializeChannels() {{
                const container = document.getElementById('container');
                for (let i = 0; i < N_CHANNELS; i++) {{
                    const div = document.createElement('div');
                    div.id = `channel-${{i}}`;
                    div.className = 'channel-plot';
                    container.appendChild(div);
                }}
            }}
            
            function updateChart() {{
                const timePoints = Array.from({{length: N_SAMPLES}}, (_, i) => i / SAMPLE_RATE);
                
                for (let ch = 0; ch < N_CHANNELS; ch++) {{
                    const channelData = generateChannelSignal(ch, N_SAMPLES, timeOffset);
                    
                    const trace = {{
                        x: timePoints,
                        y: channelData,
                        type: 'scatter',
                        mode: 'lines',
                        line: {{ color: '#1E88E5', width: 1.5 }},
                        name: CHANNEL_NAMES[ch],
                        hovertemplate: `${{CHANNEL_NAMES[ch]}}: %{{y:.1f}} μV<extra></extra>`
                    }};
                    
                    const layout = {{
                        height: 100,
                        margin: {{ t: 5, b: (ch === N_CHANNELS - 1 ? 35 : 20), l: 60, r: 10 }},
                        plot_bgcolor: '#0e1117',
                        paper_bgcolor: '#0e1117',
                        showlegend: false,
                        xaxis: {{
                            title: ch === N_CHANNELS - 1 ? 'Time (seconds)' : '',
                            color: '#888',
                            gridcolor: '#2a2a2a',
                            dtick: 1.0,
                            showgrid: true,
                            range: [0, WINDOW_DURATION]
                        }},
                        yaxis: {{
                            title: {{ text: CHANNEL_NAMES[ch], font: {{ size: 11, color: '#888' }} }},
                            color: '#888',
                            gridcolor: '#2a2a2a',
                            showgrid: true,
                            zeroline: true,
                            zerolinecolor: '#444',
                            range: [-80, 80]
                        }}
                    }};
                    
                    const config = {{ displayModeBar: false, responsive: true }};
                    Plotly.react(`channel-${{ch}}`, [trace], layout, config);
                }}
                
                if (IS_STREAMING) {{ timeOffset += 0.1; }}
            }}
            
            initializeChannels();
            updateChart();
            if (IS_STREAMING) {{ updateTimer = setInterval(updateChart, UPDATE_INTERVAL); }}
        </script>
    </body>
    </html>
    """
    
    components.html(html_code, height=height, scrolling=True)
