def generate_botnet_viz_html(G, bot_scores, communities, botnet):
    """Generate interactive botnet visualization"""
    import plotly.graph_objects as go
    import networkx as nx
    import numpy as np
    
    # Layout
    if G.number_of_nodes() > 0:
        pos = nx.spring_layout(G, k=2/np.sqrt(max(G.number_of_nodes(), 1)), iterations=50)
    else:
        pos = {}
    
    # Edge trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos.get(edge[0], (0, 0))
        x1, y1 = pos.get(edge[1], (0, 0))
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines',
                            line=dict(width=0.5, color='#888'), hoverinfo='none')
    
    # Node trace
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    
    roles = botnet.get('role_assignments', {}).get('assignments', {})
    
    for node in G.nodes():
        x, y = pos.get(node, (0, 0))
        node_x.append(x)
        node_y.append(y)
        
        score = bot_scores.get(node, 0.5)
        node_color.append(score)
        
        role = roles.get(node, {}).get('role', 'unknown')
        size = {'hub': 30, 'coordinator': 20, 'active_member': 15, 'peripheral': 10}.get(role, 10)
        node_size.append(size)
        
        node_text.append(f"ID: {node[:20]}...<br>Bot Score: {score:.0%}<br>Role: {role}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(showscale=True, colorscale='RdYlGn_r', color=node_color, size=node_size,
                    colorbar=dict(thickness=15, title='Bot Probability'),
                    line=dict(width=1, color='white'))
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title=f"Botnet: {botnet.get('auto_name', botnet['botnet_id'])} ({botnet['size']} accounts)",
                        showlegend=False, hovermode='closest',
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white'
                    ))
    
    metrics = botnet.get('metrics', {})
    targets = botnet.get('targets', {})
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Botnet: {botnet.get('auto_name', botnet['botnet_id'])}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1400px; margin: 0 auto; }}
            .header {{ background: linear-gradient(135deg, #d32f2f 0%, #b71c1c 100%); color: white; padding: 20px; border-radius: 12px 12px 0 0; }}
            .content {{ background: white; padding: 20px; border-radius: 0 0 12px 12px; }}
            .stats {{ display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0; }}
            .stat {{ background: #f5f5f5; padding: 15px 20px; border-radius: 8px; min-width: 150px; }}
            .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
            .stat-label {{ font-size: 12px; color: #666; text-transform: uppercase; }}
            #graph {{ height: 600px; }}
            .targets {{ margin-top: 20px; }}
            .target-list {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .target {{ background: #fff3f3; border: 1px solid #ffcccc; padding: 8px 12px; border-radius: 6px; font-size: 13px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 {botnet.get('auto_name', 'Unknown Botnet')}</h1>
                <p>Confidence: {botnet.get('confidence', 0):.0%} | Discovered: {botnet.get('discovered_at', 'Unknown')}</p>
            </div>
            <div class="content">
                <div class="stats">
                    <div class="stat"><div class="stat-value">{botnet['size']}</div><div class="stat-label">Accounts</div></div>
                    <div class="stat"><div class="stat-value">{metrics.get('total_comments', 0)}</div><div class="stat-label">Comments</div></div>
                    <div class="stat"><div class="stat-value">{metrics.get('unique_videos_targeted', 0)}</div><div class="stat-label">Videos Targeted</div></div>
                    <div class="stat"><div class="stat-value">{metrics.get('unique_channels_targeted', 0)}</div><div class="stat-label">Channels Targeted</div></div>
                </div>
                
                <div id="graph"></div>
                
                <div class="targets">
                    <h3>Top Targets</h3>
                    <div class="target-list">
                        {"".join(f'<div class="target">{t.get("channel", "Unknown")} ({t.get("comments", 0)} comments)</div>' for t in targets.get('top_channels', [])[:10])}
                    </div>
                </div>
            </div>
        </div>
        <script>
            var data = {fig.to_json()};
            Plotly.newPlot('graph', data.data, data.layout, {{responsive: true}});
        </script>
    </body>
    </html>
    """


def generate_dashboard_html(botnets, stats):
    """Generate main discovery dashboard"""
    
    botnet_cards = ""
    for b in botnets:
        botnet_cards += f"""
        <div class="botnet-card">
            <div class="botnet-header">
                <h3>{b.get('auto_name', b['botnet_id'])}</h3>
                <span class="confidence">{b.get('confidence', 0):.0%}</span>
            </div>
            <div class="botnet-stats">
                <span>👥 {b['size']} accounts</span>
                <span>📅 {b.get('discovered_at', 'Unknown')[:10]}</span>
            </div>
            <a href="/api/botnets/{b['botnet_id']}/visualize" class="btn">View Network →</a>
        </div>
        """
    
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Botnet Discovery Dashboard</title>
        <style>
            * {{ box-sizing: border-box; }}
            body {{ font-family: -apple-system, sans-serif; margin: 0; padding: 20px; background: #1a1a2e; color: white; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .header {{ text-align: center; padding: 40px 20px; }}
            .header h1 {{ font-size: 36px; margin: 0; }}
            .header p {{ color: #888; }}
            .stats-row {{ display: flex; gap: 20px; justify-content: center; margin: 30px 0; }}
            .stat-box {{ background: #16213e; padding: 20px 30px; border-radius: 12px; text-align: center; }}
            .stat-box .value {{ font-size: 32px; font-weight: bold; color: #e94560; }}
            .stat-box .label {{ font-size: 12px; color: #888; text-transform: uppercase; margin-top: 5px; }}
            .section-title {{ font-size: 24px; margin: 40px 0 20px; }}
            .botnet-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 20px; }}
            .botnet-card {{ background: #16213e; border-radius: 12px; padding: 20px; border: 1px solid #0f3460; }}
            .botnet-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
            .botnet-header h3 {{ margin: 0; font-size: 16px; }}
            .confidence {{ background: #e94560; padding: 4px 10px; border-radius: 12px; font-size: 12px; }}
            .botnet-stats {{ color: #888; font-size: 13px; margin-bottom: 15px; display: flex; gap: 15px; }}
            .btn {{ display: inline-block; background: #0f3460; color: white; padding: 8px 16px; border-radius: 6px; text-decoration: none; font-size: 13px; }}
            .btn:hover {{ background: #e94560; }}
            .run-discovery {{ background: #e94560; color: white; border: none; padding: 15px 30px; border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 20px; }}
            .run-discovery:hover {{ background: #ff6b6b; }}
            .empty {{ text-align: center; padding: 60px; color: #888; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🕵️ Botnet Discovery Dashboard</h1>
                <p>Discover and map coordinated bot networks on YouTube</p>
            </div>
            
            <div class="stats-row">
                <div class="stat-box">
                    <div class="value">{len(botnets)}</div>
                    <div class="label">Botnets Discovered</div>
                </div>
                <div class="stat-box">
                    <div class="value">{sum(b.get('size', 0) for b in botnets)}</div>
                    <div class="label">Bot Accounts</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats.get('total_votes', 0)}</div>
                    <div class="label">Community Votes</div>
                </div>
                <div class="stat-box">
                    <div class="value">{stats.get('tracked_authors', 0)}</div>
                    <div class="label">Authors Tracked</div>
                </div>
            </div>
            
            <h2 class="section-title">Discovered Botnets</h2>
            
            {"<div class='botnet-grid'>" + botnet_cards + "</div>" if botnets else "<div class='empty'><p>No botnets discovered yet.</p><p>Visit YouTube videos to collect data, then run discovery.</p></div>"}
            
            <div style="text-align: center; margin-top: 40px;">
                <button class="run-discovery" onclick="runDiscovery()">🔍 Run Discovery</button>
            </div>
        </div>
        
        <script>
            async function runDiscovery() {{
                const btn = document.querySelector('.run-discovery');
                btn.textContent = '⏳ Running...';
                btn.disabled = true;
                
                try {{
                    const resp = await fetch('/api/discover', {{
                        method: 'POST',
                        headers: {{'Content-Type': 'application/json'}},
                        body: JSON.stringify({{min_bot_probability: 0.5, min_botnet_size: 3}})
                    }});
                    const data = await resp.json();
                    alert('Discovery complete! Found ' + (data.phases?.identification?.botnets_found || 0) + ' botnets');
                    location.reload();
                }} catch(e) {{
                    alert('Error: ' + e.message);
                }} finally {{
                    btn.textContent = '🔍 Run Discovery';
                    btn.disabled = false;
                }}
            }}
        </script>
    </body>
    </html>
    """