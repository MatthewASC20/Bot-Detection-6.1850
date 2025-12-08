"""
Network visualization for bot cluster detection - Paper-ready outputs
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple
import logging
import os

from config.config import Config

logger = logging.getLogger(__name__)

# Paper-ready color palette
CLUSTER_COLORS = [
    '#E63946',  # Red
    '#457B9D',  # Blue
    '#2A9D8F',  # Teal
    '#E9C46A',  # Yellow
    '#F4A261',  # Orange
    '#9B5DE5',  # Purple
    '#00BBF9',  # Cyan
    '#00F5D4',  # Mint
    '#FEE440',  # Bright Yellow
    '#F15BB5',  # Pink
]

NOISE_COLOR = '#CCCCCC'  # Gray for unclustered accounts


class NetworkVisualizer:
    """Create network visualizations of bot clusters - paper-ready outputs"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.GRAPHS_DIR
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set matplotlib defaults for paper-quality figures
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'legend.fontsize': 11,
            'xtick.labelsize': 11,
            'ytick.labelsize': 11,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })
    
    def visualize_bot_network(self, G: nx.Graph, 
                             bot_scores: Dict[str, float],
                             communities: Optional[Dict[str, int]] = None,
                             comments_df: Optional[pd.DataFrame] = None,
                             title: str = "Bot Coordination Network",
                             filename: str = "bot_network.html") -> str:
        """
        Create interactive network visualization with Plotly.
        Nodes are colored by bot probability (red=high, green=low).
        Cluster information shown in hover text and legend.
        """
        # Limit graph size for readability
        if G.number_of_nodes() > Config.MAX_GRAPH_NODES:
            top_nodes = sorted(bot_scores.items(), key=lambda x: x[1], reverse=True)[:Config.MAX_GRAPH_NODES]
            top_node_ids = [node for node, _ in top_nodes]
            G = G.subgraph(top_node_ids)
            logger.info(f"Limited graph to {len(top_node_ids)} nodes for visualization")
        
        # Calculate layout
        pos = self._calculate_layout(G)
        
        # Prepare edge traces
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 1)
            
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(
                    width=min(weight * Config.EDGE_WIDTH_MULTIPLIER, 5),
                    color='rgba(100, 100, 100, 0.3)'
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Build author metadata for click-to-inspect panel
        author_meta: Dict[str, Dict] = {}
        if comments_df is not None and not comments_df.empty:
            base_cols = ['author_id', 'author', 'comment_id', 'video_id', 'published_at', 'text']
            available_cols = [c for c in base_cols if c in comments_df.columns]
            comments_subset = comments_df[available_cols].copy()
            for author_id, group in comments_subset.groupby('author_id'):
                display_name = group['author'].dropna().iloc[0] if 'author' in group.columns and not group['author'].dropna().empty else author_id
                sorted_group = group.sort_values('published_at')
                comments = []
                for _, row in sorted_group.tail(20).iterrows():
                    comments.append({
                        'comment_id': row.get('comment_id'),
                        'video_id': row.get('video_id'),
                        'published_at': str(row.get('published_at')),
                        'text': (row.get('text') or '')[:500]
                    })
                author_meta[author_id] = {
                    'author_name': display_name,
                    'total_comments': len(group),
                    'videos_commented': group['video_id'].nunique() if 'video_id' in group.columns else 0,
                    'comments': comments
                }

        # Prepare node data - always color by bot probability
        node_x = []
        node_y = []
        node_color = []
        node_size = []
        node_text = []
        node_symbols = []
        node_customdata = []
        
        # Define symbols for clusters (circle for noise/unclustered)
        available_symbols = ['square', 'diamond', 'cross', 'x', 'triangle-up', 
                            'triangle-down', 'pentagon', 'hexagon', 'star']
        
        # Find which clusters are actually present in the current graph
        clusters_in_graph = set()
        if communities:
            for node in G.nodes():
                cluster_id = communities.get(node, -1)
                if cluster_id != -1:
                    clusters_in_graph.add(cluster_id)
        
        # Map only populated clusters to symbols
        populated_clusters = sorted(clusters_in_graph)
        cluster_to_symbol = {-1: 'circle'}  # Unclustered always circle
        for i, cluster_id in enumerate(populated_clusters):
            cluster_to_symbol[cluster_id] = available_symbols[i % len(available_symbols)]
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            bot_score = bot_scores.get(node, 0)
            degree = G.degree(node)
            cluster_id = communities.get(node, -1) if communities else -1
            meta = author_meta.get(node, {})
            
            # Color by bot probability
            node_color.append(bot_score)
            
            # Size by degree (larger nodes = more connections)
            node_size.append(max(10, min(30, 10 + degree * 3)))
            
            # Symbol by cluster (only for populated clusters)
            node_symbols.append(cluster_to_symbol.get(cluster_id, 'circle'))
            
            # Hover text with full details
            text = f"<b>Account:</b> {node[:30]}{'...' if len(node) > 30 else ''}<br>"
            text += f"<b>Bot Probability:</b> {bot_score:.1%}<br>"
            text += f"<b>Connections:</b> {degree}<br>"
            if communities:
                cluster_label = 'Unclustered' if cluster_id == -1 else f'Botnet Cluster {cluster_id}'
                text += f"<b>Cluster:</b> {cluster_label}"
            node_text.append(text)
            
            node_customdata.append({
                'author_id': node,
                'author_name': meta.get('author_name', node),
                'cluster_id': cluster_id,
                'bot_probability': bot_score,
                'total_comments': meta.get('total_comments', 0),
                'videos_commented': meta.get('videos_commented', 0),
                'channel_url': f"https://www.youtube.com/channel/{node}",
                'comments': meta.get('comments', [])
            })
        
        # Create main node trace with bot probability coloring
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers',
            hoverinfo='text',
            hovertext=node_text,
            name='Accounts',
            marker=dict(
                showscale=True,
                colorscale='RdYlGn_r',  # Red (high prob) to Green (low prob)
                color=node_color,
                size=node_size,
                symbol=node_symbols,
                cmin=0,
                cmax=1,
                colorbar=dict(
                    title=dict(text='Bot<br>Probability', font=dict(size=14)),
                    thickness=20,
                    len=0.6,
                    tickformat='.0%',
                    tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                    ticktext=['0%', '25%', '50%', '75%', '100%'],
                    xanchor='left',
                    x=1.02
                ),
                line=dict(width=1.5, color='white')
            ),
            showlegend=False,
            customdata=node_customdata
        )
        
        # Build the figure
        fig_data = edge_traces + [node_trace]
        
        # Add cluster legend entries only for populated clusters in the graph
        if populated_clusters:
            for cluster_id in populated_clusters:
                cluster_nodes = [n for n in G.nodes() if communities.get(n, -1) == cluster_id]
                symbol = cluster_to_symbol[cluster_id]
                
                # Add legend entry
                legend_trace = go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    name=f'Cluster {cluster_id} (n={len(cluster_nodes)})',
                    marker=dict(
                        size=12,
                        symbol=symbol,
                        color='#666',
                        line=dict(width=1, color='white')
                    ),
                    showlegend=True
                )
                fig_data.append(legend_trace)
        
        # Calculate summary stats for annotation
        avg_prob = np.mean(list(bot_scores.values())) if bot_scores else 0
        max_prob = max(bot_scores.values()) if bot_scores else 0
        high_prob_count = sum(1 for p in bot_scores.values() if p > 0.5)
        
        n_clusters = len(populated_clusters)
        
        # Create figure
        fig = go.Figure(
            data=fig_data,
            layout=go.Layout(
                title=dict(
                    text=f"<b>{title}</b>",
                    font=dict(size=20, family='Arial'),
                    x=0.5
                ),
                showlegend=True if populated_clusters else False,
                legend=dict(
                    title=dict(text='<b>Botnet Clusters</b>'),
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#666',
                    borderwidth=1
                ),
                hovermode='closest',
                margin=dict(b=100, l=20, r=180, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='#FAFAFA',
                paper_bgcolor='white',
                annotations=[
                    dict(
                        text=f"<b>Summary:</b> {G.number_of_nodes()} accounts | "
                             f"Avg bot prob: {avg_prob:.1%} | "
                             f"Max: {max_prob:.1%} | "
                             f"High risk (>50%): {high_prob_count} | "
                             f"Clusters: {n_clusters}",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.12,
                        showarrow=False,
                        font=dict(size=12, color='#333'),
                        bgcolor='#f0f0f0',
                        borderpad=6
                    ),
                    dict(
                        text="<b>Visual Encoding:</b> Node size = number of connections | "
                             "Color = bot probability (red=high, green=low) | "
                             "Shape = cluster membership | "
                             "Edges = coordination relationships",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.18,
                        showarrow=False,
                        font=dict(size=10, color='#666')
                    ),
                    dict(
                        text="<b>Layout:</b> Nodes are positioned using a force-directed algorithm. "
                             "Accounts with similar coordination patterns are placed closer together. "
                             "The spatial arrangement helps identify botnet clusters.",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.25,
                        showarrow=False,
                        font=dict(size=9, color='#888'),
                        align='left'
                    )
                ]
            )
        )
        
        # Save HTML with click-to-inspect panel
        output_path = os.path.join(self.output_dir, filename)
        fig_html = fig.to_html(full_html=False, include_plotlyjs=True, div_id="botnet-graph")
        
        html_template = """<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'>
  <title>Bot Coordination Network</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f7f7f7; color: #222; }
    .layout { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; padding: 16px; }
    .chart-container { background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 8px; }
    .panel { background: #fff; border: 1px solid #e5e5e5; border-radius: 8px; padding: 16px; max-height: 90vh; overflow: auto; }
    .panel h2 { margin: 0 0 8px 0; }
    .panel h3 { margin: 16px 0 8px 0; }
    .meta { color: #555; font-size: 13px; line-height: 1.4; margin-bottom: 8px; }
    .comment { border: 1px solid #eee; border-radius: 6px; padding: 8px; margin-bottom: 8px; background: #fafafa; }
    .comment .small { color: #666; font-size: 12px; margin-bottom: 4px; }
    .comment .text { white-space: pre-wrap; }
    a.button { display: inline-block; padding: 6px 10px; background: #2a9d8f; color: #fff; text-decoration: none; border-radius: 4px; margin-top: 6px; }
    a.button:hover { background: #23867b; }
    @media (max-width: 900px) {
      .layout { grid-template-columns: 1fr; }
      .panel { order: -1; }
    }
  </style>
</head>
<body>
  <div class="layout">
    <div class="chart-container">
      {{FIG_PLACEHOLDER}}
    </div>
    <div class="panel" id="account-panel">
      <h2>Select an account</h2>
      <p>Click a node in the graph to see its comments and channel link.</p>
    </div>
  </div>
  <script>
    const plot = document.getElementById('botnet-graph') || document.querySelector('.plotly-graph-div');
    const panel = document.getElementById('account-panel');
    const escapeHtml = (unsafe) => {
      if (unsafe === undefined || unsafe === null) return '';
      return String(unsafe)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/\"/g, '&quot;')
        .replace(/'/g, '&#039;');
    };
    const renderAccount = (meta) => {
      if (!meta) {
        panel.innerHTML = "<h2>Select an account</h2><p>Click a node in the graph to see its comments and channel link.</p>";
        return;
      }
      const bp = parseFloat(meta.bot_probability || 0);
      const comments = (meta.comments || []).map(c => {
        const text = escapeHtml(c.text || '');
        const vid = escapeHtml(c.video_id || '');
        const cid = escapeHtml(c.comment_id || '');
        const ts = escapeHtml(c.published_at || '');
        return `<div class='comment'>
                  <div class='small'>Video: ${vid} &middot; ${ts} &middot; Comment ID: ${cid}</div>
                  <div class='text'>${text}</div>
                </div>`;
      }).join('') || "<p class='meta'>No comments captured for this account.</p>";
      const chan = meta.channel_url ? `<a class='button' href='${escapeHtml(meta.channel_url)}' target='_blank' rel='noopener'>Open channel</a>` : '';
      panel.innerHTML = `
        <h2>${escapeHtml(meta.author_name || meta.author_id || 'Account')}</h2>
        <div class='meta'>
          Account ID: ${escapeHtml(meta.author_id || '')}<br>
          Cluster: ${escapeHtml(meta.cluster_id)}<br>
          Bot probability: ${(bp * 100).toFixed(1)}%<br>
          Comments analyzed: ${escapeHtml(meta.total_comments || 0)} &middot; Videos: ${escapeHtml(meta.videos_commented || 0)}<br>
          ${chan}
        </div>
        <h3>Recent comments</h3>
        ${comments}
      `;
    };
    renderAccount(null);
    if (plot && plot.on) {
      plot.on('plotly_click', function(evt) {
        if (!evt || !evt.points || !evt.points.length) return;
        const pt = evt.points[0];
        if (!pt || !pt.customdata) return;
        renderAccount(pt.customdata);
      });
    }
  </script>
</body>
</html>
"""
        html_page = html_template.replace("{{FIG_PLACEHOLDER}}", fig_html)
        
        with open(output_path, 'w') as f:
            f.write(html_page)
        
        # Also save static PNG for paper (if kaleido is properly configured)
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1200, height=800, scale=2)
            logger.info(f"Saved network visualization to {output_path} and {png_path}")
        except Exception as e:
            logger.warning(f"Could not save PNG (HTML still saved): {e}")
            logger.info(f"Saved network visualization to {output_path}")
        
        return output_path
    
    def visualize_cluster_analysis(self, detection_results: pd.DataFrame,
                                   filename: str = "cluster_analysis.html") -> str:
        """
        Create comprehensive cluster analysis visualization for papers
        """
        cluster_data = detection_results[detection_results['cluster_id'] != -1].copy()
        
        if cluster_data.empty:
            logger.warning("No clusters found for visualization")
            return ""
        
        # Aggregate by cluster
        cluster_summary = cluster_data.groupby('cluster_id').agg({
            'final_bot_probability': ['mean', 'std', 'count'],
            'classification': lambda x: (x == 'likely_bot').sum()
        }).round(3)
        cluster_summary.columns = ['avg_bot_prob', 'std_bot_prob', 'size', 'bot_count']
        cluster_summary = cluster_summary.reset_index()
        cluster_summary['cluster_label'] = cluster_summary['cluster_id'].apply(lambda x: f'Cluster {x}')
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Average Bot Probability by Cluster</b>',
                '<b>Cluster Size Distribution</b>',
                '<b>Bot Classification by Cluster</b>',
                '<b>Bot Probability Distribution per Cluster</b>'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "box"}]],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Assign colors to clusters
        colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(cluster_summary))]
        
        # 1. Average bot probability
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_label'],
                y=cluster_summary['avg_bot_prob'],
                error_y=dict(type='data', array=cluster_summary['std_bot_prob'], color='#333'),
                marker_color=colors,
                text=[f'{p:.1%}' for p in cluster_summary['avg_bot_prob']],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. Cluster sizes
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_label'],
                y=cluster_summary['size'],
                marker_color=colors,
                text=cluster_summary['size'],
                textposition='outside',
                showlegend=False
            ),
            row=1, col=2
        )
        
        # 3. Bot count per cluster (stacked bar)
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_label'],
                y=cluster_summary['bot_count'],
                name='Likely Bots',
                marker_color='#E63946',
                text=cluster_summary['bot_count'],
                textposition='inside'
            ),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_label'],
                y=cluster_summary['size'] - cluster_summary['bot_count'],
                name='Uncertain',
                marker_color='#457B9D',
                text=cluster_summary['size'] - cluster_summary['bot_count'],
                textposition='inside'
            ),
            row=2, col=1
        )
        
        # 4. Box plots of probability distribution
        for i, cluster_id in enumerate(sorted(cluster_data['cluster_id'].unique())):
            probs = cluster_data[cluster_data['cluster_id'] == cluster_id]['final_bot_probability']
            fig.add_trace(
                go.Box(
                    y=probs,
                    name=f'Cluster {cluster_id}',
                    marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    boxmean='sd',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title=dict(
                text='<b>Botnet Cluster Analysis</b>',
                font=dict(size=22, family='Arial'),
                x=0.5
            ),
            barmode='stack',
            height=900,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=0.48,
                xanchor='center',
                x=0.25
            ),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        # Update axes labels
        fig.update_yaxes(title_text='Bot Probability', row=1, col=1)
        fig.update_yaxes(title_text='Number of Accounts', row=1, col=2)
        fig.update_yaxes(title_text='Number of Accounts', row=2, col=1)
        fig.update_yaxes(title_text='Bot Probability', row=2, col=2)
        
        # Add gridlines
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
        fig.update_xaxes(showgrid=False)
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1200, height=900, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"Saved cluster analysis to {output_path}")
        return output_path
    
    def visualize_coordination_heatmap(self, coordination_matrix: np.ndarray,
                                       author_ids: List[str],
                                       cluster_assignments: Dict[str, int],
                                       filename: str = "coordination_heatmap.html") -> str:
        """
        Create a heatmap of the coordination matrix, sorted by cluster
        """
        n = len(author_ids)
        if n > 100:
            # Sample for visualization
            sample_idx = np.random.choice(n, 100, replace=False)
            coordination_matrix = coordination_matrix[np.ix_(sample_idx, sample_idx)]
            author_ids = [author_ids[i] for i in sample_idx]
        
        # Sort by cluster
        sorted_indices = sorted(range(len(author_ids)), 
                               key=lambda i: (cluster_assignments.get(author_ids[i], 999), author_ids[i]))
        
        sorted_matrix = coordination_matrix[np.ix_(sorted_indices, sorted_indices)]
        sorted_authors = [author_ids[i][:15] + '...' if len(author_ids[i]) > 15 else author_ids[i] 
                         for i in sorted_indices]
        sorted_clusters = [cluster_assignments.get(author_ids[i], -1) for i in sorted_indices]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=sorted_matrix,
            x=sorted_authors,
            y=sorted_authors,
            colorscale='Reds',
            colorbar=dict(title='Coordination<br>Score'),
            hovertemplate='Author 1: %{y}<br>Author 2: %{x}<br>Coordination: %{z:.3f}<extra></extra>'
        ))
        
        # Add cluster boundaries
        cluster_changes = [0]
        prev_cluster = sorted_clusters[0]
        for i, c in enumerate(sorted_clusters):
            if c != prev_cluster:
                cluster_changes.append(i)
                prev_cluster = c
        cluster_changes.append(len(sorted_clusters))
        
        # Add cluster labels
        annotations = []
        for i in range(len(cluster_changes) - 1):
            start, end = cluster_changes[i], cluster_changes[i + 1]
            mid = (start + end) / 2
            cluster_id = sorted_clusters[start]
            label = 'Unclustered' if cluster_id == -1 else f'Cluster {cluster_id}'
            
            # Add boundary lines
            if i > 0:
                fig.add_hline(y=start - 0.5, line_dash='dash', line_color='#333', line_width=2)
                fig.add_vline(x=start - 0.5, line_dash='dash', line_color='#333', line_width=2)
        
        fig.update_layout(
            title=dict(
                text='<b>Account Coordination Matrix</b><br><sub>Darker = higher coordination between accounts</sub>',
                font=dict(size=18, family='Arial'),
                x=0.5
            ),
            xaxis=dict(title='Account', tickangle=45, tickfont=dict(size=8)),
            yaxis=dict(title='Account', tickfont=dict(size=8)),
            width=1000,
            height=900,
            plot_bgcolor='white'
        )
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1000, height=900, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"Saved coordination heatmap to {output_path}")
        return output_path
    
    def visualize_cross_video_analysis(self, comments_df: pd.DataFrame,
                                       detection_results: pd.DataFrame,
                                       cross_video_scores: Dict[str, float],
                                       filename: str = "cross_video_analysis.html") -> str:
        """
        Visualize cross-video bot activity patterns
        """
        # Get video information
        video_counts = comments_df.groupby('video_id').agg({
            'author_id': 'nunique',
            'comment_id': 'count'
        }).rename(columns={'author_id': 'unique_authors', 'comment_id': 'total_comments'})
        
        # Count multi-video authors
        author_video_counts = comments_df.groupby('author_id')['video_id'].nunique()
        multi_video_authors = author_video_counts[author_video_counts > 1]
        
        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                '<b>Cross-Video Coordination Score Distribution</b>',
                '<b>Authors Active Across Multiple Videos</b>',
                '<b>Comments Per Video</b>',
                '<b>Cross-Video Score vs Bot Probability</b>'
            ),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "scatter"}]],
            vertical_spacing=0.15
        )
        
        # 1. Distribution of cross-video scores
        scores = list(cross_video_scores.values())
        fig.add_trace(
            go.Histogram(
                x=scores,
                nbinsx=30,
                marker_color='#457B9D',
                name='Cross-Video Score'
            ),
            row=1, col=1
        )
        
        # 2. Multi-video author distribution
        video_count_dist = multi_video_authors.value_counts().sort_index()
        fig.add_trace(
            go.Bar(
                x=[f'{v} videos' for v in video_count_dist.index],
                y=video_count_dist.values,
                marker_color='#E63946',
                text=video_count_dist.values,
                textposition='outside',
                name='Authors'
            ),
            row=1, col=2
        )
        
        # 3. Comments per video
        video_labels = [f'Video {i+1}' for i in range(len(video_counts))]
        fig.add_trace(
            go.Bar(
                x=video_labels,
                y=video_counts['total_comments'].values,
                marker_color='#2A9D8F',
                text=video_counts['total_comments'].values,
                textposition='outside',
                name='Comments'
            ),
            row=2, col=1
        )
        
        # 4. Scatter: cross-video score vs bot probability
        merged = detection_results.copy()
        merged['cross_video_score'] = merged['author_id'].map(cross_video_scores).fillna(0)
        
        # Color by cluster
        cluster_ids = merged['cluster_id'].values
        colors = [CLUSTER_COLORS[c % len(CLUSTER_COLORS)] if c != -1 else NOISE_COLOR 
                  for c in cluster_ids]
        
        fig.add_trace(
            go.Scatter(
                x=merged['cross_video_score'],
                y=merged['final_bot_probability'],
                mode='markers',
                marker=dict(
                    color=colors,
                    size=8,
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=[f"Author: {a}<br>Cluster: {c}" for a, c in zip(merged['author_id'], merged['cluster_id'])],
                hoverinfo='text',
                name='Accounts'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='<b>Cross-Video Bot Activity Analysis</b>',
                font=dict(size=20, family='Arial'),
                x=0.5
            ),
            height=900,
            width=1200,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text='Cross-Video Coordination Score', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Number of Authors', row=1, col=2)
        fig.update_yaxes(title_text='Number of Comments', row=2, col=1)
        fig.update_xaxes(title_text='Cross-Video Coordination Score', row=2, col=2)
        fig.update_yaxes(title_text='Bot Probability', row=2, col=2)
        
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1200, height=900, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"Saved cross-video analysis to {output_path}")
        return output_path
    
    def visualize_temporal_patterns(self, comments_df: pd.DataFrame,
                                   detection_results: pd.DataFrame,
                                   filename: str = "temporal_patterns.html") -> str:
        """
        Visualize temporal patterns of bot vs human comments - paper ready
        """
        comments_df = comments_df.copy()
        
        # Map bot classification to comments
        author_classification = detection_results.set_index('author_id')['classification'].to_dict()
        comments_df['is_bot'] = comments_df['author_id'].map(
            lambda x: author_classification.get(x, 'unknown') == 'likely_bot'
        )
        comments_df['classification'] = comments_df['is_bot'].map(
            {True: 'Likely Bot', False: 'Likely Human'}
        )
        
        # Convert timestamp
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        comments_df['hour'] = comments_df['timestamp'].dt.hour
        
        # Aggregate by hour
        hourly_counts = comments_df.groupby(['hour', 'classification']).size().reset_index(name='count')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(
                '<b>Comment Activity by Hour of Day</b>',
                '<b>Bot vs Human Comment Distribution</b>'
            ),
            specs=[[{"type": "scatter"}, {"type": "pie"}]]
        )
        
        # 1. Line chart by hour
        for classification, color in [('Likely Bot', '#E63946'), ('Likely Human', '#457B9D')]:
            data = hourly_counts[hourly_counts['classification'] == classification]
            fig.add_trace(
                go.Scatter(
                    x=data['hour'],
                    y=data['count'],
                    mode='lines+markers',
                    name=classification,
                    line=dict(color=color, width=3),
                    marker=dict(size=8)
                ),
                row=1, col=1
            )
        
        # 2. Pie chart
        classification_counts = comments_df['classification'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=classification_counts.index,
                values=classification_counts.values,
                marker=dict(colors=['#E63946', '#457B9D']),
                textinfo='label+percent',
                textfont=dict(size=14)
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=dict(
                text='<b>Temporal Analysis of Bot Activity</b>',
                font=dict(size=20, family='Arial'),
                x=0.5
            ),
            height=500,
            width=1200,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=-0.2,
                xanchor='center',
                x=0.25
            ),
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(
            title_text='Hour of Day (UTC)',
            tickmode='linear',
            tick0=0,
            dtick=2,
            row=1, col=1
        )
        fig.update_yaxes(title_text='Number of Comments', row=1, col=1)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#EEE')
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1200, height=500, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"Saved temporal patterns to {output_path}")
        return output_path
    
    def create_summary_figure(self, detection_results: pd.DataFrame,
                             comments_df: pd.DataFrame,
                             cross_video_scores: Dict[str, float] = None,
                             filename: str = "detection_summary.html") -> str:
        """
        Create a single comprehensive summary figure for papers
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                '<b>Bot Probability Distribution</b>',
                '<b>Classification Results</b>',
                '<b>Accounts per Cluster</b>',
                '<b>Cluster Bot Probabilities</b>',
                '<b>Cross-Video Activity</b>',
                '<b>Key Metrics</b>'
            ),
            specs=[
                [{"type": "histogram"}, {"type": "pie"}, {"type": "bar"}],
                [{"type": "box"}, {"type": "scatter"}, {"type": "table"}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.08
        )
        
        # 1. Histogram of bot probabilities
        fig.add_trace(
            go.Histogram(
                x=detection_results['final_bot_probability'],
                nbinsx=20,
                marker_color='#457B9D',
                name='Bot Probability'
            ),
            row=1, col=1
        )
        
        # 2. Pie chart of classifications
        class_counts = detection_results['classification'].value_counts()
        fig.add_trace(
            go.Pie(
                labels=['Likely Bot' if x == 'likely_bot' else 'Likely Human' for x in class_counts.index],
                values=class_counts.values,
                marker=dict(colors=['#E63946', '#457B9D']),
                textinfo='label+percent'
            ),
            row=1, col=2
        )
        
        # 3. Cluster sizes
        cluster_counts = detection_results[detection_results['cluster_id'] != -1].groupby('cluster_id').size()
        if not cluster_counts.empty:
            colors = [CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(len(cluster_counts))]
            fig.add_trace(
                go.Bar(
                    x=[f'Cluster {c}' for c in cluster_counts.index],
                    y=cluster_counts.values,
                    marker_color=colors,
                    text=cluster_counts.values,
                    textposition='outside'
                ),
                row=1, col=3
            )
        
        # 4. Box plots by cluster
        for i, cluster_id in enumerate(sorted(detection_results['cluster_id'].unique())):
            if cluster_id == -1:
                continue
            probs = detection_results[detection_results['cluster_id'] == cluster_id]['final_bot_probability']
            fig.add_trace(
                go.Box(
                    y=probs,
                    name=f'C{cluster_id}',
                    marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 5. Cross-video scatter
        if cross_video_scores:
            merged = detection_results.copy()
            merged['cross_video_score'] = merged['author_id'].map(cross_video_scores).fillna(0)
            fig.add_trace(
                go.Scatter(
                    x=merged['cross_video_score'],
                    y=merged['final_bot_probability'],
                    mode='markers',
                    marker=dict(
                        color=merged['final_bot_probability'],
                        colorscale='RdYlGn_r',
                        size=6,
                        opacity=0.6
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # 6. Summary table
        n_clusters = detection_results[detection_results['cluster_id'] != -1]['cluster_id'].nunique()
        n_bots = (detection_results['classification'] == 'likely_bot').sum()
        avg_prob = detection_results['final_bot_probability'].mean()
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#457B9D',
                    font=dict(color='white', size=12),
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Total Accounts', 'Total Comments', 'Botnet Clusters', 
                         'Likely Bots', 'Avg Bot Prob', 'High Conf Bots (>90%)'],
                        [
                            len(detection_results),
                            len(comments_df),
                            n_clusters,
                            n_bots,
                            f'{avg_prob:.1%}',
                            (detection_results['final_bot_probability'] > 0.9).sum()
                        ]
                    ],
                    fill_color=[['#f9f9f9', 'white'] * 3],
                    align='left',
                    font=dict(size=11)
                )
            ),
            row=2, col=3
        )
        
        fig.update_layout(
            title=dict(
                text='<b>BotBuster Detection Summary</b>',
                font=dict(size=22, family='Arial'),
                x=0.5
            ),
            height=800,
            width=1400,
            showlegend=False,
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        
        fig.update_xaxes(title_text='Bot Probability', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        fig.update_yaxes(title_text='Accounts', row=1, col=3)
        fig.update_xaxes(title_text='Cluster', row=2, col=1)
        fig.update_yaxes(title_text='Bot Probability', row=2, col=1)
        fig.update_xaxes(title_text='Cross-Video Score', row=2, col=2)
        fig.update_yaxes(title_text='Bot Probability', row=2, col=2)
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        
        try:
            png_path = output_path.replace('.html', '.png')
            fig.write_image(png_path, width=1400, height=800, scale=2)
        except Exception as e:
            logger.warning(f"Could not save PNG: {e}")
        
        logger.info(f"Saved summary figure to {output_path}")
        return output_path
    
    def create_summary_report(self, detection_results: pd.DataFrame,
                            comments_df: pd.DataFrame,
                            network_metrics: Optional[Dict] = None) -> Dict:
        """
        Create a summary report of detection results
        """
        summary = {
            'total_accounts': len(detection_results),
            'total_comments': len(comments_df),
            'classification_counts': detection_results['classification'].value_counts().to_dict(),
            'avg_bot_probability': float(detection_results['final_bot_probability'].mean()),
            'high_confidence_bots': int((detection_results['final_bot_probability'] > 0.9).sum()),
            'clusters_found': int(detection_results['cluster_id'].nunique() - 1),
            'noise_points': int((detection_results['cluster_id'] == -1).sum())
        }
        
        top_bots = detection_results.nlargest(10, 'final_bot_probability')[
            ['author_id', 'final_bot_probability', 'cluster_id']
        ].to_dict('records')
        summary['top_bot_accounts'] = top_bots
        
        cluster_stats = []
        for cluster_id in detection_results['cluster_id'].unique():
            if cluster_id == -1:
                continue
            cluster_data = detection_results[detection_results['cluster_id'] == cluster_id]
            cluster_stats.append({
                'cluster_id': int(cluster_id),
                'size': int(len(cluster_data)),
                'avg_bot_probability': float(cluster_data['final_bot_probability'].mean()),
                'bot_count': int((cluster_data['classification'] == 'likely_bot').sum())
            })
        
        summary['cluster_statistics'] = sorted(cluster_stats, 
                                              key=lambda x: x['avg_bot_probability'], 
                                              reverse=True)
        
        if network_metrics:
            summary['network_metrics'] = network_metrics
        
        return summary
    
    def _calculate_layout(self, G: nx.Graph) -> Dict:
        """Calculate graph layout based on configuration"""
        if Config.GRAPH_LAYOUT == 'spring':
            return nx.spring_layout(G, k=1/np.sqrt(max(G.number_of_nodes(), 1)), iterations=50, seed=42)
        elif Config.GRAPH_LAYOUT == 'circular':
            return nx.circular_layout(G)
        elif Config.GRAPH_LAYOUT == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        else:
            return nx.spring_layout(G, seed=42)
