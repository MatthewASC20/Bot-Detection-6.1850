"""
Network visualization for bot cluster detection
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import logging
import os

from config.config import Config

logger = logging.getLogger(__name__)

class NetworkVisualizer:
    """Create network visualizations of bot clusters"""
    
    def __init__(self, output_dir: str = None):
        if output_dir is None:
            output_dir = Config.GRAPHS_DIR
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def visualize_bot_network(self, G: nx.Graph, 
                             bot_scores: Dict[str, float],
                             communities: Optional[Dict[str, int]] = None,
                             title: str = "YouTube Comment Network",
                             filename: str = "bot_network.html") -> str:
        """
        Create interactive network visualization with Plotly
        
        Args:
            G: NetworkX graph
            bot_scores: Dictionary mapping node to bot probability
            communities: Optional community assignments
            title: Plot title
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        # Limit graph size for readability
        if G.number_of_nodes() > Config.MAX_GRAPH_NODES:
            # Keep only nodes with highest bot scores
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
                    color='rgba(125, 125, 125, 0.5)'
                ),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        node_size = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node properties
            bot_score = bot_scores.get(node, 0)
            degree = G.degree(node)
            
            # Color by bot probability
            node_color.append(bot_score)
            
            # Size by degree
            node_size.append(5 + degree * Config.NODE_SIZE_MULTIPLIER / 10)
            
            # Hover text
            text = f"Author: {node}<br>"
            text += f"Bot Probability: {bot_score:.2%}<br>"
            text += f"Connections: {degree}<br>"
            if communities and node in communities:
                text += f"Community: {communities[node]}"
            node_text.append(text)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='RdYlGn_r',  # Red for bots, green for humans
                reversescale=False,
                color=node_color,
                size=node_size,
                colorbar=dict(
                    thickness=15,
                    title="Bot Probability",
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=0.5)
            )
        )
        
        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           plot_bgcolor='white'
                       ))
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        logger.info(f"Saved network visualization to {output_path}")
        
        return output_path
    
    def visualize_cluster_comparison(self, detection_results: pd.DataFrame,
                                    filename: str = "cluster_comparison.html") -> str:
        """
        Create visualization comparing different clusters
        
        Args:
            detection_results: DataFrame with detection results
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        # Filter to clusters only (not noise)
        cluster_data = detection_results[detection_results['cluster_id'] != -1].copy()
        
        if cluster_data.empty:
            logger.warning("No clusters found for visualization")
            return ""
        
        # Aggregate by cluster
        cluster_summary = cluster_data.groupby('cluster_id').agg({
            'final_bot_probability': ['mean', 'std', 'count'],
            'classification': lambda x: (x == 'likely_bot').mean()
        }).round(3)
        
        cluster_summary.columns = ['avg_bot_prob', 'std_bot_prob', 'size', 'bot_ratio']
        cluster_summary = cluster_summary.reset_index()
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Bot Probability', 'Cluster Sizes', 
                          'Bot Ratio by Cluster', 'Probability Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                  [{"type": "bar"}, {"type": "box"}]]
        )
        
        # 1. Bar chart of average bot probability
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_id'].astype(str),
                y=cluster_summary['avg_bot_prob'],
                error_y=dict(type='data', array=cluster_summary['std_bot_prob']),
                name='Bot Probability',
                marker_color='indianred'
            ),
            row=1, col=1
        )
        
        # 2. Pie chart of cluster sizes
        fig.add_trace(
            go.Pie(
                labels=cluster_summary['cluster_id'].astype(str),
                values=cluster_summary['size'],
                name='Cluster Size'
            ),
            row=1, col=2
        )
        
        # 3. Bot ratio by cluster
        fig.add_trace(
            go.Bar(
                x=cluster_summary['cluster_id'].astype(str),
                y=cluster_summary['bot_ratio'],
                name='Bot Ratio',
                marker_color='darkblue'
            ),
            row=2, col=1
        )
        
        # 4. Box plot of probability distribution
        for cluster_id in cluster_data['cluster_id'].unique():
            cluster_probs = cluster_data[cluster_data['cluster_id'] == cluster_id]['final_bot_probability']
            fig.add_trace(
                go.Box(
                    y=cluster_probs,
                    name=f'Cluster {cluster_id}',
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title_text="Bot Cluster Analysis",
            showlegend=False,
            height=800
        )
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        logger.info(f"Saved cluster comparison to {output_path}")
        
        return output_path
    
    def visualize_temporal_patterns(self, comments_df: pd.DataFrame,
                                   bot_labels: pd.Series,
                                   filename: str = "temporal_patterns.html") -> str:
        """
        Visualize temporal patterns of bot vs human comments
        
        Args:
            comments_df: DataFrame with comments
            bot_labels: Series with bot classifications
            filename: Output filename
            
        Returns:
            Path to saved visualization
        """
        # Merge bot labels
        comments_df = comments_df.copy()
        comments_df['is_bot'] = comments_df['author_id'].map(
            bot_labels.set_index('author_id')['classification'] == 'likely_bot'
        )
        
        # Convert to datetime
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        
        # Create hourly bins
        comments_df['hour'] = comments_df['timestamp'].dt.hour
        comments_df['day'] = comments_df['timestamp'].dt.date
        
        # Aggregate by hour
        hourly_counts = comments_df.groupby(['hour', 'is_bot']).size().reset_index(name='count')
        
        # Create line plot
        fig = px.line(hourly_counts, 
                     x='hour', 
                     y='count', 
                     color='is_bot',
                     title='Comment Activity by Hour of Day',
                     labels={'hour': 'Hour of Day', 'count': 'Number of Comments', 
                            'is_bot': 'Account Type'},
                     color_discrete_map={True: 'red', False: 'blue'})
        
        fig.update_layout(
            xaxis=dict(tickmode='linear', tick0=0, dtick=1),
            legend_title='Account Type',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save
        output_path = os.path.join(self.output_dir, filename)
        fig.write_html(output_path)
        logger.info(f"Saved temporal patterns to {output_path}")
        
        return output_path
    
    def create_summary_report(self, detection_results: pd.DataFrame,
                            comments_df: pd.DataFrame,
                            network_metrics: Optional[Dict] = None) -> Dict:
        """
        Create a summary report of detection results
        
        Args:
            detection_results: DataFrame with detection results
            comments_df: Original comments DataFrame
            network_metrics: Optional network analysis metrics
            
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'total_accounts': len(detection_results),
            'total_comments': len(comments_df),
            'classification_counts': detection_results['classification'].value_counts().to_dict(),
            'avg_bot_probability': detection_results['final_bot_probability'].mean(),
            'high_confidence_bots': (detection_results['final_bot_probability'] > 0.9).sum(),
            'clusters_found': detection_results['cluster_id'].nunique() - 1,  # Exclude noise
            'noise_points': (detection_results['cluster_id'] == -1).sum()
        }
        
        # Add top bot accounts
        top_bots = detection_results.nlargest(10, 'final_bot_probability')[
            ['author_id', 'final_bot_probability', 'cluster_id']
        ].to_dict('records')
        summary['top_bot_accounts'] = top_bots
        
        # Add cluster statistics
        cluster_stats = []
        for cluster_id in detection_results['cluster_id'].unique():
            if cluster_id == -1:
                continue
            
            cluster_data = detection_results[detection_results['cluster_id'] == cluster_id]
            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'avg_bot_probability': cluster_data['final_bot_probability'].mean(),
                'bot_count': (cluster_data['classification'] == 'likely_bot').sum()
            })
        
        summary['cluster_statistics'] = sorted(cluster_stats, 
                                              key=lambda x: x['avg_bot_probability'], 
                                              reverse=True)
        
        # Add network metrics if provided
        if network_metrics:
            summary['network_metrics'] = network_metrics
        
        return summary
    
    def _calculate_layout(self, G: nx.Graph) -> Dict:
        """Calculate graph layout based on configuration"""
        if Config.GRAPH_LAYOUT == 'spring':
            return nx.spring_layout(G, k=1/np.sqrt(G.number_of_nodes()), iterations=50)
        elif Config.GRAPH_LAYOUT == 'circular':
            return nx.circular_layout(G)
        elif Config.GRAPH_LAYOUT == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)
        else:
            return nx.spring_layout(G)
