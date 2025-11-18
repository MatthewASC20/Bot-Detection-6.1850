"""
Network feature extraction for detecting coordinated bot networks
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import logging
from collections import defaultdict
import community as community_louvain

from config.config import Config

logger = logging.getLogger(__name__)

class NetworkFeatures:
    """Extract network-based features from user interactions"""
    
    @staticmethod
    def build_co_occurrence_network(comments_df: pd.DataFrame) -> nx.Graph:
        """
        Build network where nodes are authors and edges represent co-occurrence
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        # Group by video to find co-occurrences
        video_groups = comments_df.groupby('video_id')
        
        edge_weights = defaultdict(int)
        
        for video_id, group in video_groups:
            authors = group['author_id'].unique()
            
            # Create edges between all authors who commented on same video
            for i in range(len(authors)):
                for j in range(i+1, len(authors)):
                    edge_key = tuple(sorted([authors[i], authors[j]]))
                    edge_weights[edge_key] += 1
        
        # Add edges with weights to graph
        for (author1, author2), weight in edge_weights.items():
            if weight >= Config.MIN_EDGE_WEIGHT:
                G.add_edge(author1, author2, weight=weight)
        
        logger.info(f"Built co-occurrence network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    @staticmethod
    def build_reply_network(comments_df: pd.DataFrame) -> nx.DiGraph:
        """
        Build directed network of reply relationships
        
        Args:
            comments_df: DataFrame with comments including replies
            
        Returns:
            Directed NetworkX graph
        """
        G = nx.DiGraph()
        
        # Filter for replies
        replies_df = comments_df[comments_df['is_reply'] == True]
        
        for _, reply in replies_df.iterrows():
            if pd.notna(reply['parent_id']):
                # Find parent comment author
                parent_comment = comments_df[comments_df['comment_id'] == reply['parent_id']]
                if not parent_comment.empty:
                    parent_author = parent_comment.iloc[0]['author_id']
                    reply_author = reply['author_id']
                    
                    # Add edge from replier to parent
                    if G.has_edge(reply_author, parent_author):
                        G[reply_author][parent_author]['weight'] += 1
                    else:
                        G.add_edge(reply_author, parent_author, weight=1)
        
        logger.info(f"Built reply network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    @staticmethod
    def detect_communities(G: nx.Graph) -> Dict[str, int]:
        """
        Detect communities using Louvain algorithm
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping node to community ID
        """
        if G.number_of_nodes() == 0:
            return {}
        
        # Use Louvain community detection
        partition = community_louvain.best_partition(
            G, 
            resolution=Config.COMMUNITY_RESOLUTION
        )
        
        # Count community sizes
        community_sizes = defaultdict(int)
        for node, comm_id in partition.items():
            community_sizes[comm_id] += 1
        
        logger.info(f"Detected {len(community_sizes)} communities")
        for comm_id, size in sorted(community_sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  Community {comm_id}: {size} members")
        
        return partition
    
    @staticmethod
    def extract_network_metrics(G: nx.Graph) -> Dict[str, Dict]:
        """
        Extract network metrics for each node
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping node to metrics dictionary
        """
        metrics = {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        
        try:
            betweenness_centrality = nx.betweenness_centrality(G)
        except:
            betweenness_centrality = {node: 0 for node in G.nodes()}
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=100)
        except:
            eigenvector_centrality = {node: 0 for node in G.nodes()}
        
        try:
            pagerank = nx.pagerank(G, max_iter=100)
        except:
            pagerank = {node: 0 for node in G.nodes()}
        
        # Calculate clustering coefficient
        clustering = nx.clustering(G)
        
        for node in G.nodes():
            metrics[node] = {
                'degree': G.degree(node),
                'weighted_degree': G.degree(node, weight='weight'),
                'degree_centrality': degree_centrality.get(node, 0),
                'betweenness_centrality': betweenness_centrality.get(node, 0),
                'eigenvector_centrality': eigenvector_centrality.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'clustering_coefficient': clustering.get(node, 0),
                'neighbors': list(G.neighbors(node))
            }
        
        return metrics
    
    @staticmethod
    def find_cliques(G: nx.Graph, min_size: int = None) -> List[Set[str]]:
        """
        Find cliques (fully connected subgraphs) in the network
        
        Args:
            G: NetworkX graph
            min_size: Minimum clique size
            
        Returns:
            List of cliques (sets of nodes)
        """
        if min_size is None:
            min_size = Config.MIN_CLUSTER_SIZE
        
        cliques = []
        
        for clique in nx.find_cliques(G):
            if len(clique) >= min_size:
                cliques.append(set(clique))
        
        # Sort by size
        cliques.sort(key=len, reverse=True)
        
        logger.info(f"Found {len(cliques)} cliques with size >= {min_size}")
        if cliques:
            logger.info(f"  Largest clique has {len(cliques[0])} members")
        
        return cliques
    
    @staticmethod
    def calculate_network_cohesion(G: nx.Graph, node_group: Set[str]) -> float:
        """
        Calculate cohesion score for a group of nodes
        
        Args:
            G: NetworkX graph
            node_group: Set of node IDs
            
        Returns:
            Cohesion score (0-1)
        """
        if len(node_group) < 2:
            return 0.0
        
        # Create subgraph
        subgraph = G.subgraph(node_group)
        
        # Calculate density
        density = nx.density(subgraph)
        
        # Calculate average clustering
        clustering = nx.average_clustering(subgraph)
        
        # Combine metrics
        cohesion = (density + clustering) / 2
        
        return cohesion
    
    @staticmethod
    def detect_star_patterns(G: nx.Graph, min_degree: int = 10) -> List[Tuple[str, Set[str]]]:
        """
        Detect star patterns (one central node connected to many others)
        
        Args:
            G: NetworkX graph
            min_degree: Minimum degree for center node
            
        Returns:
            List of tuples (center_node, connected_nodes)
        """
        star_patterns = []
        
        for node in G.nodes():
            degree = G.degree(node)
            
            if degree >= min_degree:
                neighbors = set(G.neighbors(node))
                
                # Check if neighbors are poorly connected to each other
                neighbor_edges = 0
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 < n2 and G.has_edge(n1, n2):
                            neighbor_edges += 1
                
                possible_edges = len(neighbors) * (len(neighbors) - 1) / 2
                if possible_edges > 0:
                    neighbor_density = neighbor_edges / possible_edges
                    
                    # Star pattern if neighbors have low interconnection
                    if neighbor_density < 0.1:
                        star_patterns.append((node, neighbors))
        
        logger.info(f"Found {len(star_patterns)} star patterns")
        
        return star_patterns
    
    @staticmethod
    def build_temporal_network(comments_df: pd.DataFrame, 
                              time_window_hours: int = 1) -> nx.Graph:
        """
        Build network based on temporal proximity
        
        Args:
            comments_df: DataFrame with comments
            time_window_hours: Time window for connections
            
        Returns:
            NetworkX graph
        """
        G = nx.Graph()
        
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        comments_df = comments_df.sort_values('timestamp')
        
        # Create edges between authors posting close in time
        for i in range(len(comments_df)):
            current = comments_df.iloc[i]
            current_time = current['timestamp']
            
            # Look ahead within time window
            j = i + 1
            while j < len(comments_df):
                next_comment = comments_df.iloc[j]
                time_diff = (next_comment['timestamp'] - current_time).total_seconds() / 3600
                
                if time_diff > time_window_hours:
                    break
                
                if current['author_id'] != next_comment['author_id']:
                    if G.has_edge(current['author_id'], next_comment['author_id']):
                        G[current['author_id']][next_comment['author_id']]['weight'] += 1
                    else:
                        G.add_edge(current['author_id'], next_comment['author_id'], weight=1)
                
                j += 1
        
        return G
    
    @staticmethod
    def calculate_author_network_features(comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive network features for each author
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with network features per author
        """
        # Build different types of networks
        co_occurrence_net = NetworkFeatures.build_co_occurrence_network(comments_df)
        reply_net = NetworkFeatures.build_reply_network(comments_df)
        temporal_net = NetworkFeatures.build_temporal_network(comments_df)
        
        # Detect communities
        communities = NetworkFeatures.detect_communities(co_occurrence_net)
        
        # Extract metrics
        co_metrics = NetworkFeatures.extract_network_metrics(co_occurrence_net)
        
        # Find cliques and stars
        cliques = NetworkFeatures.find_cliques(co_occurrence_net)
        stars = NetworkFeatures.detect_star_patterns(co_occurrence_net)
        
        # Compile features for each author
        network_features = []
        unique_authors = comments_df['author_id'].unique()
        
        for author_id in unique_authors:
            features = {'author_id': author_id}
            
            # Co-occurrence network metrics
            if author_id in co_metrics:
                features.update({
                    f'co_{key}': value 
                    for key, value in co_metrics[author_id].items()
                    if key != 'neighbors'
                })
                features['co_neighbor_count'] = len(co_metrics[author_id]['neighbors'])
            else:
                features.update({
                    'co_degree': 0, 'co_weighted_degree': 0,
                    'co_degree_centrality': 0, 'co_betweenness_centrality': 0,
                    'co_eigenvector_centrality': 0, 'co_pagerank': 0,
                    'co_clustering_coefficient': 0, 'co_neighbor_count': 0
                })
            
            # Community membership
            features['community_id'] = communities.get(author_id, -1)
            if features['community_id'] != -1:
                community_members = [n for n, c in communities.items() if c == features['community_id']]
                features['community_size'] = len(community_members)
            else:
                features['community_size'] = 0
            
            # Reply network metrics
            features['replies_sent'] = reply_net.out_degree(author_id) if author_id in reply_net else 0
            features['replies_received'] = reply_net.in_degree(author_id) if author_id in reply_net else 0
            
            # Temporal network metrics
            features['temporal_degree'] = temporal_net.degree(author_id) if author_id in temporal_net else 0
            
            # Clique membership
            features['in_clique'] = any(author_id in clique for clique in cliques)
            features['max_clique_size'] = max(
                (len(clique) for clique in cliques if author_id in clique),
                default=0
            )
            
            # Star pattern involvement
            features['is_star_center'] = any(author_id == center for center, _ in stars)
            features['in_star_pattern'] = any(
                author_id == center or author_id in neighbors 
                for center, neighbors in stars
            )
            
            network_features.append(features)
        
        return pd.DataFrame(network_features)
