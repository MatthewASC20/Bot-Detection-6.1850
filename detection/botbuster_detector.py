"""
BotBuster Detection Algorithm

Implements the BotBuster bot detection methodology as described in the paper:
"Detecting Coordinated Misinformation Botnets in Social Media"

Key features:
1. Synchronized posting times between accounts
2. Semantic similarity between comments (using OpenAI embeddings)
3. Semantic similarity to broader discussion

The algorithm clusters accounts based on coordinated activity and assigns
bot probabilities at both comment and account levels.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import logging
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN
import networkx as nx

from config.config import Config
from features.semantic_features import SemanticFeatures
from features.temporal_features import TemporalFeatures

logger = logging.getLogger(__name__)


class BotBusterDetector:
    """
    BotBuster coordinated botnet detection system.
    
    Implements detection based on:
    - Temporal synchronization (comments posted close together in time)
    - Semantic similarity (similar content using embeddings)
    - Discussion context (deviation from normal discussion patterns)
    """
    
    def __init__(self):
        self.semantic_features = SemanticFeatures()
        self.similarity_matrix = None
        self.temporal_matrix = None
        self.coordination_matrix = None
        self.comment_bot_probs = {}
        self.account_bot_probs = {}
        self.botnet_clusters = {}
        
    def calculate_temporal_synchronization(self, comments_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate temporal synchronization matrix between comments.
        
        Comments posted within a short time window by different authors
        are considered temporally synchronized.
        
        Args:
            comments_df: DataFrame with comments and timestamps
            
        Returns:
            Temporal synchronization matrix (n_comments x n_comments)
        """
        logger.info("Calculating temporal synchronization matrix...")
        
        n = len(comments_df)
        temporal_matrix = np.zeros((n, n))
        
        # Convert timestamps
        timestamps = pd.to_datetime(comments_df['published_at']).values
        author_ids = comments_df['author_id'].values
        video_ids = comments_df['video_id'].values
        
        window_seconds = Config.TEMPORAL_WINDOW_SECONDS
        
        for i in range(n):
            for j in range(i + 1, n):
                # Only consider comments on the same video by different authors
                if video_ids[i] != video_ids[j]:
                    continue
                if author_ids[i] == author_ids[j]:
                    continue
                
                # Calculate time difference in seconds
                time_diff = abs((timestamps[i] - timestamps[j]).astype('timedelta64[s]').astype(float))
                
                if time_diff <= window_seconds:
                    # Synchronization score: higher when closer in time
                    # Score of 1.0 for same time, decreasing to 0 at window boundary
                    sync_score = 1.0 - (time_diff / window_seconds)
                    temporal_matrix[i, j] = sync_score
                    temporal_matrix[j, i] = sync_score
        
        self.temporal_matrix = temporal_matrix
        logger.info(f"Temporal synchronization matrix computed: shape {temporal_matrix.shape}")
        
        return temporal_matrix
    
    def calculate_coordination_score(self, comments_df: pd.DataFrame) -> np.ndarray:
        """
        Calculate coordination score between comments.
        
        Coordination = weighted average of:
        - Temporal synchronization (posting at similar times)
        - Semantic similarity (similar content)
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Coordination matrix (n_comments x n_comments)
        """
        logger.info("Calculating coordination scores...")
        
        # Get temporal synchronization matrix
        temporal_matrix = self.calculate_temporal_synchronization(comments_df)
        
        # Get semantic similarity matrix
        semantic_matrix = self.semantic_features.calculate_pairwise_semantic_similarity(comments_df)
        self.similarity_matrix = semantic_matrix
        
        # Calculate coordination as weighted average
        temporal_weight = Config.TEMPORAL_SIMILARITY_WEIGHT
        semantic_weight = Config.SEMANTIC_SIMILARITY_WEIGHT
        
        coordination_matrix = (
            temporal_weight * temporal_matrix + 
            semantic_weight * semantic_matrix
        )
        
        # Zero out diagonal and same-author comparisons
        np.fill_diagonal(coordination_matrix, 0)
        author_ids = comments_df['author_id'].values
        for i in range(len(author_ids)):
            for j in range(len(author_ids)):
                if author_ids[i] == author_ids[j]:
                    coordination_matrix[i, j] = 0
        
        self.coordination_matrix = coordination_matrix
        logger.info(f"Coordination matrix computed: shape {coordination_matrix.shape}")
        
        return coordination_matrix
    
    def build_coordination_graph(self, comments_df: pd.DataFrame,
                                  coordination_matrix: np.ndarray,
                                  threshold: float = 0.5) -> nx.Graph:
        """
        Build a coordination graph where:
        - Nodes are accounts (author_ids)
        - Edges represent coordination level between accounts
        
        Args:
            comments_df: DataFrame with comments
            coordination_matrix: Pairwise coordination scores
            threshold: Minimum coordination score to create edge
            
        Returns:
            NetworkX graph of account coordination
        """
        logger.info("Building coordination graph...")
        
        G = nx.Graph()
        
        # Add all authors as nodes
        unique_authors = comments_df['author_id'].unique()
        for author in unique_authors:
            G.add_node(author)
        
        # Calculate average coordination between each pair of authors
        author_ids = comments_df['author_id'].values
        comment_ids = comments_df['comment_id'].tolist()
        
        # Build mapping from author to their comment indices
        author_to_indices = {}
        for idx, author in enumerate(author_ids):
            if author not in author_to_indices:
                author_to_indices[author] = []
            author_to_indices[author].append(idx)
        
        # Calculate pairwise author coordination
        authors_list = list(unique_authors)
        for i, author1 in enumerate(authors_list):
            for author2 in authors_list[i+1:]:
                # Get all coordination scores between these authors' comments
                indices1 = author_to_indices.get(author1, [])
                indices2 = author_to_indices.get(author2, [])
                
                if not indices1 or not indices2:
                    continue
                
                coord_scores = []
                for idx1 in indices1:
                    for idx2 in indices2:
                        coord_scores.append(coordination_matrix[idx1, idx2])
                
                if coord_scores:
                    avg_coordination = np.mean(coord_scores)
                    max_coordination = np.max(coord_scores)
                    
                    # Use combination of avg and max
                    coordination = 0.5 * avg_coordination + 0.5 * max_coordination
                    
                    if coordination >= threshold:
                        G.add_edge(author1, author2, weight=coordination)
        
        logger.info(f"Built coordination graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return G
    
    def detect_botnet_clusters(self, comments_df: pd.DataFrame,
                                coordination_graph: nx.Graph) -> Dict[str, int]:
        """
        Detect botnet clusters based on coordination patterns.
        
        Uses community detection to find groups of highly coordinated accounts.
        
        Args:
            comments_df: DataFrame with comments
            coordination_graph: Graph of account coordination
            
        Returns:
            Dictionary mapping author_id to cluster_id
        """
        logger.info("Detecting botnet clusters...")
        
        if coordination_graph.number_of_edges() == 0:
            logger.warning("No coordination edges found - all accounts assigned to cluster -1")
            return {author: -1 for author in comments_df['author_id'].unique()}
        
        # Extract adjacency matrix for clustering
        nodes = list(coordination_graph.nodes())
        n = len(nodes)
        
        if n == 0:
            return {}
        
        # Build distance matrix (1 - coordination)
        adj_matrix = nx.to_numpy_array(coordination_graph, nodelist=nodes)
        
        # Convert to distance for DBSCAN (higher coordination = lower distance)
        distance_matrix = 1 - adj_matrix
        np.fill_diagonal(distance_matrix, 0)
        
        # Use DBSCAN with precomputed distances
        clustering = DBSCAN(
            eps=0.5,  # Distance threshold
            min_samples=Config.MIN_CLUSTER_SIZE,
            metric='precomputed'
        )
        
        labels = clustering.fit_predict(distance_matrix)
        
        # Map labels to author IDs
        cluster_assignments = {}
        for i, node in enumerate(nodes):
            cluster_assignments[node] = int(labels[i])
        
        # Assign -1 to any authors not in the graph
        for author in comments_df['author_id'].unique():
            if author not in cluster_assignments:
                cluster_assignments[author] = -1
        
        # Count clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"Found {n_clusters} botnet clusters, {n_noise} noise points")
        
        self.botnet_clusters = cluster_assignments
        return cluster_assignments
    
    def calculate_comment_bot_probabilities(self, comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate bot probability for each comment.
        
        Based on:
        - Semantic similarity to other comments
        - Semantic similarity to broader discussion
        - Temporal coordination patterns
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping comment_id to bot probability
        """
        logger.info("Calculating per-comment bot probabilities...")
        
        if self.similarity_matrix is None:
            raise ValueError("Must call calculate_coordination_score first")
        
        # Get discussion similarity scores
        discussion_scores = self.semantic_features.calculate_comment_to_discussion_similarity(comments_df)
        
        # Calculate semantic-based bot probabilities
        semantic_bot_probs = self.semantic_features.calculate_comment_bot_probability(
            comments_df, self.similarity_matrix, discussion_scores
        )
        
        # Incorporate temporal synchronization
        comment_ids = comments_df['comment_id'].tolist()
        author_ids = comments_df['author_id'].values
        
        final_bot_probs = {}
        
        for i, comment_id in enumerate(comment_ids):
            # Get temporal synchronization with other authors
            temporal_sync_scores = []
            for j in range(len(comment_ids)):
                if i != j and author_ids[i] != author_ids[j]:
                    temporal_sync_scores.append(self.temporal_matrix[i, j])
            
            if temporal_sync_scores:
                avg_temporal_sync = np.mean(temporal_sync_scores)
                max_temporal_sync = np.max(temporal_sync_scores)
                temporal_score = 0.4 * avg_temporal_sync + 0.6 * max_temporal_sync
            else:
                temporal_score = 0.0
            
            # Get semantic probability
            semantic_prob = semantic_bot_probs.get(comment_id, 0.0)
            
            # Combine: both temporal and semantic coordination increase bot probability
            combined_prob = (
                Config.TEMPORAL_SIMILARITY_WEIGHT * temporal_score +
                Config.SEMANTIC_SIMILARITY_WEIGHT * semantic_prob
            )
            
            final_bot_probs[comment_id] = min(max(combined_prob, 0.0), 1.0)
        
        self.comment_bot_probs = final_bot_probs
        logger.info(f"Calculated bot probabilities for {len(final_bot_probs)} comments")
        
        return final_bot_probs
    
    def calculate_account_bot_probabilities(self, comments_df: pd.DataFrame,
                                             cluster_assignments: Dict[str, int]) -> Dict[str, float]:
        """
        Calculate bot probability for each account.
        
        Based on:
        - Average bot probability of their comments
        - Cluster membership (accounts in same cluster = coordinated)
        - Coordination level with other accounts
        
        Args:
            comments_df: DataFrame with comments
            cluster_assignments: Dictionary mapping author_id to cluster_id
            
        Returns:
            Dictionary mapping author_id to bot probability
        """
        logger.info("Calculating per-account bot probabilities...")
        
        if not self.comment_bot_probs:
            raise ValueError("Must call calculate_comment_bot_probabilities first")
        
        account_probs = {}
        
        # Calculate cluster-level statistics
        cluster_sizes = {}
        for author, cluster_id in cluster_assignments.items():
            if cluster_id != -1:  # Exclude noise
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += 1
        
        for author_id, group in comments_df.groupby('author_id'):
            # Average comment bot probability
            comment_probs = [
                self.comment_bot_probs.get(cid, 0.0) 
                for cid in group['comment_id']
            ]
            avg_comment_prob = np.mean(comment_probs) if comment_probs else 0.0
            
            # Cluster-based probability boost
            cluster_id = cluster_assignments.get(author_id, -1)
            if cluster_id != -1:
                cluster_size = cluster_sizes.get(cluster_id, 1)
                # Larger clusters = more suspicious
                # Normalize by max expected cluster size (e.g., 50)
                cluster_boost = min(cluster_size / 50.0, 0.5)
            else:
                cluster_boost = 0.0
            
            # Calculate coordination level with other accounts
            if self.coordination_matrix is not None:
                author_indices = group.index.tolist()
                if author_indices:
                    # Get coordination scores with other authors' comments
                    comment_indices = [
                        comments_df.index.get_loc(idx) 
                        for idx in author_indices 
                        if idx in comments_df.index
                    ]
                    
                    if comment_indices:
                        other_author_coords = []
                        author_ids_list = comments_df['author_id'].values
                        for idx in comment_indices:
                            for j in range(len(author_ids_list)):
                                if author_ids_list[j] != author_id:
                                    other_author_coords.append(self.coordination_matrix[idx, j])
                        
                        avg_coordination = np.mean(other_author_coords) if other_author_coords else 0.0
                    else:
                        avg_coordination = 0.0
                else:
                    avg_coordination = 0.0
            else:
                avg_coordination = 0.0
            
            # Final probability: weighted combination
            final_prob = (
                0.6 * avg_comment_prob +    # Comment-level signals
                0.2 * cluster_boost +        # Cluster membership
                0.2 * avg_coordination       # Overall coordination level
            )
            
            account_probs[author_id] = min(max(final_prob, 0.0), 1.0)
        
        self.account_bot_probs = account_probs
        logger.info(f"Calculated bot probabilities for {len(account_probs)} accounts")
        
        return account_probs
    
    def detect_bots(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main bot detection method implementing the full BotBuster algorithm.
        
        Args:
            comments_df: DataFrame with YouTube comments
            
        Returns:
            DataFrame with detection results per author
        """
        logger.info("=" * 50)
        logger.info("Starting BotBuster Detection Algorithm")
        logger.info("=" * 50)
        
        if comments_df.empty:
            logger.warning("No comments to analyze")
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_columns = ['comment_id', 'video_id', 'text', 'author_id', 'published_at']
        missing_columns = [col for col in required_columns if col not in comments_df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Reset index for consistent indexing
        comments_df = comments_df.reset_index(drop=True)
        
        # Step 1: Calculate coordination scores (temporal + semantic)
        logger.info("\nStep 1: Calculating coordination scores...")
        coordination_matrix = self.calculate_coordination_score(comments_df)
        
        # Step 2: Build coordination graph
        logger.info("\nStep 2: Building coordination graph...")
        coordination_graph = self.build_coordination_graph(
            comments_df, coordination_matrix, threshold=0.3
        )
        
        # Step 3: Detect botnet clusters
        logger.info("\nStep 3: Detecting botnet clusters...")
        cluster_assignments = self.detect_botnet_clusters(comments_df, coordination_graph)
        
        # Step 4: Calculate per-comment bot probabilities
        logger.info("\nStep 4: Calculating comment-level bot probabilities...")
        self.calculate_comment_bot_probabilities(comments_df)
        
        # Step 5: Calculate per-account bot probabilities
        logger.info("\nStep 5: Calculating account-level bot probabilities...")
        account_probs = self.calculate_account_bot_probabilities(comments_df, cluster_assignments)
        
        # Step 6: Compile results
        logger.info("\nStep 6: Compiling results...")
        results = []
        
        for author_id in comments_df['author_id'].unique():
            author_comments = comments_df[comments_df['author_id'] == author_id]
            
            result = {
                'author_id': author_id,
                'cluster_id': cluster_assignments.get(author_id, -1),
                'comment_count': len(author_comments),
                'comment_ids': list(author_comments['comment_id']),
                'avg_comment_bot_prob': np.mean([
                    self.comment_bot_probs.get(cid, 0.0) 
                    for cid in author_comments['comment_id']
                ]),
                'final_bot_probability': account_probs.get(author_id, 0.0),
            }
            
            # Classification
            prob = result['final_bot_probability']
            if prob >= Config.BOT_PROBABILITY_THRESHOLD:
                result['classification'] = 'likely_bot'
            elif prob >= Config.SUSPICIOUS_PROBABILITY_THRESHOLD:
                result['classification'] = 'suspicious'
            else:
                result['classification'] = 'likely_human'
            
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Sort by bot probability
        results_df = results_df.sort_values('final_bot_probability', ascending=False)
        
        # Log summary
        bot_count = (results_df['classification'] == 'likely_bot').sum()
        suspicious_count = (results_df['classification'] == 'suspicious').sum()
        human_count = (results_df['classification'] == 'likely_human').sum()
        
        logger.info("=" * 50)
        logger.info("BotBuster Detection Complete")
        logger.info(f"Total accounts: {len(results_df)}")
        logger.info(f"Likely bots: {bot_count}")
        logger.info(f"Suspicious: {suspicious_count}")
        logger.info(f"Likely humans: {human_count}")
        logger.info(f"Average bot probability: {results_df['final_bot_probability'].mean():.2%}")
        logger.info("=" * 50)
        
        return results_df
    
    def get_comment_probabilities(self) -> Dict[str, float]:
        """Get the per-comment bot probabilities (after detection is run)"""
        return self.comment_bot_probs
    
    def get_botnet_graph(self, comments_df: pd.DataFrame,
                          results_df: pd.DataFrame) -> nx.Graph:
        """
        Get a NetworkX graph representing the botnet structure.
        
        Nodes are accounts with bot probability as attribute.
        Edges represent coordination with weight as attribute.
        
        Args:
            comments_df: Original comments DataFrame
            results_df: Detection results DataFrame
            
        Returns:
            NetworkX graph
        """
        if self.coordination_matrix is None:
            raise ValueError("Must run detect_bots first")
        
        # Build the coordination graph
        G = self.build_coordination_graph(
            comments_df, self.coordination_matrix, threshold=0.2
        )
        
        # Add bot probabilities as node attributes
        bot_probs = results_df.set_index('author_id')['final_bot_probability'].to_dict()
        clusters = results_df.set_index('author_id')['cluster_id'].to_dict()
        
        for node in G.nodes():
            G.nodes[node]['bot_probability'] = bot_probs.get(node, 0.0)
            G.nodes[node]['cluster_id'] = clusters.get(node, -1)
        
        return G

