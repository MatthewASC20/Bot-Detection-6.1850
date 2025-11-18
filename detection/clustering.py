"""
Unsupervised clustering algorithms for bot detection
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import hdbscan
from typing import Dict, List, Tuple, Optional
import logging

from config.config import Config

logger = logging.getLogger(__name__)

class ClusteringDetector:
    """Unsupervised clustering for bot detection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.hdbscan = None
        self.dbscan = None
        
    def prepare_features(self, features_df: pd.DataFrame, 
                        feature_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Prepare and scale features for clustering
        
        Args:
            features_df: DataFrame with features
            feature_columns: Specific columns to use (if None, use all numeric)
            
        Returns:
            Scaled feature array
        """
        if feature_columns is None:
            # Use all numeric columns except author_id
            feature_columns = [col for col in features_df.columns 
                             if col != 'author_id' and features_df[col].dtype in ['float64', 'int64']]
        
        features = features_df[feature_columns].values
        
        # Handle infinite values
        features = np.nan_to_num(features, nan=0, posinf=1, neginf=0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction
        if features_scaled.shape[1] > 10:
            features_scaled = self.pca.fit_transform(features_scaled)
            logger.info(f"Reduced features from {features.shape[1]} to {features_scaled.shape[1]} dimensions")
        
        return features_scaled
    
    def cluster_hdbscan(self, features: np.ndarray, 
                        min_cluster_size: Optional[int] = None) -> np.ndarray:
        """
        Perform HDBSCAN clustering
        
        Args:
            features: Feature array
            min_cluster_size: Minimum cluster size
            
        Returns:
            Cluster labels (-1 for noise/outliers)
        """
        if min_cluster_size is None:
            min_cluster_size = Config.MIN_CLUSTER_SIZE
        
        self.hdbscan = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=Config.MIN_SAMPLES,
            cluster_selection_epsilon=Config.CLUSTER_SELECTION_EPSILON,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        labels = self.hdbscan.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return labels
    
    def cluster_dbscan(self, features: np.ndarray, 
                      eps: float = 0.5, 
                      min_samples: Optional[int] = None) -> np.ndarray:
        """
        Perform DBSCAN clustering
        
        Args:
            features: Feature array
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Cluster labels
        """
        if min_samples is None:
            min_samples = Config.MIN_SAMPLES
        
        self.dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = self.dbscan.fit_predict(features)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
        
        return labels
    
    def identify_bot_clusters(self, features_df: pd.DataFrame, 
                            labels: np.ndarray,
                            threshold_features: List[str] = None) -> Dict[int, float]:
        """
        Identify which clusters are likely bots based on feature analysis
        
        Args:
            features_df: DataFrame with features
            labels: Cluster labels
            threshold_features: Features to check for bot behavior
            
        Returns:
            Dictionary mapping cluster_id to bot probability
        """
        if threshold_features is None:
            threshold_features = [
                'automation_score', 'username_pattern_score', 
                'targeting_score', 'template_score'
            ]
        
        cluster_bot_scores = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_mask = labels == cluster_id
            cluster_data = features_df[cluster_mask]
            
            bot_indicators = []
            
            # Check available threshold features
            for feature in threshold_features:
                if feature in cluster_data.columns:
                    mean_score = cluster_data[feature].mean()
                    bot_indicators.append(mean_score)
            
            # Check behavioral patterns
            if 'comments_per_hour' in cluster_data.columns:
                high_rate_ratio = (cluster_data['comments_per_hour'] > 
                                 Config.SUSPICIOUS_COMMENT_RATE).mean()
                bot_indicators.append(high_rate_ratio)
            
            if 'account_age_score' in cluster_data.columns:
                young_accounts_ratio = (cluster_data['account_age_score'] < 0.3).mean()
                bot_indicators.append(young_accounts_ratio)
            
            # Average bot probability for cluster
            cluster_bot_scores[cluster_id] = np.mean(bot_indicators) if bot_indicators else 0.0
        
        return cluster_bot_scores
    
    def ensemble_clustering(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine multiple clustering methods
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (consensus_labels, confidence_scores)
        """
        clusterings = []
        
        # HDBSCAN with different parameters
        for min_size in [3, 5, 10]:
            try:
                labels = self.cluster_hdbscan(features, min_cluster_size=min_size)
                clusterings.append(labels)
            except:
                logger.warning(f"HDBSCAN with min_size={min_size} failed")
        
        # DBSCAN with different eps values
        for eps in [0.3, 0.5, 0.7]:
            try:
                labels = self.cluster_dbscan(features, eps=eps)
                clusterings.append(labels)
            except:
                logger.warning(f"DBSCAN with eps={eps} failed")
        
        if not clusterings:
            return np.zeros(len(features)), np.zeros(len(features))
        
        # Create consensus clustering
        consensus_matrix = np.zeros((len(features), len(features)))
        
        for labels in clusterings:
            for i in range(len(features)):
                for j in range(i+1, len(features)):
                    if labels[i] == labels[j] and labels[i] != -1:
                        consensus_matrix[i][j] += 1
                        consensus_matrix[j][i] += 1
        
        # Normalize by number of clusterings
        consensus_matrix /= len(clusterings)
        
        # Final clustering on consensus matrix
        final_clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        distance_matrix = 1 - consensus_matrix
        consensus_labels = final_clustering.fit_predict(distance_matrix)
        
        # Calculate confidence scores
        confidence_scores = np.zeros(len(features))
        for i in range(len(features)):
            if consensus_labels[i] != -1:
                cluster_mask = consensus_labels == consensus_labels[i]
                confidence_scores[i] = consensus_matrix[i][cluster_mask].mean()
        
        return consensus_labels, confidence_scores
    
    def detect_bots(self, all_features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to detect bots using clustering
        
        Args:
            all_features_df: DataFrame with all extracted features
            
        Returns:
            DataFrame with bot detection results
        """
        logger.info("Starting unsupervised bot detection...")
        
        # Prepare features
        features = self.prepare_features(all_features_df)
        
        # Perform ensemble clustering
        labels, confidence_scores = self.ensemble_clustering(features)
        
        # Identify bot clusters
        cluster_bot_scores = self.identify_bot_clusters(all_features_df, labels)
        
        # Create results dataframe
        results_df = all_features_df[['author_id']].copy()
        results_df['cluster_id'] = labels
        results_df['cluster_confidence'] = confidence_scores
        
        # Map cluster bot scores to individual accounts
        results_df['cluster_bot_probability'] = results_df['cluster_id'].map(
            lambda x: cluster_bot_scores.get(x, 0.0)
        )
        
        # Calculate individual bot scores based on features
        individual_scores = self._calculate_individual_bot_scores(all_features_df)
        results_df['individual_bot_probability'] = results_df['author_id'].map(individual_scores)
        
        # Combine cluster and individual scores
        results_df['final_bot_probability'] = (
            results_df['cluster_bot_probability'] * 0.6 +
            results_df['individual_bot_probability'] * 0.4
        )
        
        # Classify based on thresholds
        results_df['classification'] = pd.cut(
            results_df['final_bot_probability'],
            bins=[0, Config.SUSPICIOUS_PROBABILITY_THRESHOLD, 
                  Config.BOT_PROBABILITY_THRESHOLD, 1.0],
            labels=['likely_human', 'suspicious', 'likely_bot']
        )
        
        # Log summary
        bot_count = (results_df['classification'] == 'likely_bot').sum()
        suspicious_count = (results_df['classification'] == 'suspicious').sum()
        human_count = (results_df['classification'] == 'likely_human').sum()
        
        logger.info(f"Detection complete: {bot_count} bots, {suspicious_count} suspicious, {human_count} humans")
        
        return results_df
    
    def _calculate_individual_bot_scores(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate individual bot probability scores
        
        Args:
            features_df: DataFrame with features
            
        Returns:
            Dictionary mapping author_id to bot probability
        """
        scores = {}
        
        for _, row in features_df.iterrows():
            author_id = row['author_id']
            
            weighted_features = []
            weights = []
            
            # Map features to weights from config
            feature_weight_map = {
                'burst_score': Config.FEATURE_WEIGHTS.get('temporal_burst', 0.25),
                'template_score': Config.FEATURE_WEIGHTS.get('text_similarity', 0.20),
                'co_degree_centrality': Config.FEATURE_WEIGHTS.get('network_connectivity', 0.20),
                'account_age_score': Config.FEATURE_WEIGHTS.get('account_age', 0.15),
                'username_pattern_score': Config.FEATURE_WEIGHTS.get('username_pattern', 0.10),
                'comments_per_hour': Config.FEATURE_WEIGHTS.get('comment_rate', 0.10)
            }
            
            for feature, weight in feature_weight_map.items():
                if feature in row.index:
                    value = row[feature]
                    if feature == 'account_age_score':
                        # Invert age score (lower age = higher bot probability)
                        value = 1 - value
                    elif feature == 'comments_per_hour':
                        # Normalize comment rate
                        value = min(value / Config.SUSPICIOUS_COMMENT_RATE, 1.0)
                    
                    if not pd.isna(value):
                        weighted_features.append(value)
                        weights.append(weight)
            
            if weighted_features:
                # Normalize weights
                weights = np.array(weights) / sum(weights)
                scores[author_id] = np.average(weighted_features, weights=weights)
            else:
                scores[author_id] = 0.0
        
        return scores
