"""
Botnet Identifier - Uses full feature pipeline to identify and characterize botnets
"""
import logging
from typing import List, Dict, Set, Optional
from datetime import datetime
import pandas as pd
import numpy as np
from collections import defaultdict
import hashlib
import json

# Import your existing feature extractors
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from detection.clustering import ClusteringDetector

logger = logging.getLogger(__name__)


class BotnetIdentifier:
    """
    Identifies distinct botnets using the FULL feature extraction pipeline.
    """
    
    def __init__(self, db):
        self.db = db
        self.text_features = TextFeatures()
        self.detector = ClusteringDetector()
    
    def identify_botnets(self, accounts: List[str], 
                         connections: List[Dict],
                         min_size: int = 5) -> List[Dict]:
        """
        Identify distinct botnets from accounts using full ML pipeline.
        """
        logger.info(f"Identifying botnets from {len(accounts)} accounts")
        
        # Get ALL comments from these accounts
        all_comments = self.db.get_comments_by_author_ids(accounts)
        
        if all_comments is None or all_comments.empty:
            logger.warning("No comments found for accounts")
            return []
        
        # ========== USE FULL FEATURE PIPELINE ==========
        
        # 1. Extract ALL features using existing modules
        features_df = self._extract_all_features(all_comments)
        
        # 2. Run clustering detection
        detection_results = self.detector.detect_bots(features_df)
        
        # 3. Group accounts by cluster
        clusters = self._group_by_cluster(detection_results, min_size)
        
        logger.info(f"Found {len(clusters)} clusters meeting size threshold")
        
        # 4. Characterize each cluster as a botnet
        botnets = []
        for cluster_id, cluster_accounts in clusters.items():
            # Get cluster-specific comments
            cluster_comments = all_comments[all_comments['author_id'].isin(cluster_accounts)]
            cluster_features = features_df[features_df['author_id'].isin(cluster_accounts)]
            
            botnet = self._characterize_botnet_with_features(
                cluster_accounts, 
                cluster_comments,
                cluster_features,
                detection_results[detection_results['author_id'].isin(cluster_accounts)]
            )
            
            botnet['botnet_id'] = self._generate_botnet_id(cluster_accounts)
            botnet['cluster_id'] = cluster_id
            botnet['discovered_at'] = datetime.now().isoformat()
            
            botnets.append(botnet)
        
        # Sort by size
        botnets.sort(key=lambda x: x['size'], reverse=True)
        
        # Save to database
        for botnet in botnets:
            self._save_botnet(botnet)
        
        return botnets
    
    def _extract_all_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract ALL features using existing feature modules.
        This is the SAME pipeline used in integrated_api.py
        """
        logger.info(f"Extracting features for {len(comments_df)} comments...")
        
        # ===== TEMPORAL FEATURES =====
        burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
        temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
        temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
        
        # Synchronized posting detection
        sync_groups = TemporalFeatures.detect_synchronized_posting(comments_df)
        sync_authors = set()
        for group in sync_groups:
            sync_authors.update(group)
        temporal_df['in_sync_group'] = temporal_df['author_id'].isin(sync_authors).astype(int)
        
        # Posting regularity
        regularity_scores = TemporalFeatures.calculate_posting_regularity(comments_df)
        temporal_df['regularity_score'] = temporal_df['author_id'].map(regularity_scores)
        
        # ===== TEXT FEATURES =====
        template_scores = self.text_features.detect_template_comments(comments_df)
        spam_scores = self.text_features.detect_spam_patterns(comments_df)
        diversity_scores = self.text_features.calculate_comment_diversity(comments_df)
        linguistic_df = self.text_features.extract_linguistic_features(comments_df)
        
        linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
        linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
        linguistic_df['diversity_score'] = linguistic_df['author_id'].map(diversity_scores)
        
        # Duplicate detection
        duplicate_groups = self.text_features.find_duplicate_comments(comments_df)
        dup_comment_ids = set()
        for group in duplicate_groups:
            dup_comment_ids.update(group)
        # Map to authors
        dup_authors = comments_df[comments_df['comment_id'].isin(dup_comment_ids)]['author_id'].unique()
        linguistic_df['has_duplicates'] = linguistic_df['author_id'].isin(dup_authors).astype(int)
        
        # ===== NETWORK FEATURES =====
        network_df = NetworkFeatures.calculate_author_network_features(comments_df)
        
        # ===== BEHAVIORAL FEATURES =====
        behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
        
        # ===== MERGE ALL =====
        features_df = temporal_df
        for df in [linguistic_df, network_df, behavioral_df]:
            features_df = features_df.merge(
                df, on='author_id', how='outer', suffixes=('', '_dup')
            )
            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
        
        features_df = features_df.fillna(0)
        
        logger.info(f"Extracted {len(features_df.columns)} features for {len(features_df)} authors")
        
        return features_df
    
    def _group_by_cluster(self, detection_results: pd.DataFrame, 
                          min_size: int) -> Dict[int, Set[str]]:
        """Group accounts by their cluster assignment"""
        clusters = {}
        
        for cluster_id in detection_results['cluster_id'].unique():
            if cluster_id == -1:  # Skip noise
                continue
            
            cluster_accounts = set(
                detection_results[detection_results['cluster_id'] == cluster_id]['author_id']
            )
            
            if len(cluster_accounts) >= min_size:
                clusters[cluster_id] = cluster_accounts
        
        return clusters
    
    def _characterize_botnet_with_features(self, 
                                           accounts: Set[str],
                                           comments: pd.DataFrame,
                                           features: pd.DataFrame,
                                           detection_results: pd.DataFrame) -> Dict:
        """
        Build detailed botnet profile USING the extracted features.
        """
        accounts_list = list(accounts)
        
        botnet = {
            'accounts': accounts_list,
            'size': len(accounts),
            'metrics': {},
            'feature_profile': {},  # NEW: aggregated feature stats
            'detection_stats': {},  # NEW: from ClusteringDetector
            'targets': {},
            'temporal_patterns': {},
            'text_patterns': {},
            'network_patterns': {},  # NEW
            'behavioral_patterns': {},  # NEW
            'role_assignments': {}
        }
        
        # ===== DETECTION STATS (from ClusteringDetector) =====
        botnet['detection_stats'] = {
            'avg_bot_probability': float(detection_results['final_bot_probability'].mean()),
            'max_bot_probability': float(detection_results['final_bot_probability'].max()),
            'min_bot_probability': float(detection_results['final_bot_probability'].min()),
            'likely_bot_count': int((detection_results['classification'] == 'likely_bot').sum()),
            'suspicious_count': int((detection_results['classification'] == 'suspicious').sum()),
            'cluster_confidence': float(detection_results['cluster_confidence'].mean()) if 'cluster_confidence' in detection_results.columns else 0
        }
        
        # ===== FEATURE PROFILE (aggregated from full feature set) =====
        numeric_features = features.select_dtypes(include=[np.number]).columns.tolist()
        numeric_features = [f for f in numeric_features if f != 'author_id']
        
        feature_stats = {}
        for feat in numeric_features:
            if feat in features.columns:
                feature_stats[feat] = {
                    'mean': float(features[feat].mean()),
                    'std': float(features[feat].std()),
                    'max': float(features[feat].max())
                }
        botnet['feature_profile'] = feature_stats
        
        # ===== KEY INDICATORS (from features) =====
        
        # Temporal patterns
        if 'burst_score' in features.columns:
            botnet['temporal_patterns']['avg_burst_score'] = float(features['burst_score'].mean())
        if 'in_sync_group' in features.columns:
            botnet['temporal_patterns']['synchronized_accounts'] = int(features['in_sync_group'].sum())
        if 'regularity_score' in features.columns:
            botnet['temporal_patterns']['avg_regularity'] = float(features['regularity_score'].mean())
        if 'comments_per_hour' in features.columns:
            botnet['temporal_patterns']['avg_comments_per_hour'] = float(features['comments_per_hour'].mean())
        if 'hour_entropy' in features.columns:
            botnet['temporal_patterns']['avg_hour_entropy'] = float(features['hour_entropy'].mean())
        
        # Text patterns
        if 'template_score' in features.columns:
            botnet['text_patterns']['avg_template_score'] = float(features['template_score'].mean())
        if 'spam_score' in features.columns:
            botnet['text_patterns']['avg_spam_score'] = float(features['spam_score'].mean())
        if 'diversity_score' in features.columns:
            botnet['text_patterns']['avg_diversity'] = float(features['diversity_score'].mean())
        if 'has_duplicates' in features.columns:
            botnet['text_patterns']['accounts_with_duplicates'] = int(features['has_duplicates'].sum())
        if 'vocabulary_richness' in features.columns:
            botnet['text_patterns']['avg_vocab_richness'] = float(features['vocabulary_richness'].mean())
        if 'avg_sentiment_compound' in features.columns:
            botnet['text_patterns']['avg_sentiment'] = float(features['avg_sentiment_compound'].mean())
        
        # Network patterns
        if 'co_degree_centrality' in features.columns:
            botnet['network_patterns']['avg_degree_centrality'] = float(features['co_degree_centrality'].mean())
        if 'community_size' in features.columns:
            botnet['network_patterns']['avg_community_size'] = float(features['community_size'].mean())
        if 'in_clique' in features.columns:
            botnet['network_patterns']['accounts_in_cliques'] = int(features['in_clique'].sum())
        if 'is_star_center' in features.columns:
            botnet['network_patterns']['star_centers'] = int(features['is_star_center'].sum())
        if 'co_pagerank' in features.columns:
            botnet['network_patterns']['avg_pagerank'] = float(features['co_pagerank'].mean())
        
        # Behavioral patterns
        if 'automation_score' in features.columns:
            botnet['behavioral_patterns']['avg_automation_score'] = float(features['automation_score'].mean())
        if 'username_pattern_score' in features.columns:
            botnet['behavioral_patterns']['avg_username_pattern'] = float(features['username_pattern_score'].mean())
        if 'account_age_score' in features.columns:
            botnet['behavioral_patterns']['avg_account_age_score'] = float(features['account_age_score'].mean())
        if 'targeting_score' in features.columns:
            botnet['behavioral_patterns']['avg_targeting_score'] = float(features['targeting_score'].mean())
        
        # ===== BASIC METRICS =====
        botnet['metrics'] = {
            'total_comments': len(comments),
            'avg_comments_per_account': len(comments) / len(accounts) if accounts else 0,
            'unique_videos_targeted': comments['video_id'].nunique(),
            'unique_channels_targeted': comments['channel_title'].nunique() if 'channel_title' in comments.columns else 0,
            'date_range': {
                'first': str(comments['published_at'].min()) if len(comments) > 0 else None,
                'last': str(comments['published_at'].max()) if len(comments) > 0 else None
            }
        }
        
        # ===== TARGETS =====
        if 'channel_title' in comments.columns:
            channel_counts = comments['channel_title'].value_counts().head(10).to_dict()
            botnet['targets']['top_channels'] = [
                {'channel': k, 'comments': int(v)} for k, v in channel_counts.items()
            ]
        
        video_counts = comments['video_id'].value_counts().head(10).to_dict()
        botnet['targets']['top_videos'] = [
            {'video_id': k, 'comments': int(v)} for k, v in video_counts.items()
        ]
        
        # ===== ROLE ASSIGNMENTS (based on network features) =====
        botnet['role_assignments'] = self._assign_roles_from_features(features)
        
        # ===== CONFIDENCE SCORE =====
        botnet['confidence'] = self._calculate_confidence_from_features(botnet)
        
        # ===== AUTO NAME =====
        botnet['auto_name'] = self._generate_name(botnet)
        
        return botnet
    
    def _assign_roles_from_features(self, features: pd.DataFrame) -> Dict:
        """Assign roles based on network features"""
        roles = {}
        
        for _, row in features.iterrows():
            author_id = row['author_id']
            
            # Use actual network metrics
            centrality = row.get('co_degree_centrality', 0)
            is_star = row.get('is_star_center', False)
            pagerank = row.get('co_pagerank', 0)
            
            if is_star or centrality > 0.3:
                role = 'hub'
            elif centrality > 0.15 or pagerank > 0.01:
                role = 'coordinator'
            elif centrality > 0.05:
                role = 'active_member'
            else:
                role = 'peripheral'
            
            roles[author_id] = {
                'role': role,
                'centrality': float(centrality),
                'pagerank': float(pagerank)
            }
        
        # Count by role
        role_counts = defaultdict(int)
        for r in roles.values():
            role_counts[r['role']] += 1
        
        return {
            'assignments': roles,
            'summary': dict(role_counts)
        }
    
    def _calculate_confidence_from_features(self, botnet: Dict) -> float:
        """Calculate confidence using actual feature values"""
        score = 0.3  # Base
        
        detection = botnet.get('detection_stats', {})
        temporal = botnet.get('temporal_patterns', {})
        text = botnet.get('text_patterns', {})
        network = botnet.get('network_patterns', {})
        behavioral = botnet.get('behavioral_patterns', {})
        
        # High bot probability from detector
        if detection.get('avg_bot_probability', 0) > 0.7:
            score += 0.15
        
        # Synchronized behavior
        if temporal.get('synchronized_accounts', 0) > 3:
            score += 0.1
        
        # High burst scores
        if temporal.get('avg_burst_score', 0) > 0.5:
            score += 0.1
        
        # Template usage
        if text.get('avg_template_score', 0) > 0.3:
            score += 0.1
        
        # Low diversity
        if text.get('avg_diversity', 1) < 0.5:
            score += 0.05
        
        # Clique membership
        if network.get('accounts_in_cliques', 0) > 2:
            score += 0.1
        
        # High automation
        if behavioral.get('avg_automation_score', 0) > 0.5:
            score += 0.1
        
        # Bot-like usernames
        if behavioral.get('avg_username_pattern', 0) > 0.3:
            score += 0.05
        
        return min(score, 0.95)
    
    def _generate_name(self, botnet: Dict) -> str:
        """Generate descriptive name"""
        parts = []
        
        # Size
        size = botnet['size']
        if size > 100:
            parts.append('Large')
        elif size > 30:
            parts.append('Medium')
        else:
            parts.append('Small')
        
        # Primary characteristic
        temporal = botnet.get('temporal_patterns', {})
        text = botnet.get('text_patterns', {})
        
        if temporal.get('synchronized_accounts', 0) > 5:
            parts.append('Synchronized')
        elif text.get('avg_template_score', 0) > 0.5:
            parts.append('Template')
        elif temporal.get('avg_burst_score', 0) > 0.5:
            parts.append('Burst')
        
        # Target
        targets = botnet.get('targets', {})
        top_channels = targets.get('top_channels', [])
        if top_channels:
            channel = top_channels[0]['channel'][:15]
            parts.append(f"targeting-{channel}")
        
        # Type
        if text.get('avg_spam_score', 0) > 0.3:
            parts.append('spam-network')
        else:
            parts.append('engagement-network')
        
        return '-'.join(parts)
    
    def _generate_botnet_id(self, accounts: Set[str]) -> str:
        sorted_accounts = sorted(accounts)
        hash_input = '|'.join(sorted_accounts[:10])
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]
    
    def _save_botnet(self, botnet: Dict):
        try:
            self.db.save_botnet(botnet)
        except Exception as e:
            logger.warning(f"Could not save botnet: {e}")
    
    def get_botnet_by_id(self, botnet_id: str) -> Optional[Dict]:
        return self.db.get_botnet(botnet_id)
    
    def list_botnets(self) -> List[Dict]:
        return self.db.list_botnets()