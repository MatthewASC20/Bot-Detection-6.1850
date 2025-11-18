"""
Temporal feature extraction for detecting coordinated posting patterns
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set
import logging
from scipy import stats
from collections import defaultdict

from config.config import Config

logger = logging.getLogger(__name__)

class TemporalFeatures:
    """Extract temporal patterns indicative of bot behavior"""
    
    @staticmethod
    def extract_burst_patterns(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect burst posting patterns (multiple accounts posting in short time windows)
        
        Args:
            comments_df: DataFrame with comments and timestamps
            
        Returns:
            Dictionary mapping author_id to burst score
        """
        # Convert timestamps to datetime
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        comments_df = comments_df.sort_values('timestamp')
        
        burst_scores = {}
        author_groups = comments_df.groupby('author_id')
        
        for author_id, group in author_groups:
            if len(group) < 2:
                burst_scores[author_id] = 0.0
                continue
            
            # Calculate inter-comment intervals
            timestamps = group['timestamp'].values
            intervals = np.diff(timestamps).astype('timedelta64[s]').astype(float)  # Convert to seconds
            
            # Check for rapid posting
            rapid_posts = np.sum(intervals < Config.TIME_WINDOW_SECONDS)
            burst_score = rapid_posts / len(intervals) if len(intervals) > 0 else 0
            
            burst_scores[author_id] = burst_score
        
        return burst_scores
    
    @staticmethod
    def detect_synchronized_posting(comments_df: pd.DataFrame, 
                                  window_seconds: int = 300) -> List[Set[str]]:
        """
        Detect groups of accounts posting within the same time window
        
        Args:
            comments_df: DataFrame with comments
            window_seconds: Time window in seconds
            
        Returns:
            List of sets containing synchronized author IDs
        """
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        comments_df = comments_df.sort_values('timestamp')
        
        synchronized_groups = []
        processed_indices = set()
        
        for idx, row in comments_df.iterrows():
            if idx in processed_indices:
                continue
            
            current_time = row['timestamp']
            window_end = current_time + timedelta(seconds=window_seconds)
            
            # Find all comments within the window
            mask = (comments_df['timestamp'] >= current_time) & \
                  (comments_df['timestamp'] <= window_end)
            window_comments = comments_df[mask]
            
            # Get unique authors in this window
            authors_in_window = set(window_comments['author_id'].unique())
            
            # Consider it synchronized if 3+ different authors post in the window
            if len(authors_in_window) >= Config.MIN_CLUSTER_SIZE:
                synchronized_groups.append(authors_in_window)
                processed_indices.update(window_comments.index)
        
        # Merge overlapping groups
        merged_groups = TemporalFeatures._merge_overlapping_groups(synchronized_groups)
        
        return merged_groups
    
    @staticmethod
    def _merge_overlapping_groups(groups: List[Set[str]]) -> List[Set[str]]:
        """Merge groups that share common elements"""
        if not groups:
            return []
        
        merged = []
        for group in groups:
            # Check if this group overlaps with any existing merged group
            found_overlap = False
            for i, merged_group in enumerate(merged):
                if group & merged_group:  # If intersection is not empty
                    merged[i] = merged_group | group  # Union
                    found_overlap = True
                    break
            
            if not found_overlap:
                merged.append(group)
        
        return merged
    
    @staticmethod
    def calculate_posting_regularity(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate how regular/periodic an author's posting pattern is
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to regularity score (0-1, higher = more regular)
        """
        regularity_scores = {}
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        
        for author_id, group in comments_df.groupby('author_id'):
            if len(group) < 3:
                regularity_scores[author_id] = 0.0
                continue
            
            timestamps = group['timestamp'].sort_values()
            intervals = np.diff(timestamps.values).astype('timedelta64[s]').astype(float)
            
            if len(intervals) == 0:
                regularity_scores[author_id] = 0.0
                continue
            
            # Calculate coefficient of variation (lower = more regular)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
            
            if mean_interval > 0:
                cv = std_interval / mean_interval
                # Convert to 0-1 score (lower CV = higher regularity)
                regularity = 1.0 / (1.0 + cv)
            else:
                regularity = 0.0
            
            regularity_scores[author_id] = regularity
        
        return regularity_scores
    
    @staticmethod
    def extract_time_patterns(comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract various time-based features for each author
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with temporal features for each author
        """
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        
        temporal_features = []
        
        for author_id, group in comments_df.groupby('author_id'):
            features = {'author_id': author_id}
            
            timestamps = group['timestamp']
            
            # Basic statistics
            features['comment_count'] = len(group)
            features['first_comment'] = timestamps.min()
            features['last_comment'] = timestamps.max()
            features['active_period_hours'] = (timestamps.max() - timestamps.min()).total_seconds() / 3600
            
            # Posting rate
            if features['active_period_hours'] > 0:
                features['comments_per_hour'] = features['comment_count'] / features['active_period_hours']
            else:
                features['comments_per_hour'] = features['comment_count']
            
            # Time of day analysis
            hours = timestamps.dt.hour
            features['most_active_hour'] = hours.mode()[0] if len(hours.mode()) > 0 else 0
            features['hour_entropy'] = stats.entropy(hours.value_counts())
            
            # Day of week analysis
            days = timestamps.dt.dayofweek
            features['most_active_day'] = days.mode()[0] if len(days.mode()) > 0 else 0
            features['day_entropy'] = stats.entropy(days.value_counts())
            
            # Inter-comment intervals
            if len(timestamps) > 1:
                sorted_timestamps = timestamps.sort_values()
                intervals = np.diff(sorted_timestamps.values).astype('timedelta64[s]').astype(float)
                features['mean_interval_seconds'] = np.mean(intervals)
                features['std_interval_seconds'] = np.std(intervals)
                features['min_interval_seconds'] = np.min(intervals)
                features['max_interval_seconds'] = np.max(intervals)
            else:
                features['mean_interval_seconds'] = 0
                features['std_interval_seconds'] = 0
                features['min_interval_seconds'] = 0
                features['max_interval_seconds'] = 0
            
            temporal_features.append(features)
        
        return pd.DataFrame(temporal_features)
    
    @staticmethod
    def detect_campaign_waves(comments_df: pd.DataFrame, 
                            wave_threshold_hours: int = 24) -> List[Dict]:
        """
        Detect coordinated campaign waves (periods of increased activity)
        
        Args:
            comments_df: DataFrame with comments
            wave_threshold_hours: Hours to define a campaign wave
            
        Returns:
            List of dictionaries describing detected waves
        """
        comments_df['timestamp'] = pd.to_datetime(comments_df['published_at'])
        comments_df = comments_df.sort_values('timestamp')
        
        # Create hourly bins
        comments_df['hour_bin'] = comments_df['timestamp'].dt.floor('H')
        hourly_counts = comments_df.groupby('hour_bin').size()
        
        # Detect anomalies using z-score
        mean_count = hourly_counts.mean()
        std_count = hourly_counts.std()
        z_scores = (hourly_counts - mean_count) / std_count
        
        # Find significant spikes (z-score > 2)
        spikes = hourly_counts[z_scores > 2]
        
        waves = []
        for spike_time, count in spikes.items():
            wave_end = spike_time + timedelta(hours=wave_threshold_hours)
            wave_comments = comments_df[
                (comments_df['timestamp'] >= spike_time) & 
                (comments_df['timestamp'] <= wave_end)
            ]
            
            if len(wave_comments) >= Config.MIN_CLUSTER_SIZE:
                wave = {
                    'start_time': spike_time,
                    'end_time': wave_end,
                    'comment_count': len(wave_comments),
                    'unique_authors': wave_comments['author_id'].nunique(),
                    'z_score': z_scores[spike_time],
                    'participating_authors': list(wave_comments['author_id'].unique())
                }
                waves.append(wave)
        
        return waves
    
    @staticmethod
    def calculate_temporal_similarity(author1_comments: pd.DataFrame, 
                                    author2_comments: pd.DataFrame) -> float:
        """
        Calculate temporal similarity between two authors' posting patterns
        
        Args:
            author1_comments: Comments from first author
            author2_comments: Comments from second author
            
        Returns:
            Similarity score (0-1)
        """
        if len(author1_comments) == 0 or len(author2_comments) == 0:
            return 0.0
        
        # Convert to timestamps
        times1 = pd.to_datetime(author1_comments['published_at']).values
        times2 = pd.to_datetime(author2_comments['published_at']).values
        
        # Find overlapping time period
        min_time = max(times1.min(), times2.min())
        max_time = min(times1.max(), times2.max())
        
        if min_time >= max_time:
            return 0.0
        
        # Create hourly bins for the overlapping period
        hours = pd.date_range(start=min_time, end=max_time, freq='H')
        
        # Count comments in each bin
        hist1, _ = np.histogram(times1, bins=hours)
        hist2, _ = np.histogram(times2, bins=hours)
        
        # Calculate correlation
        if len(hist1) > 1 and len(hist2) > 1:
            correlation = np.corrcoef(hist1, hist2)[0, 1]
            # Convert to 0-1 scale
            similarity = (correlation + 1) / 2
        else:
            similarity = 0.0
        
        return max(0, min(1, similarity))
