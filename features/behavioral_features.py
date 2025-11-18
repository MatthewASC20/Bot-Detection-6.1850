"""
Behavioral feature extraction for detecting bot-like account behaviors
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
import re
from collections import Counter
import Levenshtein

from config.config import Config

logger = logging.getLogger(__name__)

class BehavioralFeatures:
    """Extract behavioral patterns indicative of bot activity"""
    
    @staticmethod
    def analyze_account_age(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze account age patterns
        
        Args:
            comments_df: DataFrame with comments and channel creation dates
            
        Returns:
            Dictionary mapping author_id to age score (0-1, lower = newer/suspicious)
        """
        age_scores = {}
        
        for author_id, group in comments_df.groupby('author_id'):
            # Get channel creation date if available
            if 'author_channel_created' in group.columns and pd.notna(group['author_channel_created'].iloc[0]):
                created_date = pd.to_datetime(group['author_channel_created'].iloc[0])
                first_comment_date = pd.to_datetime(group['published_at'].min())
                
                # Calculate account age at time of first comment
                age_days = (first_comment_date - created_date).days
                
                if age_days < 0:  # Data inconsistency
                    age_scores[author_id] = 0.5
                elif age_days < Config.SUSPICIOUS_ACCOUNT_AGE_DAYS:
                    # Newer accounts are more suspicious
                    age_scores[author_id] = age_days / Config.SUSPICIOUS_ACCOUNT_AGE_DAYS
                else:
                    age_scores[author_id] = 1.0
            else:
                # No creation date available - neutral score
                age_scores[author_id] = 0.5
        
        return age_scores
    
    @staticmethod
    def analyze_username_patterns(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect patterns in usernames that suggest bot generation
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to username pattern score
        """
        username_scores = {}
        all_usernames = comments_df[['author_id', 'author']].drop_duplicates()
        
        # Common bot username patterns
        bot_patterns = [
            r'^user\d{5,}$',  # user12345678
            r'^[a-z]+\d{4,}$',  # john1234567
            r'^\w+_\d{4,}$',  # name_123456
            r'^[A-Z][a-z]+[A-Z][a-z]+\d{2,}$',  # FirstLast123
            r'^\w{8}-\w{4}-\w{4}-\w{4}-\w{12}$',  # UUID pattern
            r'^temp_\w+$',  # temp_something
            r'^test\w*\d+$',  # test123
        ]
        
        # Check for patterns
        for _, row in all_usernames.iterrows():
            username = row['author']
            author_id = row['author_id']
            
            # Check against bot patterns
            pattern_match = any(re.match(pattern, username, re.IGNORECASE) 
                              for pattern in bot_patterns)
            
            # Check for random-looking names (high entropy)
            entropy = BehavioralFeatures._calculate_string_entropy(username)
            
            # Check for similarity to other usernames
            similarities = []
            for _, other_row in all_usernames.iterrows():
                if other_row['author_id'] != author_id:
                    sim = Levenshtein.ratio(username, other_row['author'])
                    if sim > Config.USERNAME_PATTERN_THRESHOLD:
                        similarities.append(sim)
            
            # Combine scores
            pattern_score = 1.0 if pattern_match else 0.0
            entropy_score = min(entropy / 3.0, 1.0)  # Normalize entropy
            similarity_score = len(similarities) / max(len(all_usernames) - 1, 1)
            
            username_scores[author_id] = (pattern_score * 0.4 + 
                                         entropy_score * 0.3 + 
                                         similarity_score * 0.3)
        
        return username_scores
    
    @staticmethod
    def _calculate_string_entropy(s: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not s:
            return 0.0
        
        # Calculate frequency of each character
        char_counts = Counter(s)
        length = len(s)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    @staticmethod
    def analyze_activity_patterns(comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze account activity patterns
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with activity features per author
        """
        activity_features = []
        
        for author_id, group in comments_df.groupby('author_id'):
            features = {'author_id': author_id}
            
            # Get channel statistics if available
            if 'author_subscriber_count' in group.columns:
                features['subscriber_count'] = group['author_subscriber_count'].iloc[0]
                features['video_count'] = group['author_video_count'].iloc[0] if 'author_video_count' in group.columns else 0
                features['total_views'] = group['author_total_views'].iloc[0] if 'author_total_views' in group.columns else 0
                
                # Calculate engagement ratios
                if features['video_count'] > 0:
                    features['views_per_video'] = features['total_views'] / features['video_count']
                else:
                    features['views_per_video'] = 0
                
                if features['total_views'] > 0:
                    features['subscriber_view_ratio'] = features['subscriber_count'] / features['total_views']
                else:
                    features['subscriber_view_ratio'] = 0
            else:
                features['subscriber_count'] = 0
                features['video_count'] = 0
                features['total_views'] = 0
                features['views_per_video'] = 0
                features['subscriber_view_ratio'] = 0
            
            # Comment patterns
            features['total_comments'] = len(group)
            features['unique_videos_commented'] = group['video_id'].nunique()
            features['comments_per_video'] = features['total_comments'] / max(features['unique_videos_commented'], 1)
            
            # Like patterns
            features['total_likes_received'] = group['like_count'].sum()
            features['avg_likes_per_comment'] = group['like_count'].mean()
            features['zero_like_ratio'] = (group['like_count'] == 0).mean()
            
            # Reply patterns
            features['reply_ratio'] = group['is_reply'].mean() if 'is_reply' in group.columns else 0
            
            activity_features.append(features)
        
        return pd.DataFrame(activity_features)
    
    @staticmethod
    def detect_automated_behavior(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Detect signs of automated posting behavior
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to automation score
        """
        automation_scores = {}
        
        for author_id, group in comments_df.groupby('author_id'):
            automation_indicators = []
            
            # Check for exact timestamp patterns (posting at exact seconds)
            timestamps = pd.to_datetime(group['published_at'])
            seconds = timestamps.dt.second
            # High concentration at :00 seconds suggests automation
            zero_second_ratio = (seconds == 0).mean()
            automation_indicators.append(zero_second_ratio)
            
            # Check for consistent intervals
            if len(timestamps) > 2:
                sorted_times = timestamps.sort_values()
                intervals = np.diff(sorted_times.values).astype('timedelta64[s]').astype(float)
                
                # Check for regular intervals (low standard deviation)
                if len(intervals) > 0 and np.mean(intervals) > 0:
                    interval_cv = np.std(intervals) / np.mean(intervals)
                    regularity_score = 1.0 / (1.0 + interval_cv)
                    automation_indicators.append(regularity_score)
            
            # Check for rapid consecutive posting
            rapid_posting = BehavioralFeatures._detect_rapid_posting(timestamps)
            automation_indicators.append(rapid_posting)
            
            # Check for inhuman posting hours (e.g., consistent posting 24/7)
            hour_coverage = len(timestamps.dt.hour.unique()) / 24.0
            automation_indicators.append(hour_coverage)
            
            # Average all indicators
            automation_scores[author_id] = np.mean(automation_indicators) if automation_indicators else 0.0
        
        return automation_scores
    
    @staticmethod
    def _detect_rapid_posting(timestamps: pd.Series) -> float:
        """Detect rapid consecutive posting"""
        if len(timestamps) < 2:
            return 0.0
        
        sorted_times = timestamps.sort_values()
        intervals = np.diff(sorted_times.values).astype('timedelta64[s]').astype(float)
        
        # Count posts within 60 seconds of each other
        rapid_posts = np.sum(intervals < 60)
        return rapid_posts / max(len(intervals), 1)
    
    @staticmethod
    def analyze_content_targeting(comments_df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze if accounts target specific types of content
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            Dictionary mapping author_id to targeting score
        """
        targeting_scores = {}
        
        for author_id, group in comments_df.groupby('author_id'):
            # Check channel diversity
            unique_channels = group['channel_title'].nunique() if 'channel_title' in group.columns else 1
            total_comments = len(group)
            
            # Low channel diversity suggests targeting
            channel_concentration = 1.0 / max(unique_channels, 1)
            
            # Check for keyword targeting (if available)
            if 'video_title' in group.columns:
                all_titles = ' '.join(group['video_title'].fillna(''))
                
                # Check for political keywords
                political_count = sum(1 for keyword in Config.POLITICAL_KEYWORDS 
                                    if keyword.lower() in all_titles.lower())
                political_concentration = political_count / max(len(Config.POLITICAL_KEYWORDS), 1)
            else:
                political_concentration = 0.0
            
            targeting_scores[author_id] = (channel_concentration * 0.6 + 
                                          political_concentration * 0.4)
        
        return targeting_scores
    
    @staticmethod
    def compile_behavioral_features(comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compile all behavioral features into a single DataFrame
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with all behavioral features per author
        """
        logger.info("Extracting behavioral features...")
        
        # Extract individual feature sets
        age_scores = BehavioralFeatures.analyze_account_age(comments_df)
        username_scores = BehavioralFeatures.analyze_username_patterns(comments_df)
        activity_df = BehavioralFeatures.analyze_activity_patterns(comments_df)
        automation_scores = BehavioralFeatures.detect_automated_behavior(comments_df)
        targeting_scores = BehavioralFeatures.analyze_content_targeting(comments_df)
        
        # Start with activity features as base
        features_df = activity_df.copy()
        
        # Add other scores
        features_df['account_age_score'] = features_df['author_id'].map(age_scores)
        features_df['username_pattern_score'] = features_df['author_id'].map(username_scores)
        features_df['automation_score'] = features_df['author_id'].map(automation_scores)
        features_df['targeting_score'] = features_df['author_id'].map(targeting_scores)
        
        # Fill NaN values
        features_df = features_df.fillna(0)
        
        logger.info(f"Compiled behavioral features for {len(features_df)} authors")
        
        return features_df
