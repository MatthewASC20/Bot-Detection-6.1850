"""
Botnet Discovery - Main orchestrator for the discovery pipeline
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime
import pandas as pd

from discovery.expansion_crawler import ExpansionCrawler
from discovery.botnet_identifier import BotnetIdentifier
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from detection.clustering import ClusteringDetector

logger = logging.getLogger(__name__)


class BotnetDiscovery:
    """
    Main orchestrator for botnet discovery.
    
    Pipeline:
    1. Seed collection (from high-probability sources)
    2. Initial detection (clustering on seed data)
    3. Expansion (follow suspicious accounts)
    4. Re-clustering (on expanded data)
    5. Botnet identification (group into distinct botnets)
    6. Characterization (profile each botnet)
    """
    
    def __init__(self, db, youtube_api=None):
        self.db = db
        self.youtube_api = youtube_api
        self.crawler = ExpansionCrawler(db, youtube_api)
        self.identifier = BotnetIdentifier(db)
        self.detector = ClusteringDetector()
        self.text_features = TextFeatures()
        
    def run_full_discovery(self, 
                           seed_video_ids: List[str] = None,
                           seed_author_ids: List[str] = None,
                           max_expansion_depth: int = 3,
                           min_bot_probability: float = 0.6,
                           min_botnet_size: int = 5) -> Dict:
        """
        Run the complete botnet discovery pipeline.
        
        Args:
            seed_video_ids: Videos to start from (will analyze comments)
            seed_author_ids: Specific authors to investigate
            max_expansion_depth: How deep to follow connections
            min_bot_probability: Threshold for suspicious accounts
            min_botnet_size: Minimum accounts to form a botnet
            
        Returns:
            Complete discovery results
        """
        logger.info("=" * 60)
        logger.info("STARTING BOTNET DISCOVERY PIPELINE")
        logger.info("=" * 60)
        
        results = {
            'started_at': datetime.now().isoformat(),
            'parameters': {
                'seed_video_ids': seed_video_ids,
                'seed_author_ids': seed_author_ids,
                'max_expansion_depth': max_expansion_depth,
                'min_bot_probability': min_bot_probability,
                'min_botnet_size': min_botnet_size
            },
            'phases': {}
        }
        
        # Phase 1: Collect seed accounts
        logger.info("PHASE 1: Seed Collection")
        seed_accounts = self._collect_seeds(seed_video_ids, seed_author_ids, min_bot_probability)
        results['phases']['seed_collection'] = {
            'seed_accounts_found': len(seed_accounts),
            'accounts': seed_accounts[:100]  # Limit for response size
        }
        
        if len(seed_accounts) < 3:
            results['error'] = 'Not enough seed accounts found'
            results['completed_at'] = datetime.now().isoformat()
            return results
        
        # Phase 2: Expansion
        logger.info("PHASE 2: Network Expansion")
        expansion_result = self.crawler.expand_from_seeds(
            seed_accounts,
            max_depth=max_expansion_depth,
            min_bot_probability=min_bot_probability
        )
        results['phases']['expansion'] = {
            'total_accounts': expansion_result['total_accounts_discovered'],
            'total_videos': expansion_result['total_videos_involved'],
            'expansion_history': expansion_result['expansion_history']
        }
        
        # Phase 3: Re-analyze expanded network
        logger.info("PHASE 3: Full Network Analysis")
        all_accounts = expansion_result['accounts']
        all_comments = self.db.get_comments_by_author_ids(all_accounts)
        
        if all_comments is not None and not all_comments.empty:
            features_df = self._extract_features(all_comments)
            detection_results = self.detector.detect_bots(features_df)
            
            results['phases']['analysis'] = {
                'total_comments_analyzed': len(all_comments),
                'unique_authors': len(features_df),
                'detection_summary': detection_results['classification'].value_counts().to_dict() if 'classification' in detection_results.columns else {}
            }
        
        # Phase 4: Identify distinct botnets
        logger.info("PHASE 4: Botnet Identification")
        botnets = self.identifier.identify_botnets(
            expansion_result['accounts'],
            expansion_result['connections'],
            min_size=min_botnet_size
        )
        
        results['phases']['identification'] = {
            'botnets_found': len(botnets),
            'botnets': botnets
        }
        
        # Phase 5: Generate summary
        logger.info("PHASE 5: Generating Summary")
        results['summary'] = self._generate_summary(results, botnets)
        results['completed_at'] = datetime.now().isoformat()
        
        # Save discovery run
        self.db.save_discovery_run(results)
        
        logger.info("=" * 60)
        logger.info(f"DISCOVERY COMPLETE: Found {len(botnets)} botnets")
        logger.info("=" * 60)
        
        return results
    
    def _collect_seeds(self, video_ids: List[str], 
                       author_ids: List[str],
                       min_probability: float) -> List[str]:
        """Collect initial seed accounts"""
        seeds = set()
        
        # From specified authors
        if author_ids:
            seeds.update(author_ids)
        
        # From specified videos
        if video_ids:
            for video_id in video_ids:
                comments = self.db.get_comments_by_video(video_id)
                if comments is not None and not comments.empty:
                    # Get high-probability accounts
                    for author_id in comments['author_id'].unique():
                        profile = self.db.get_author_profile(author_id)
                        if profile and profile.get('avg_bot_score', 0) >= min_probability:
                            seeds.add(author_id)
        
        # From database (accounts with high bot scores)
        if len(seeds) < 10:
            high_prob_accounts = self.db.get_high_probability_authors(min_probability, limit=50)
            seeds.update(high_prob_accounts)
        
        return list(seeds)
    
    def _extract_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from comments"""
        try:
            # Temporal
            burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
            temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
            temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
            
            # Text
            template_scores = self.text_features.detect_template_comments(comments_df)
            spam_scores = self.text_features.detect_spam_patterns(comments_df)
            linguistic_df = self.text_features.extract_linguistic_features(comments_df)
            linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
            linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
            
            # Network
            network_df = NetworkFeatures.calculate_author_network_features(comments_df)
            
            # Behavioral
            behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
            
            # Merge
            features_df = temporal_df
            for df in [linguistic_df, network_df, behavioral_df]:
                features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
                features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
            
            return features_df.fillna(0)
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return pd.DataFrame()
    
    def _generate_summary(self, results: Dict, botnets: List[Dict]) -> Dict:
        """Generate human-readable summary"""
        total_accounts = results['phases']['expansion']['total_accounts']
        total_videos = results['phases']['expansion']['total_videos']
        
        summary = {
            'headline': f"Discovered {len(botnets)} botnet(s) with {total_accounts} suspicious accounts targeting {total_videos} videos",
            'botnets': []
        }
        
        for botnet in botnets:
            summary['botnets'].append({
                'id': botnet['botnet_id'],
                'name': botnet.get('auto_name', 'Unknown'),
                'size': botnet['size'],
                'confidence': botnet.get('confidence', 0),
                'top_target': botnet.get('targets', {}).get('top_channels', [{}])[0].get('channel', 'Unknown') if botnet.get('targets', {}).get('top_channels') else 'Unknown'
            })
        
        return summary
    
    def investigate_author(self, author_id: str) -> Dict:
        """Deep investigation of a single author"""
        logger.info(f"Investigating author: {author_id}")
        
        result = {
            'author_id': author_id,
            'profile': self.db.get_author_profile(author_id),
            'history': self.crawler.crawl_author_history(author_id),
            'associated_botnets': [],
            'risk_assessment': {}
        }
        
        # Check if part of known botnet
        botnets = self.identifier.list_botnets()
        for botnet in botnets:
            if author_id in botnet.get('accounts', []):
                result['associated_botnets'].append({
                    'botnet_id': botnet['botnet_id'],
                    'name': botnet.get('auto_name'),
                    'role': botnet.get('role_assignments', {}).get('assignments', {}).get(author_id, {}).get('role', 'unknown')
                })
        
        # Risk assessment
        profile = result['profile'] or {}
        history = result['history'] or {}
        
        risk_factors = []
        risk_score = 0
        
        if profile.get('avg_bot_score', 0) > 0.7:
            risk_factors.append('High bot probability score')
            risk_score += 30
        
        if len(result['associated_botnets']) > 0:
            risk_factors.append(f"Member of {len(result['associated_botnets'])} known botnet(s)")
            risk_score += 40
        
        if history.get('total_comments', 0) > 100:
            risk_factors.append('High comment volume')
            risk_score += 10
        
        top_co = history.get('top_co_commenters', [])
        if top_co and top_co[0].get('shared_videos', 0) > 10:
            risk_factors.append('Frequently appears with same accounts')
            risk_score += 20
        
        result['risk_assessment'] = {
            'score': min(risk_score, 100),
            'level': 'HIGH' if risk_score > 60 else 'MEDIUM' if risk_score > 30 else 'LOW',
            'factors': risk_factors
        }
        
        return result