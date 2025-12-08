#!/usr/bin/env python3
"""
YouTube Botnet Detector - Main Execution Script
Educational tool for detecting coordinated bot networks in YouTube comments

Implements BotBuster algorithm for detecting coordinated misinformation botnets.
Based on: "Detecting Coordinated Misinformation Botnets in Social Media"
"""

import argparse
import logging
import sys
import os
import json
import html
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import webbrowser

from pyparsing import col

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data_collection.data_collector import DataCollector
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from features.semantic_features import SemanticFeatures
from detection.clustering import ClusteringDetector
from detection.botbuster_detector import BotBusterDetector
from visualization.network_viz import NetworkVisualizer
from storage.database import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('botnet_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class YouTubeBotnetDetector:
    """Main orchestrator for bot detection pipeline"""
    
    def __init__(self, use_cache: bool = True, method: str = 'botbuster'):
        """
        Initialize the detector.
        
        Args:
            use_cache: Whether to use API response caching
            method: Detection method - 'botbuster' (new) or 'original' (legacy)
        """
        self.collector = DataCollector(use_cache=use_cache)
        self.text_features = TextFeatures()
        self.detector = ClusteringDetector()
        self.botbuster_detector = BotBusterDetector()
        self.visualizer = NetworkVisualizer()
        self.db = DatabaseHandler()
        self.method = method
        
        logger.info(f"Initialized detector with method: {method}")
        
    def collect_data(self, mode: str, urls: List[str] = None, 
                    max_comments: int = None) -> pd.DataFrame:
        """
        Collect data based on mode
        
        Args:
            mode: Collection mode ('urls', 'political', 'search')
            urls: List of video URLs (for 'urls' mode)
            max_comments: Maximum comments per video
            
        Returns:
            DataFrame with collected comments
        """
        logger.info(f"Starting data collection in mode: {mode}")
        
        if mode == 'urls' and urls:
            return self.collector.collect_from_urls(urls, max_comments)
        elif mode == 'political':
            return self.collector.collect_political_content(max_comments_per_video=max_comments)
        elif mode == 'search':
            # Search for political content
            queries = Config.POLITICAL_KEYWORDS[:5]  # Use first 5 keywords
            return self.collector.collect_by_search(queries, max_comments_per_video=max_comments)
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    """
    This is the CORRECT extract_all_features method for main.py
    It uses ALL 5 feature modules (79+ features total)

    Replace your current extract_all_features method with this one.
    """

    def extract_all_features(self, comments_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract all features from comments using ALL feature modules.
        
        This should produce 79+ features across:
        - Temporal (15 features): burst patterns, posting times, intervals
        - Text (22 features): linguistic, spam, templates, similarity
        - Network (17 features): co-occurrence, communities, centrality
        - Behavioral (16 features): account age, automation, targeting
        - Semantic (11 features): embeddings, topics, intents
        
        Args:
            comments_df: DataFrame with comments
            
        Returns:
            DataFrame with all features per author
        """
        logger.info("Extracting features using FULL pipeline...")
        
        # ============ TEMPORAL FEATURES ============
        logger.info("Extracting temporal features...")
        from features.temporal_features import TemporalFeatures
        
        burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
        temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
        temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
        
        # Add regularity scores
        regularity_scores = TemporalFeatures.calculate_posting_regularity(comments_df)
        temporal_df['regularity_score'] = temporal_df['author_id'].map(regularity_scores)
        
        logger.info(f"  → Temporal: {len(temporal_df.columns) - 1} features")
        
        # ============ TEXT FEATURES ============
        logger.info("Extracting text features...")
        from features.text_features import TextFeatures
        
        text_extractor = TextFeatures()
        template_scores = text_extractor.detect_template_comments(comments_df)
        spam_scores = text_extractor.detect_spam_patterns(comments_df)
        diversity_scores = text_extractor.calculate_comment_diversity(comments_df)
        linguistic_df = text_extractor.extract_linguistic_features(comments_df)
        
        linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
        linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
        linguistic_df['diversity_score'] = linguistic_df['author_id'].map(diversity_scores)
        
        logger.info(f"  → Text: {len(linguistic_df.columns) - 1} features")
        
        # ============ NETWORK FEATURES ============
        logger.info("Extracting network features...")
        from features.network_features import NetworkFeatures
        
        network_df = NetworkFeatures.calculate_author_network_features(comments_df)
        
        logger.info(f"  → Network: {len(network_df.columns) - 1} features")
        
        # ============ BEHAVIORAL FEATURES ============
        logger.info("Extracting behavioral features...")
        from features.behavioral_features import BehavioralFeatures
        
        behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
        
        logger.info(f"  → Behavioral: {len(behavioral_df.columns) - 1} features")
        
        # ============ SEMANTIC FEATURES ============
        logger.info("Extracting semantic features...")
        from features.semantic_features import SemanticFeatures
        
        semantic_extractor = SemanticFeatures(use_openai=True)  # Falls back to local if no API key
        semantic_df = semantic_extractor.extract_semantic_features(comments_df)
        
        logger.info(f"  → Semantic: {len(semantic_df.columns) - 1} features")
        
        # ============ MERGE ALL FEATURES ============
        logger.info("Merging all features...")
        
        features_df = temporal_df
        
        for df in [linguistic_df, network_df, behavioral_df, semantic_df]:
            features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
            # Remove duplicate columns
            features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        total_features = len(features_df.columns) - 1  # Exclude author_id
        logger.info(f"✅ Extracted {total_features} total features for {len(features_df)} authors")
        
        # Log feature summary
        logger.info("Feature categories:")
        logger.info(f"  - Temporal: burst_score, regularity_score, comments_per_hour, hour_entropy, etc.")
        logger.info(f"  - Text: template_score, spam_score, diversity_score, sentiment, etc.")
        logger.info(f"  - Network: co_degree_centrality, community_id, in_clique, etc.")
        logger.info(f"  - Behavioral: account_age_score, automation_score, targeting_score, etc.")
        logger.info(f"  - Semantic: avg_semantic_similarity, intent_bot_score, etc.")
        
        return features_df
    
    def detect_bots(self, features_df: pd.DataFrame, comments_df: pd.DataFrame,
                    labeled_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Run the configured detection method(s) and return primary results.
        
        When method='both', both pipelines are executed; BotBuster results are
        returned for downstream visualization while originals are retained for
        comparison/inspection.
        """
        logger.info("Running bot detection...")
        
        if self.method == 'botbuster':
            if comments_df is None or comments_df.empty:
                raise ValueError("comments_df is required for BotBuster detection")
            results = self.botbuster_detector.detect_bots(comments_df.reset_index(drop=True))
            self.botbuster_results = results
            return results
        
        if self.method == 'original':
            if features_df is None or features_df.empty:
                raise ValueError("features_df is required for the original detection method")
            results = self.detector.detect_bots(features_df)
            self.original_results = results
            return results
        
        if self.method == 'both':
            if features_df is None or features_df.empty:
                raise ValueError("features_df is required when running both detection methods")
            if comments_df is None or comments_df.empty:
                raise ValueError("comments_df is required when running both detection methods")
            
            logger.info("Running original clustering method...")
            original_results = self.detector.detect_bots(features_df)
            
            logger.info("Running BotBuster method...")
            botbuster_results = self.botbuster_detector.detect_bots(comments_df.reset_index(drop=True))
            
            self.original_results = original_results
            self.botbuster_results = botbuster_results
            self.comparison_results = self._compare_results(original_results, botbuster_results)
            
            logger.info("Completed both detection methods; using BotBuster results for visualization/summary")
            return botbuster_results
        
        raise ValueError(f"Unknown detection method: {self.method}")
    
    def _compare_results(self, original: pd.DataFrame, botbuster: pd.DataFrame) -> Dict:
        """Lightweight comparison between original and BotBuster results"""
        if original is None or botbuster is None:
            return {}
        
        merged = original.merge(
            botbuster,
            on='author_id',
            suffixes=('_original', '_botbuster'),
            how='outer'
        )
        
        comparison = {
            'original_method': {
                'total_accounts': len(original),
                'avg_bot_probability': float(original['final_bot_probability'].mean()),
                'likely_bots': int((original['classification'] == 'likely_bot').sum()),
                'suspicious': int((original['classification'] == 'suspicious').sum()),
                'likely_humans': int((original['classification'] == 'likely_human').sum()),
                'clusters_found': int(original['cluster_id'].nunique() - (1 if -1 in original['cluster_id'].values else 0))
            },
            'botbuster_method': {
                'total_accounts': len(botbuster),
                'avg_bot_probability': float(botbuster['final_bot_probability'].mean()),
                'likely_bots': int((botbuster['classification'] == 'likely_bot').sum()),
                'suspicious': int((botbuster['classification'] == 'suspicious').sum()),
                'likely_humans': int((botbuster['classification'] == 'likely_human').sum()),
                'clusters_found': int(botbuster['cluster_id'].nunique() - (1 if -1 in botbuster['cluster_id'].values else 0))
            }
        }
        
        common_authors = merged.dropna(subset=['final_bot_probability_original', 'final_bot_probability_botbuster'])
        if len(common_authors) > 1:
            comparison['probability_correlation'] = float(
                common_authors['final_bot_probability_original'].corr(
                    common_authors['final_bot_probability_botbuster']
                )
            )
        else:
            comparison['probability_correlation'] = None
        
        merged['classification_original'] = (
            merged['classification_original']
            .astype('string')
            .fillna('unknown')
        )
        merged['classification_botbuster'] = (
            merged['classification_botbuster']
            .astype('string')
            .fillna('unknown')
        )
        comparison['classification_agreement'] = float(
            (merged['classification_original'] == merged['classification_botbuster']).mean()
        )
        
        comparison['original_only_flags'] = merged[
            (merged['classification_original'].isin(['likely_bot', 'suspicious'])) &
            (merged['classification_botbuster'] == 'likely_human')
        ]['author_id'].dropna().tolist()
        
        comparison['botbuster_only_flags'] = merged[
            (merged['classification_botbuster'].isin(['likely_bot', 'suspicious'])) &
            (merged['classification_original'] == 'likely_human')
        ]['author_id'].dropna().tolist()
        
        return comparison
    
    def visualize_results(self, detection_results: pd.DataFrame, 
                         comments_df: pd.DataFrame) -> Dict[str, str]:
        """
        Create visualizations of detection results - paper-ready outputs
        
        Args:
            detection_results: DataFrame with detection results
            comments_df: Original comments DataFrame
            
        Returns:
            Dictionary with paths to visualization files
        """
        logger.info("Creating paper-ready visualizations...")
        
        visualization_files = {}
        
        # Build network graph
        G = NetworkFeatures.build_co_occurrence_network(comments_df)
        
        # Create bot scores dictionary
        bot_scores = detection_results.set_index('author_id')['final_bot_probability'].to_dict()
        
        # Get cluster assignments
        cluster_assignments = detection_results.set_index('author_id')['cluster_id'].to_dict()
        
        # Create network visualization with cluster labels
        if G.number_of_nodes() > 0:
            network_path = self.visualizer.visualize_bot_network(
                G, bot_scores, cluster_assignments, comments_df=comments_df,
                title="Bot Coordination Network - Cluster Membership",
                filename="bot_network.html"
            )
            visualization_files['network'] = network_path
        
        # Create comprehensive cluster analysis
        cluster_path = self.visualizer.visualize_cluster_analysis(
            detection_results,
            filename="cluster_analysis.html"
        )
        visualization_files['clusters'] = cluster_path
        
        # Create temporal patterns visualization
        temporal_path = self.visualizer.visualize_temporal_patterns(
            comments_df, detection_results,
            filename="temporal_patterns.html"
        )
        visualization_files['temporal'] = temporal_path
        
        # BotBuster-specific visualizations
        if self.method in ('botbuster', 'both'):
            # Get coordination matrix and cross-video scores from detector
            coordination_matrix = self.botbuster_detector.get_coordination_matrix()
            cross_video_scores = self.botbuster_detector.get_cross_video_scores()
            
            # Create coordination heatmap
            if coordination_matrix is not None:
                author_ids = comments_df.groupby('author_id').first().reset_index()['author_id'].tolist()
                # Aggregate to author-level for heatmap
                author_matrix = self._aggregate_to_author_matrix(
                    coordination_matrix, comments_df, author_ids
                )
                if author_matrix is not None:
                    heatmap_path = self.visualizer.visualize_coordination_heatmap(
                        author_matrix, author_ids, cluster_assignments,
                        filename="coordination_heatmap.html"
                    )
                    visualization_files['heatmap'] = heatmap_path
            
            # Create cross-video analysis
            if cross_video_scores:
                cross_video_path = self.visualizer.visualize_cross_video_analysis(
                    comments_df, detection_results, cross_video_scores,
                    filename="cross_video_analysis.html"
                )
                visualization_files['cross_video'] = cross_video_path
            
            # Create comprehensive summary figure
            summary_path = self.visualizer.create_summary_figure(
                detection_results, comments_df, cross_video_scores,
                filename="detection_summary.html"
            )
            visualization_files['summary'] = summary_path
        
        return visualization_files
    
    def _aggregate_to_author_matrix(self, comment_matrix: np.ndarray, 
                                    comments_df: pd.DataFrame,
                                    author_ids: List[str]) -> Optional[np.ndarray]:
        """Aggregate comment-level coordination matrix to author-level"""
        try:
            n_authors = len(author_ids)
            author_matrix = np.zeros((n_authors, n_authors))
            
            author_to_idx = {a: i for i, a in enumerate(author_ids)}
            comment_authors = comments_df['author_id'].values
            
            for i in range(len(comment_matrix)):
                for j in range(i + 1, len(comment_matrix)):
                    if i < len(comment_authors) and j < len(comment_authors):
                        a1, a2 = comment_authors[i], comment_authors[j]
                        if a1 in author_to_idx and a2 in author_to_idx and a1 != a2:
                            idx1, idx2 = author_to_idx[a1], author_to_idx[a2]
                            author_matrix[idx1, idx2] = max(author_matrix[idx1, idx2], 
                                                            comment_matrix[i, j])
                            author_matrix[idx2, idx1] = author_matrix[idx1, idx2]
            
            return author_matrix
        except Exception as e:
            logger.warning(f"Could not aggregate to author matrix: {e}")
            return None
    
    def save_results(self, detection_results: pd.DataFrame, 
                    summary: Dict,
                    timestamp: Optional[str] = None) -> str:
        """
        Save detection results and summary
        
        Args:
            detection_results: DataFrame with detection results
            summary: Summary statistics dictionary
            timestamp: Optional timestamp string to keep artifacts aligned
            
        Returns:
            Path to results file
        """
        # Prepare results for database (convert lists to JSON strings)
        db_results = detection_results.copy()
        if 'comment_ids' in db_results.columns:
            db_results['comment_ids'] = db_results['comment_ids'].apply(json.dumps)
        
        # Save to database
        self.db.save_detection_results(db_results)
        
        # Save to CSV
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(Config.REPORTS_DIR, f"bot_detection_results_{timestamp}.csv")
        detection_results.to_csv(csv_path, index=False)
        
        # Save summary JSON
        json_path = os.path.join(Config.REPORTS_DIR, f"detection_summary_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Track last saved artifacts for downstream references
        self.last_results_path = csv_path
        self.last_summary_path = json_path
        self.last_results_timestamp = timestamp
        
        logger.info(f"Results saved to {csv_path}")
        logger.info(f"Summary saved to {json_path}")
        
        return csv_path
    
    def run_full_pipeline(self, mode: str, urls: List[str] = None,
                         max_comments: int = None,
                         labeled_data_path: str = None) -> Dict:
        """
        Run the complete bot detection pipeline
        
        Args:
            mode: Data collection mode
            urls: Video URLs (for 'urls' mode)
            max_comments: Maximum comments per video
            labeled_data_path: Path to labeled data CSV
            
        Returns:
            Dictionary with results and file paths
        """
        try:
            # Step 1: Collect data
            logger.info("=" * 50)
            logger.info("STEP 1: DATA COLLECTION")
            logger.info("=" * 50)
            
            comments_df = self.collect_data(mode, urls, max_comments)
            
            if comments_df.empty:
                logger.error("No data collected. Exiting.")
                return {}
            
            # Step 2: Extract features (for original method or comparison)
            logger.info("=" * 50)
            logger.info("STEP 2: FEATURE EXTRACTION")
            logger.info("=" * 50)
            
            features_df = None
            if self.method == 'original' or self.method == 'both':
                features_df = self.extract_all_features(comments_df)
            else:
                logger.info("Skipping traditional feature extraction (using BotBuster)")
                # Create minimal features_df for compatibility
                features_df = comments_df[['author_id']].drop_duplicates()
            
            print("\n📊 Feature Statistics:")
            for col in features_df.select_dtypes(include=[np.number]).columns:
                if col != 'author_id':
                    print(f"  {col}: min={features_df[col].min():.3f}, max={features_df[col].max():.3f}, mean={features_df[col].mean():.3f}")

            # Step 3: Load labeled data (if provided)
            labeled_data = None
            if labeled_data_path:
                logger.info(f"Loading labeled data from {labeled_data_path}")
                labeled_data = self.collector.load_labeled_data(labeled_data_path)
            
            # Step 4: Detect bots
            logger.info("=" * 50)
            logger.info(f"STEP 3: BOT DETECTION ({self.method.upper()} METHOD)")
            logger.info("=" * 50)
            
            detection_results = self.detect_bots(features_df, comments_df, labeled_data)
            
            # Step 5: Visualize results
            logger.info("=" * 50)
            logger.info("STEP 4: VISUALIZATION")
            logger.info("=" * 50)
            
            visualization_files = self.visualize_results(detection_results, comments_df)
            
            # Step 6: Generate summary
            summary = self.visualizer.create_summary_report(
                detection_results, comments_df
            )
            
            # Add method info to summary
            summary['detection_method'] = self.method
            
            # Add BotBuster-specific summary info
            if self.method in ('botbuster', 'both'):
                comment_probs = self.botbuster_detector.get_comment_probabilities()
                if comment_probs:
                    summary['comment_level_stats'] = {
                        'total_comments_analyzed': len(comment_probs),
                        'avg_comment_bot_prob': np.mean(list(comment_probs.values())),
                        'max_comment_bot_prob': max(comment_probs.values()),
                        'high_prob_comments': sum(1 for p in comment_probs.values() if p > 0.7)
                    }
            
            # Persist the cluster-level comment evidence used for clustering decisions
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cluster_evidence_paths = self._save_cluster_comment_evidence(
                detection_results, comments_df, timestamp
            )
            if cluster_evidence_paths:
                summary['cluster_comment_evidence_file'] = cluster_evidence_paths.get('json')
                summary['cluster_comment_evidence_html'] = cluster_evidence_paths.get('html')
                summary['cluster_comment_evidence_index'] = cluster_evidence_paths.get('index')
            
            # Step 7: Save results
            logger.info("=" * 50)
            logger.info("STEP 5: SAVING RESULTS")
            logger.info("=" * 50)
            
            results_file = self.save_results(detection_results, summary, timestamp=timestamp)
            
            # Save comment-level probabilities for BotBuster
            if self.method in ('botbuster', 'both'):
                self._save_comment_probabilities(comments_df)
            
            # Print summary
            self.print_summary(summary)
            
            result_payload = {
                'summary': summary,
                'results_file': results_file,
                'visualizations': visualization_files,
                'detection_results': detection_results,
                'comments_df': comments_df,
                'cluster_comment_evidence': cluster_evidence_paths.get('json') if cluster_evidence_paths else "",
                'cluster_comment_evidence_html': cluster_evidence_paths.get('html') if cluster_evidence_paths else "",
                'cluster_comment_evidence_index': cluster_evidence_paths.get('index') if cluster_evidence_paths else "",
                'summary_file': getattr(self, 'last_summary_path', None)
            }
            
            if self.method == 'both':
                result_payload['original_results'] = getattr(self, 'original_results', None)
                result_payload['comparison'] = getattr(self, 'comparison_results', None)
            
            return result_payload
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
    
    def _open_visualizations(self, visualization_files: Dict[str, str]):
        """Best-effort open primary visualization HTMLs in the default browser"""
        if not visualization_files:
            return
        
        priority = [
            visualization_files.get('summary'),
            visualization_files.get('network'),
            visualization_files.get('clusters'),
            visualization_files.get('heatmap'),
            visualization_files.get('temporal'),
        ]
        opened = False
        for path in priority:
            if path:
                try:
                    abspath = os.path.abspath(path)
                    webbrowser.open(f"file://{abspath}", new=2)
                    opened = True
                except Exception as e:
                    logger.warning(f"Could not open visualization {path}: {e}")
        if opened:
            logger.info("Opened visualization(s) in default browser")

    def _print_next_steps(self, results: Dict):
        """Print concise next-step suggestions after a run"""
        viz = results.get('visualizations', {}) if results else {}
        print("\nNext steps:")
        print("  1) Review the HTML visualizations (network, cluster analysis, summary).")
        if viz:
            paths = [p for p in [
                viz.get('summary'),
                viz.get('network'),
                viz.get('clusters'),
                viz.get('heatmap'),
                viz.get('temporal')
            ] if p]
            if paths:
                print(f"     Opened best-available viz in your browser. Files live under: {os.path.dirname(paths[0])}")
        print("  2) Inspect the cluster evidence HTML for the comments that drove clustering.")
        if results.get('cluster_comment_evidence_index'):
            print(f"     Index: {results['cluster_comment_evidence_index']}")
        elif results.get('cluster_comment_evidence_html'):
            print(f"     Latest: {results['cluster_comment_evidence_html']}")
        print("  3) Export/quote findings as needed; consider re-running with --method both to compare detectors.")

    def _format_comment_records(self, comments_subset: pd.DataFrame) -> List[Dict]:
        """Convert a comment subset to serializable dicts"""
        records = []
        for _, row in comments_subset.iterrows():
            records.append({
                'comment_id': row.get('comment_id'),
                'author_id': row.get('author_id'),
                'video_id': row.get('video_id'),
                'published_at': str(row.get('published_at')),
                'text': row.get('text', '')
            })
        return records

    def _build_cluster_comment_evidence(self, detection_results: pd.DataFrame,
                                        comments_df: pd.DataFrame,
                                        max_comments_per_cluster: int = 100) -> Dict:
        """
        Build a per-cluster set of the comments that triggered clustering.
        
        For BotBuster this uses high-coordination comment pairs; for the legacy
        clustering path it falls back to a sample of comments from each cluster.
        """
        if detection_results is None or detection_results.empty:
            return {}
        if comments_df is None or comments_df.empty:
            return {}
        
        comments_df = comments_df.reset_index(drop=True).copy()
        cluster_assignments = detection_results.set_index('author_id')['cluster_id'].to_dict()
        comments_df['cluster_id'] = comments_df['author_id'].map(cluster_assignments).fillna(-1).astype(int)
        
        cluster_ids = sorted([cid for cid in comments_df['cluster_id'].unique() if cid != -1])
        if not cluster_ids:
            return {}
        
        use_coordination = (
            self.method in ('botbuster', 'both') and
            self.botbuster_detector.get_coordination_matrix() is not None
        )
        evidence_by_cluster = {}
        coord_threshold = None
        
        if use_coordination:
            coord_threshold = getattr(self.botbuster_detector, 'last_coordination_threshold', None) or 0.3
            evidence_by_cluster = self.botbuster_detector.extract_cluster_comment_evidence(
                comments_df=comments_df,
                cluster_assignments=cluster_assignments,
                max_comments_per_cluster=max_comments_per_cluster,
                coordination_threshold=coord_threshold
            )
        
        evidence_payload = {
            'generated_at': datetime.now().isoformat(),
            'detection_method': self.method,
            'coordination_threshold': coord_threshold if use_coordination else None,
            'max_comments_per_cluster': max_comments_per_cluster,
            'clusters': []
        }
        
        for cluster_id in cluster_ids:
            cluster_comments = comments_df[comments_df['cluster_id'] == cluster_id]
            entry = {
                'cluster_id': int(cluster_id),
                'account_count': int((detection_results['cluster_id'] == cluster_id).sum()),
                'comment_count': int(len(cluster_comments))
            }
            
            cluster_evidence = evidence_by_cluster.get(int(cluster_id), {})
            evidence_comments = cluster_evidence.get('comments', []) if cluster_evidence else []
            available_count = cluster_evidence.get('available_count', len(evidence_comments)) if cluster_evidence else len(cluster_comments)
            evidence_source = 'high_coordination_pairs' if evidence_comments else 'all_cluster_comments_sample'
            truncated = cluster_evidence.get('truncated', False) if cluster_evidence else False
            
            if not evidence_comments:
                evidence_comments = self._format_comment_records(
                    cluster_comments.head(max_comments_per_cluster)
                )
                truncated = len(cluster_comments) > len(evidence_comments)
            
            entry.update({
                'evidence_source': evidence_source,
                'available_evidence_comments': int(available_count),
                'evidence_comment_count': len(evidence_comments),
                'truncated': truncated,
                'evidence_comments': evidence_comments
            })
            
            evidence_payload['clusters'].append(entry)
        
        return evidence_payload

    def _render_cluster_comment_evidence_html(self, evidence_payload: Dict, output_path: str,
                                              max_text_len: int = 400):
        """Render a lightweight HTML view of cluster comment evidence for easier inspection"""
        if not evidence_payload or not evidence_payload.get('clusters'):
            return
        
        def esc(val: str) -> str:
            return html.escape(str(val)) if val is not None else ''
        
        lines = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'><head>")
        lines.append("<meta charset='UTF-8'><title>Cluster Comment Evidence</title>")
        lines.append("""
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
            h1 { margin-bottom: 6px; }
            h2 { margin-top: 32px; }
            .meta { color: #555; margin-bottom: 16px; }
            table { border-collapse: collapse; width: 100%; margin-top: 12px; }
            th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
            th { background: #f5f5f5; text-align: left; }
            tr:nth-child(even) { background: #fafafa; }
            .pill { display: inline-block; padding: 3px 8px; border-radius: 10px; font-size: 12px; color: #fff; }
            .pill-coord { background: #0077b6; }
            .pill-sample { background: #6c757d; }
            .pill-trunc { background: #e76f51; }
            .small { color: #666; font-size: 13px; }
            .text { white-space: pre-wrap; }
        </style>
        """)
        lines.append("</head><body>")
        
        lines.append("<h1>Cluster Comment Evidence</h1>")
        lines.append(f"<div class='meta'>Generated: {esc(evidence_payload.get('generated_at',''))} &middot; "
                     f"Method: {esc(evidence_payload.get('detection_method',''))} "
                     f"&middot; Coordination threshold: {esc(evidence_payload.get('coordination_threshold',''))}</div>")
        lines.append("<div class='meta'><a href='cluster_comment_evidence_index.html'>View all runs</a></div>")
        
        for cluster in sorted(evidence_payload.get('clusters', []), key=lambda c: c.get('cluster_id', 0)):
            cid = cluster.get('cluster_id', -1)
            lines.append(f"<h2>Cluster {esc(cid)}</h2>")
            lines.append("<div class='meta'>"
                         f"Accounts: {esc(cluster.get('account_count',0))} | "
                         f"Cluster comments: {esc(cluster.get('comment_count',0))} | "
                         f"Evidence comments: {esc(cluster.get('evidence_comment_count',0))}"
                         "</div>")
            
            source = cluster.get('evidence_source', 'unknown')
            pill_class = 'pill-coord' if source == 'high_coordination_pairs' else 'pill-sample'
            source_label = 'High coordination pairs' if source == 'high_coordination_pairs' else 'Sample from cluster'
            pills = [f"<span class='pill {pill_class}'>{esc(source_label)}</span>"]
            if cluster.get('truncated'):
                pills.append("<span class='pill pill-trunc'>Truncated for brevity</span>")
            lines.append("<div>" + " ".join(pills) + "</div>")
            
            lines.append("<table>")
            lines.append("<tr><th>Comment ID</th><th>Author</th><th>Video</th>"
                         "<th>Published</th><th>Max Coordination</th><th>Partner Author</th>"
                         "<th>Partner Comment</th><th>Text</th></tr>")
            
            for comment in cluster.get('evidence_comments', []):
                text_raw = comment.get('text', '')
                text = text_raw if len(text_raw) <= max_text_len else text_raw[:max_text_len] + '...'
                coord = comment.get('max_coordination_in_cluster', '')
                partner = comment.get('coordinated_with', {}) or {}
                lines.append("<tr>"
                             f"<td>{esc(comment.get('comment_id',''))}</td>"
                             f"<td>{esc(comment.get('author_id',''))}</td>"
                             f"<td>{esc(comment.get('video_id',''))}</td>"
                             f"<td>{esc(comment.get('published_at',''))}</td>"
                             f"<td>{esc(coord)}</td>"
                             f"<td>{esc(partner.get('author_id',''))}</td>"
                             f"<td>{esc(partner.get('comment_id',''))}</td>"
                             f"<td class='text'>{esc(text)}</td>"
                             "</tr>")
            lines.append("</table>")
            lines.append("<div class='small'>Showing up to "
                         f"{esc(evidence_payload.get('max_comments_per_cluster', ''))} comments per cluster.</div>")
        
        lines.append("</body></html>")
        
        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
    
    def _save_cluster_comment_evidence(self, detection_results: pd.DataFrame,
                                       comments_df: pd.DataFrame,
                                       timestamp: str,
                                       max_comments_per_cluster: int = 100) -> Dict[str, str]:
        """Save per-cluster comment evidence to JSON and HTML for easy viewing"""
        evidence_payload = self._build_cluster_comment_evidence(
            detection_results, comments_df, max_comments_per_cluster
        )
        if not evidence_payload or not evidence_payload.get('clusters'):
            return {}
        
        json_path = os.path.join(Config.REPORTS_DIR, f"cluster_comment_evidence_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(evidence_payload, f, indent=2, default=str)
        
        html_path = os.path.join(Config.REPORTS_DIR, f"cluster_comment_evidence_{timestamp}.html")
        self._render_cluster_comment_evidence_html(evidence_payload, html_path)
        index_path = self._update_cluster_evidence_index()
        
        logger.info(f"Saved cluster comment evidence to {json_path} and {html_path}")
        return {'json': json_path, 'html': html_path, 'index': index_path}
    
    def _save_comment_probabilities(self, comments_df: pd.DataFrame):
        """Save per-comment bot probabilities to CSV"""
        comment_probs = self.botbuster_detector.get_comment_probabilities()
        
        if not comment_probs:
            return
        
        # Create DataFrame with comment details and probabilities
        prob_df = comments_df[['comment_id', 'video_id', 'text', 'author_id', 'author', 'published_at']].copy()
        prob_df['bot_probability'] = prob_df['comment_id'].map(comment_probs)
        prob_df = prob_df.sort_values('bot_probability', ascending=False)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(Config.REPORTS_DIR, f"comment_bot_probabilities_{timestamp}.csv")
        prob_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved comment-level probabilities to {csv_path}")

    def _update_cluster_evidence_index(self) -> str:
        """Build an index HTML page linking to all saved cluster evidence runs"""
        pattern = os.path.join(Config.REPORTS_DIR, "cluster_comment_evidence_*.html")
        html_files = glob.glob(pattern)
        if not html_files:
            return ""
        
        runs = []
        for path in html_files:
            ts = os.path.basename(path).replace("cluster_comment_evidence_", "").replace(".html", "")
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(path))
            except Exception:
                mtime = datetime.min
            runs.append((mtime, ts, path))
        
        runs.sort(key=lambda x: x[0], reverse=True)
        
        def esc(val: str) -> str:
            return html.escape(str(val)) if val is not None else ''
        
        lines = []
        lines.append("<!DOCTYPE html>")
        lines.append("<html lang='en'><head>")
        lines.append("<meta charset='UTF-8'><title>Cluster Evidence Runs</title>")
        lines.append("""
        <style>
            body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
            h1 { margin-bottom: 10px; }
            table { border-collapse: collapse; width: 100%; margin-top: 12px; }
            th, td { border: 1px solid #ddd; padding: 8px; }
            th { background: #f5f5f5; text-align: left; }
            tr:nth-child(even) { background: #fafafa; }
            .pill { display: inline-block; padding: 3px 8px; border-radius: 10px; font-size: 12px; color: #fff; }
            .pill-latest { background: #2a9d8f; }
        </style>
        """)
        lines.append("</head><body>")
        lines.append("<h1>Cluster Comment Evidence Runs</h1>")
        lines.append("<table>")
        lines.append("<tr><th>Run</th><th>Timestamp</th><th>Path</th></tr>")
        for i, (mtime, ts, path) in enumerate(runs):
            fname = os.path.basename(path)
            pill = "<span class='pill pill-latest'>Latest</span>" if i == 0 else ""
            lines.append(
                "<tr>"
                f"<td><a href='{esc(fname)}'>{esc(fname)}</a> {pill}</td>"
                f"<td>{esc(mtime)}</td>"
                f"<td>{esc(path)}</td>"
                "</tr>"
            )
        lines.append("</table>")
        lines.append("</body></html>")
        
        index_path = os.path.join(Config.REPORTS_DIR, "cluster_comment_evidence_index.html")
        with open(index_path, 'w') as f:
            f.write("\n".join(lines))
        
        logger.info(f"Updated cluster evidence index at {index_path}")
        return index_path
    
    def print_summary(self, summary: Dict):
        """Print detection summary to console"""
        print("\n" + "=" * 60)
        method = summary.get('detection_method', 'unknown').upper()
        print(f"BOT DETECTION SUMMARY ({method} METHOD)")
        print("=" * 60)
        print(f"Total Accounts Analyzed: {summary['total_accounts']}")
        print(f"Total Comments Analyzed: {summary['total_comments']}")
        print(f"Clusters Found: {summary['clusters_found']}")
        print(f"Average Bot Probability: {summary['avg_bot_probability']:.2%}")
        
        print("\nClassification Results:")
        for classification, count in summary['classification_counts'].items():
            percentage = (count / summary['total_accounts']) * 100
            print(f"  {classification}: {count} ({percentage:.1f}%)")
        
        print(f"\nHigh Confidence Bots (>90% probability): {summary['high_confidence_bots']}")
        
        # BotBuster-specific stats
        if 'comment_level_stats' in summary:
            stats = summary['comment_level_stats']
            print("\nComment-Level Analysis (BotBuster):")
            print(f"  Total Comments Analyzed: {stats['total_comments_analyzed']}")
            print(f"  Average Comment Bot Probability: {stats['avg_comment_bot_prob']:.2%}")
            print(f"  Max Comment Bot Probability: {stats['max_comment_bot_prob']:.2%}")
            print(f"  High-Probability Comments (>70%): {stats['high_prob_comments']}")
        
        if summary.get('top_bot_accounts'):
            print("\nTop 5 Suspected Bot Accounts:")
            for i, account in enumerate(summary['top_bot_accounts'][:5], 1):
                print(f"  {i}. {account['author_id'][:30]}... (Probability: {account['final_bot_probability']:.2%})")
        
        print("=" * 60)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YouTube Botnet Detector - Educational Tool for Detecting Coordinated Bot Networks"
    )
    
    parser.add_argument(
        '--mode',
        choices=['urls', 'political', 'search'],
        default='political',
        help='Data collection mode'
    )
    
    parser.add_argument(
        '--urls',
        type=str,
        help='Comma-separated list of YouTube video URLs (for urls mode)'
    )
    
    parser.add_argument(
        '--max-comments',
        type=int,
        default=1000,
        help='Maximum comments to collect per video'
    )
    
    parser.add_argument(
        '--labeled-data',
        type=str,
        help='Path to CSV file with labeled bot data'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable API response caching'
    )
    
    parser.add_argument(
        '--clear-db',
        action='store_true',
        help='Clear database before starting'
    )
    
    parser.add_argument(
        '--method',
        choices=['botbuster', 'original', 'both'],
        default='botbuster',
        help='Detection method: botbuster (semantic+temporal coordination), original (clustering-based), or both for comparison'
    )
    
    args = parser.parse_args()
    
    # Parse URLs if provided
    urls = None
    if args.urls:
        urls = [url.strip() for url in args.urls.split(',')]
    
    # Initialize detector with selected method
    detector = YouTubeBotnetDetector(use_cache=not args.no_cache, method=args.method)
    
    # Clear database if requested
    if args.clear_db:
        logger.warning("Clearing database...")
        detector.db.clear_all_data()
    
    # Run pipeline
    results = detector.run_full_pipeline(
        mode=args.mode,
        urls=urls,
        max_comments=args.max_comments,
        labeled_data_path=args.labeled_data
    )
    
    if results:
        print("\n✅ Bot detection completed successfully!")
        print(f"🔬 Detection Method: {args.method.upper()}")
        print(f"📊 Results saved to: {results['results_file']}")
        if results.get('summary_file'):
            print(f"📝 Summary saved to: {results['summary_file']}")
        if results.get('cluster_comment_evidence'):
            print(f"🧵 Cluster evidence (comments): {results['cluster_comment_evidence']}")
        if results.get('cluster_comment_evidence_html'):
            print(f"🧾 Cluster evidence (HTML): {results['cluster_comment_evidence_html']}")
        if results.get('cluster_comment_evidence_index'):
            print(f"📚 Cluster evidence index (all runs): {results['cluster_comment_evidence_index']}")
        if results.get('visualizations'):
            print("📈 Visualizations created:")
            for viz_type, path in results['visualizations'].items():
                print(f"   - {viz_type}: {path}")
        if results.get('visualizations'):
            detector._open_visualizations(results.get('visualizations'))
        detector._print_next_steps(results)

if __name__ == "__main__":
    main()
