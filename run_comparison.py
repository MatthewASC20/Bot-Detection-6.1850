#!/usr/bin/env python3
"""
BotBuster Comparison Script

Runs both the original clustering method and the new BotBuster method
on the same dataset to compare results.

Usage:
    python run_comparison.py --urls "VIDEO_URL1,VIDEO_URL2"
    python run_comparison.py --mode political --max-comments 500
"""

import argparse
import logging
import sys
import os
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config
from data_collection.data_collector import DataCollector
from detection.clustering import ClusteringDetector
from detection.botbuster_detector import BotBusterDetector
from features.temporal_features import TemporalFeatures
from features.text_features import TextFeatures
from features.network_features import NetworkFeatures
from features.behavioral_features import BehavioralFeatures
from visualization.network_viz import NetworkVisualizer
from storage.database import DatabaseHandler

# Configure logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format=Config.LOG_FORMAT,
    handlers=[
        logging.FileHandler('comparison_run.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def extract_features(comments_df: pd.DataFrame, text_features: TextFeatures) -> pd.DataFrame:
    """Extract features using original method"""
    logger.info("Extracting features (original method)...")
    
    # Temporal features
    burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
    temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
    temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
    
    # Text features
    template_scores = text_features.detect_template_comments(comments_df)
    spam_scores = text_features.detect_spam_patterns(comments_df)
    diversity_scores = text_features.calculate_comment_diversity(comments_df)
    linguistic_df = text_features.extract_linguistic_features(comments_df)
    
    linguistic_df['template_score'] = linguistic_df['author_id'].map(template_scores)
    linguistic_df['spam_score'] = linguistic_df['author_id'].map(spam_scores)
    linguistic_df['diversity_score'] = linguistic_df['author_id'].map(diversity_scores)
    
    # Network features
    network_df = NetworkFeatures.calculate_author_network_features(comments_df)
    
    # Behavioral features
    behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
    
    # Merge all features
    features_df = temporal_df
    for df in [linguistic_df, network_df, behavioral_df]:
        features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
        features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
    
    features_df = features_df.fillna(0)
    
    return features_df


def run_comparison(mode: str, urls: List[str] = None, max_comments: int = 500):
    """Run comparison between original and BotBuster methods"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize components
    collector = DataCollector(use_cache=True)
    text_features = TextFeatures()
    original_detector = ClusteringDetector()
    botbuster_detector = BotBusterDetector()
    visualizer = NetworkVisualizer()
    
    # Collect data
    logger.info("=" * 60)
    logger.info("STEP 1: DATA COLLECTION")
    logger.info("=" * 60)
    
    if mode == 'urls' and urls:
        comments_df = collector.collect_from_urls(urls, max_comments)
    elif mode == 'political':
        comments_df = collector.collect_political_content(max_comments_per_video=max_comments)
    elif mode == 'search':
        queries = Config.POLITICAL_KEYWORDS[:5]
        comments_df = collector.collect_by_search(queries, max_comments_per_video=max_comments)
    else:
        logger.error(f"Invalid mode: {mode}")
        return
    
    if comments_df.empty:
        logger.error("No data collected. Exiting.")
        return
    
    logger.info(f"Collected {len(comments_df)} comments from {comments_df['video_id'].nunique()} videos")
    logger.info(f"Unique authors: {comments_df['author_id'].nunique()}")
    
    # Run original method
    logger.info("=" * 60)
    logger.info("STEP 2: ORIGINAL METHOD (Clustering)")
    logger.info("=" * 60)
    
    features_df = extract_features(comments_df, text_features)
    original_results = original_detector.detect_bots(features_df)
    
    # Run BotBuster method
    logger.info("=" * 60)
    logger.info("STEP 3: BOTBUSTER METHOD (Semantic + Temporal)")
    logger.info("=" * 60)
    
    comments_df_reset = comments_df.reset_index(drop=True)
    botbuster_results = botbuster_detector.detect_bots(comments_df_reset)
    
    # Compare results
    logger.info("=" * 60)
    logger.info("STEP 4: COMPARISON")
    logger.info("=" * 60)
    
    comparison = compare_results(original_results, botbuster_results)
    
    # Save results
    logger.info("=" * 60)
    logger.info("STEP 5: SAVING RESULTS")
    logger.info("=" * 60)
    
    # Save comparison report
    comparison_path = os.path.join(Config.REPORTS_DIR, f"comparison_report_{timestamp}.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    # Save individual results
    original_path = os.path.join(Config.REPORTS_DIR, f"original_results_{timestamp}.csv")
    botbuster_path = os.path.join(Config.REPORTS_DIR, f"botbuster_results_{timestamp}.csv")
    
    original_results.to_csv(original_path, index=False)
    botbuster_results.to_csv(botbuster_path, index=False)
    
    # Save comment-level probabilities from BotBuster
    comment_probs = botbuster_detector.get_comment_probabilities()
    if comment_probs:
        prob_df = comments_df[['comment_id', 'video_id', 'text', 'author_id', 'author', 'published_at']].copy()
        prob_df['bot_probability'] = prob_df['comment_id'].map(comment_probs)
        prob_df = prob_df.sort_values('bot_probability', ascending=False)
        prob_path = os.path.join(Config.REPORTS_DIR, f"comment_probabilities_{timestamp}.csv")
        prob_df.to_csv(prob_path, index=False)
    
    # Print comparison summary
    print_comparison_summary(comparison)
    
    logger.info(f"\nResults saved to:")
    logger.info(f"  - Comparison: {comparison_path}")
    logger.info(f"  - Original results: {original_path}")
    logger.info(f"  - BotBuster results: {botbuster_path}")
    
    return comparison


def compare_results(original: pd.DataFrame, botbuster: pd.DataFrame) -> Dict:
    """Compare results from both methods"""
    
    # Merge on author_id
    merged = original.merge(
        botbuster, 
        on='author_id', 
        suffixes=('_original', '_botbuster'),
        how='outer'
    )
    
    # Calculate metrics
    comparison = {
        'original_method': {
            'total_accounts': len(original),
            'avg_bot_probability': original['final_bot_probability'].mean(),
            'likely_bots': (original['classification'] == 'likely_bot').sum(),
            'suspicious': (original['classification'] == 'suspicious').sum(),
            'likely_humans': (original['classification'] == 'likely_human').sum(),
            'clusters_found': original['cluster_id'].nunique() - (1 if -1 in original['cluster_id'].values else 0),
            'top_5_bot_accounts': original.nlargest(5, 'final_bot_probability')[
                ['author_id', 'final_bot_probability', 'cluster_id']
            ].to_dict('records')
        },
        'botbuster_method': {
            'total_accounts': len(botbuster),
            'avg_bot_probability': botbuster['final_bot_probability'].mean(),
            'likely_bots': (botbuster['classification'] == 'likely_bot').sum(),
            'suspicious': (botbuster['classification'] == 'suspicious').sum(),
            'likely_humans': (botbuster['classification'] == 'likely_human').sum(),
            'clusters_found': botbuster['cluster_id'].nunique() - (1 if -1 in botbuster['cluster_id'].values else 0),
            'top_5_bot_accounts': botbuster.nlargest(5, 'final_bot_probability')[
                ['author_id', 'final_bot_probability', 'cluster_id']
            ].to_dict('records')
        }
    }
    
    # Calculate correlation between methods
    common_authors = merged.dropna(subset=['final_bot_probability_original', 'final_bot_probability_botbuster'])
    if len(common_authors) > 1:
        correlation = common_authors['final_bot_probability_original'].corr(
            common_authors['final_bot_probability_botbuster']
        )
    else:
        correlation = None
    
    # Calculate agreement
    merged['classification_original'] = merged['classification_original'].fillna('unknown')
    merged['classification_botbuster'] = merged['classification_botbuster'].fillna('unknown')
    
    agreement = (merged['classification_original'] == merged['classification_botbuster']).mean()
    
    # Accounts flagged by one method but not the other
    original_only_bots = merged[
        (merged['classification_original'].isin(['likely_bot', 'suspicious'])) & 
        (merged['classification_botbuster'] == 'likely_human')
    ]['author_id'].tolist()
    
    botbuster_only_bots = merged[
        (merged['classification_botbuster'].isin(['likely_bot', 'suspicious'])) & 
        (merged['classification_original'] == 'likely_human')
    ]['author_id'].tolist()
    
    comparison['comparison'] = {
        'probability_correlation': correlation,
        'classification_agreement': agreement,
        'original_only_suspicious': len(original_only_bots),
        'botbuster_only_suspicious': len(botbuster_only_bots),
        'sample_original_only': original_only_bots[:5],
        'sample_botbuster_only': botbuster_only_bots[:5]
    }
    
    return comparison


def print_comparison_summary(comparison: Dict):
    """Print comparison summary to console"""
    
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY: Original vs BotBuster Method")
    print("=" * 70)
    
    orig = comparison['original_method']
    bb = comparison['botbuster_method']
    comp = comparison['comparison']
    
    print(f"\n{'Metric':<35} {'Original':<15} {'BotBuster':<15}")
    print("-" * 65)
    print(f"{'Total Accounts':<35} {orig['total_accounts']:<15} {bb['total_accounts']:<15}")
    print(f"{'Avg Bot Probability':<35} {orig['avg_bot_probability']:.2%}{'':<8} {bb['avg_bot_probability']:.2%}")
    print(f"{'Likely Bots':<35} {orig['likely_bots']:<15} {bb['likely_bots']:<15}")
    print(f"{'Suspicious':<35} {orig['suspicious']:<15} {bb['suspicious']:<15}")
    print(f"{'Likely Humans':<35} {orig['likely_humans']:<15} {bb['likely_humans']:<15}")
    print(f"{'Clusters Found':<35} {orig['clusters_found']:<15} {bb['clusters_found']:<15}")
    
    print("\n" + "-" * 65)
    print("Method Agreement Statistics:")
    print("-" * 65)
    
    if comp['probability_correlation'] is not None:
        print(f"Probability Correlation: {comp['probability_correlation']:.3f}")
    print(f"Classification Agreement: {comp['classification_agreement']:.1%}")
    print(f"Original-only Suspicious: {comp['original_only_suspicious']}")
    print(f"BotBuster-only Suspicious: {comp['botbuster_only_suspicious']}")
    
    print("\n" + "-" * 65)
    print("Top 5 Suspected Bots by Each Method:")
    print("-" * 65)
    
    print("\nOriginal Method:")
    for i, acc in enumerate(orig['top_5_bot_accounts'], 1):
        print(f"  {i}. {acc['author_id'][:35]}... Prob: {acc['final_bot_probability']:.2%}")
    
    print("\nBotBuster Method:")
    for i, acc in enumerate(bb['top_5_bot_accounts'], 1):
        print(f"  {i}. {acc['author_id'][:35]}... Prob: {acc['final_bot_probability']:.2%}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Original vs BotBuster bot detection methods"
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
        default=500,
        help='Maximum comments to collect per video'
    )
    
    args = parser.parse_args()
    
    # Parse URLs if provided
    urls = None
    if args.urls:
        urls = [url.strip() for url in args.urls.split(',')]
    
    # Run comparison
    run_comparison(
        mode=args.mode,
        urls=urls,
        max_comments=args.max_comments
    )


if __name__ == "__main__":
    main()

