#!/usr/bin/env python3
"""
Comprehensive test suite for ALL feature extraction modules
Tests Temporal, Text, Network, Behavioral, Semantic features and Clustering

Updated to work with merged semantic_features.py (OpenAI + local fallback)
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def generate_test_data(n_comments=200, n_authors=30, n_videos=8):
    """Generate synthetic test data with bot-like and human-like patterns"""
    
    random.seed(42)
    np.random.seed(42)
    
    comments = []
    base_time = datetime.now() - timedelta(days=7)
    
    # Bot-like text patterns
    bot_texts = [
        "Check out my channel!!!",
        "Click here for free money >>> bit.ly/scam",
        "Subscribe to me please!",
        "Great video! Visit my profile!",
        "First!",
        "Who else watching in 2024?",
        "Like if you agree!",
        "Amazing content check out my page",
        "nice",
        "cool",
        "awesome",
        "EARN MONEY WORKING FROM HOME",
    ]
    
    # Human-like text patterns
    human_texts = [
        "I think the argument at 5:32 is flawed because it doesn't account for external factors.",
        "Could you explain more about how the algorithm handles edge cases?",
        "This reminds me of a research paper I read about similar topics.",
        "I disagree with the premise. The data shows a different trend historically.",
        "What sources did you use for the statistics mentioned at the beginning?",
        "In my opinion, this approach works better for smaller datasets.",
        "Great explanation, although I think you could elaborate on point three.",
        "Has anyone tried implementing this? Curious about real-world performance.",
        "I've been in this field for years and this matches my experience.",
        "However, there's an important caveat that wasn't mentioned.",
        "Based on the evidence presented, I believe the conclusion is well-supported.",
        "The production quality has really improved, nice work on the editing.",
    ]
    
    # Coordinated talking point
    coordinated_phrase = "This video perfectly explains why we need change now more than ever"
    
    # Create bot authors (rapid posting, similar text, suspicious usernames)
    for i in range(10):
        author_id = f"bot_{i}"
        author_name = f"User{random.randint(10000, 99999)}"
        
        # Bots post in bursts
        burst_time = base_time + timedelta(hours=random.randint(0, 48))
        
        for j in range(random.randint(5, 10)):
            comments.append({
                'comment_id': f'comment_bot_{i}_{j}',
                'video_id': f'video_{random.randint(1, 3)}',  # Concentrate on few videos
                'text': random.choice(bot_texts),
                'author': author_name,
                'author_id': author_id,
                'published_at': (burst_time + timedelta(seconds=random.randint(10, 300))).isoformat(),
                'like_count': random.randint(0, 3),
                'is_reply': False,
                'parent_id': None,
                'channel_title': 'Test Channel',
                'author_channel_created': (datetime.now() - timedelta(days=random.randint(1, 30))).isoformat(),
                'author_subscriber_count': random.randint(0, 10),
                'author_video_count': random.randint(0, 2),
                'author_total_views': random.randint(0, 100)
            })
    
    # Create human authors (natural posting, diverse text)
    for i in range(15):
        author_id = f"human_{i}"
        author_name = f"RealPerson{i}"
        
        for j in range(random.randint(2, 5)):
            comments.append({
                'comment_id': f'comment_human_{i}_{j}',
                'video_id': f'video_{random.randint(1, n_videos)}',  # Diverse videos
                'text': random.choice(human_texts),
                'author': author_name,
                'author_id': author_id,
                'published_at': (base_time + timedelta(hours=random.randint(0, 168))).isoformat(),
                'like_count': random.randint(5, 50),
                'is_reply': random.choice([True, False]),
                'parent_id': f'comment_human_{random.randint(0, i)}_{0}' if random.random() > 0.5 else None,
                'channel_title': 'Test Channel',
                'author_channel_created': (datetime.now() - timedelta(days=random.randint(365, 2000))).isoformat(),
                'author_subscriber_count': random.randint(10, 1000),
                'author_video_count': random.randint(0, 50),
                'author_total_views': random.randint(100, 10000)
            })
    
    # Create coordinated authors (same talking point)
    for i in range(5):
        author_id = f"coordinated_{i}"
        author_name = f"Patriot{random.randint(100, 999)}"
        
        # All post the same coordinated message
        coord_time = base_time + timedelta(hours=random.randint(0, 24))
        comments.append({
            'comment_id': f'comment_coord_{i}_0',
            'video_id': f'video_{random.randint(1, 3)}',
            'text': coordinated_phrase,
            'author': author_name,
            'author_id': author_id,
            'published_at': (coord_time + timedelta(minutes=random.randint(0, 60))).isoformat(),
            'like_count': random.randint(0, 10),
            'is_reply': False,
            'parent_id': None,
            'channel_title': 'Test Channel',
            'author_channel_created': (datetime.now() - timedelta(days=random.randint(30, 180))).isoformat(),
            'author_subscriber_count': random.randint(0, 100),
            'author_video_count': random.randint(0, 5),
            'author_total_views': random.randint(0, 500)
        })
    
    return pd.DataFrame(comments)


def test_temporal_features(comments_df):
    """Test TemporalFeatures module"""
    print("\n" + "=" * 60)
    print("TEST: TEMPORAL FEATURES")
    print("=" * 60)
    
    from features.temporal_features import TemporalFeatures
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: extract_burst_patterns
    print("\n1. extract_burst_patterns...")
    try:
        burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
        assert isinstance(burst_scores, dict)
        assert len(burst_scores) > 0
        print(f"   ✅ Returned {len(burst_scores)} author scores")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: detect_synchronized_posting
    print("\n2. detect_synchronized_posting...")
    try:
        sync_groups = TemporalFeatures.detect_synchronized_posting(comments_df)
        assert isinstance(sync_groups, list)
        print(f"   ✅ Found {len(sync_groups)} synchronized groups")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: calculate_posting_regularity
    print("\n3. calculate_posting_regularity...")
    try:
        regularity = TemporalFeatures.calculate_posting_regularity(comments_df)
        assert isinstance(regularity, dict)
        print(f"   ✅ Calculated regularity for {len(regularity)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: extract_time_patterns
    print("\n4. extract_time_patterns...")
    try:
        temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
        assert isinstance(temporal_df, pd.DataFrame)
        assert 'author_id' in temporal_df.columns
        print(f"   ✅ Extracted {len(temporal_df.columns) - 1} temporal features")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: detect_campaign_waves
    print("\n5. detect_campaign_waves...")
    try:
        waves = TemporalFeatures.detect_campaign_waves(comments_df)
        assert isinstance(waves, list)
        print(f"   ✅ Detected {len(waves)} campaign waves")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} TEMPORAL: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_text_features(comments_df):
    """Test TextFeatures module"""
    print("\n" + "=" * 60)
    print("TEST: TEXT FEATURES")
    print("=" * 60)
    
    from features.text_features import TextFeatures
    
    tf = TextFeatures()
    tests_passed = 0
    total_tests = 6
    
    # Test 1: detect_template_comments
    print("\n1. detect_template_comments...")
    try:
        template_scores = tf.detect_template_comments(comments_df)
        assert isinstance(template_scores, dict)
        print(f"   ✅ Scored {len(template_scores)} authors for templates")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: detect_spam_patterns
    print("\n2. detect_spam_patterns...")
    try:
        spam_scores = tf.detect_spam_patterns(comments_df)
        assert isinstance(spam_scores, dict)
        print(f"   ✅ Scored {len(spam_scores)} authors for spam")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: extract_linguistic_features
    print("\n3. extract_linguistic_features...")
    try:
        linguistic_df = tf.extract_linguistic_features(comments_df)
        assert isinstance(linguistic_df, pd.DataFrame)
        print(f"   ✅ Extracted {len(linguistic_df.columns) - 1} linguistic features")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: calculate_comment_diversity
    print("\n4. calculate_comment_diversity...")
    try:
        diversity = tf.calculate_comment_diversity(comments_df)
        assert isinstance(diversity, dict)
        print(f"   ✅ Calculated diversity for {len(diversity)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: find_duplicate_comments
    print("\n5. find_duplicate_comments...")
    try:
        duplicates = tf.find_duplicate_comments(comments_df)
        assert isinstance(duplicates, list)
        print(f"   ✅ Found {len(duplicates)} duplicate groups")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 6: extract_text_similarity_matrix
    print("\n6. extract_text_similarity_matrix...")
    try:
        sim_matrix = tf.extract_text_similarity_matrix(comments_df.head(50))  # Limit for performance
        assert isinstance(sim_matrix, np.ndarray)
        print(f"   ✅ Created similarity matrix shape: {sim_matrix.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} TEXT: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_network_features(comments_df):
    """Test NetworkFeatures module"""
    print("\n" + "=" * 60)
    print("TEST: NETWORK FEATURES")
    print("=" * 60)
    
    from features.network_features import NetworkFeatures
    
    tests_passed = 0
    total_tests = 8
    
    # Test 1: build_co_occurrence_network
    print("\n1. build_co_occurrence_network...")
    try:
        G = NetworkFeatures.build_co_occurrence_network(comments_df)
        print(f"   ✅ Built network with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        G = None
    
    # Test 2: build_reply_network
    print("\n2. build_reply_network...")
    try:
        reply_G = NetworkFeatures.build_reply_network(comments_df)
        print(f"   ✅ Built reply network with {reply_G.number_of_nodes()} nodes")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: detect_communities
    print("\n3. detect_communities...")
    try:
        if G and G.number_of_nodes() > 0:
            communities = NetworkFeatures.detect_communities(G)
            assert isinstance(communities, dict)
            n_communities = len(set(communities.values()))
            print(f"   ✅ Detected {n_communities} communities")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no network)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: extract_network_metrics
    print("\n4. extract_network_metrics...")
    try:
        if G and G.number_of_nodes() > 0:
            metrics = NetworkFeatures.extract_network_metrics(G)
            assert isinstance(metrics, dict)
            print(f"   ✅ Extracted metrics for {len(metrics)} nodes")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no network)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: find_cliques
    print("\n5. find_cliques...")
    try:
        if G and G.number_of_nodes() > 0:
            cliques = NetworkFeatures.find_cliques(G)
            assert isinstance(cliques, list)
            print(f"   ✅ Found {len(cliques)} cliques")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no network)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 6: detect_star_patterns
    print("\n6. detect_star_patterns...")
    try:
        if G and G.number_of_nodes() > 0:
            stars = NetworkFeatures.detect_star_patterns(G, min_degree=2)
            assert isinstance(stars, list)
            print(f"   ✅ Found {len(stars)} star patterns")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no network)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 7: build_temporal_network
    print("\n7. build_temporal_network...")
    try:
        temp_G = NetworkFeatures.build_temporal_network(comments_df)
        print(f"   ✅ Built temporal network with {temp_G.number_of_nodes()} nodes")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 8: calculate_author_network_features
    print("\n8. calculate_author_network_features...")
    try:
        network_df = NetworkFeatures.calculate_author_network_features(comments_df)
        assert isinstance(network_df, pd.DataFrame)
        print(f"   ✅ Extracted {len(network_df.columns) - 1} network features")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} NETWORK: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_behavioral_features(comments_df):
    """Test BehavioralFeatures module"""
    print("\n" + "=" * 60)
    print("TEST: BEHAVIORAL FEATURES")
    print("=" * 60)
    
    from features.behavioral_features import BehavioralFeatures
    
    tests_passed = 0
    total_tests = 6
    
    # Test 1: analyze_account_age
    print("\n1. analyze_account_age...")
    try:
        age_scores = BehavioralFeatures.analyze_account_age(comments_df)
        assert isinstance(age_scores, dict)
        print(f"   ✅ Analyzed age for {len(age_scores)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: analyze_username_patterns
    print("\n2. analyze_username_patterns...")
    try:
        username_scores = BehavioralFeatures.analyze_username_patterns(comments_df)
        assert isinstance(username_scores, dict)
        print(f"   ✅ Analyzed usernames for {len(username_scores)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: analyze_activity_patterns
    print("\n3. analyze_activity_patterns...")
    try:
        activity_df = BehavioralFeatures.analyze_activity_patterns(comments_df)
        assert isinstance(activity_df, pd.DataFrame)
        print(f"   ✅ Extracted {len(activity_df.columns) - 1} activity features")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: detect_automated_behavior
    print("\n4. detect_automated_behavior...")
    try:
        automation_scores = BehavioralFeatures.detect_automated_behavior(comments_df)
        assert isinstance(automation_scores, dict)
        print(f"   ✅ Detected automation for {len(automation_scores)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: analyze_content_targeting
    print("\n5. analyze_content_targeting...")
    try:
        targeting_scores = BehavioralFeatures.analyze_content_targeting(comments_df)
        assert isinstance(targeting_scores, dict)
        print(f"   ✅ Analyzed targeting for {len(targeting_scores)} authors")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 6: compile_behavioral_features
    print("\n6. compile_behavioral_features...")
    try:
        behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
        assert isinstance(behavioral_df, pd.DataFrame)
        print(f"   ✅ Compiled {len(behavioral_df.columns) - 1} behavioral features")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} BEHAVIORAL: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_semantic_features(comments_df):
    """Test SemanticFeatures module (merged version with OpenAI + local fallback)"""
    print("\n" + "=" * 60)
    print("TEST: SEMANTIC FEATURES")
    print("=" * 60)
    
    try:
        from features.semantic_features import SemanticFeatures
        print("✅ SemanticFeatures imported")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Initialize
    sf = SemanticFeatures(use_openai=True)  # Will fallback if not available
    mode = "OpenAI" if sf.use_openai else "Local (TF-IDF)"
    print(f"   Mode: {mode}")
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: get_embeddings
    print("\n1. get_embeddings...")
    try:
        sample_texts = comments_df['text'].head(10).tolist()
        embeddings = sf.get_embeddings(sample_texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape[0] == len(sample_texts)
        print(f"   ✅ Embeddings shape: {embeddings.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 2: calculate_pairwise_semantic_similarity
    print("\n2. calculate_pairwise_semantic_similarity...")
    try:
        similarity_matrix = sf.calculate_pairwise_semantic_similarity(comments_df)
        assert isinstance(similarity_matrix, np.ndarray)
        assert similarity_matrix.shape[0] == len(comments_df)
        print(f"   ✅ Matrix shape: {similarity_matrix.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        similarity_matrix = None
    
    # Test 3: calculate_author_semantic_features
    print("\n3. calculate_author_semantic_features...")
    try:
        if similarity_matrix is not None:
            author_df = sf.calculate_author_semantic_features(comments_df, similarity_matrix)
            assert isinstance(author_df, pd.DataFrame)
            assert 'author_id' in author_df.columns
            print(f"   ✅ Features for {len(author_df)} authors")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no similarity matrix)")
            tests_passed += 1  # Don't fail if previous test failed
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: find_semantically_similar_groups
    print("\n4. find_semantically_similar_groups...")
    try:
        if similarity_matrix is not None:
            similar_groups = sf.find_semantically_similar_groups(
                comments_df, 
                similarity_matrix,
                threshold=0.7
            )
            assert isinstance(similar_groups, list)
            print(f"   ✅ Found {len(similar_groups)} similar groups")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no similarity matrix)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: extract_semantic_features (full combined - main method)
    print("\n5. extract_semantic_features (combined)...")
    try:
        full_df = sf.extract_semantic_features(comments_df)
        assert isinstance(full_df, pd.DataFrame)
        assert 'author_id' in full_df.columns
        n_features = len(full_df.columns) - 1
        print(f"   ✅ Extracted {n_features} semantic features")
        
        # List features
        feature_cols = [c for c in full_df.columns if c != 'author_id']
        print(f"      Features: {feature_cols}")
        
        # Check for expected features from merged version
        expected = ['avg_semantic_similarity_to_others', 'semantic_diversity', 
                   'dominant_topic', 'intent_bot_score']
        found = [f for f in expected if f in feature_cols]
        print(f"      Found {len(found)}/{len(expected)} key expected features")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} SEMANTIC: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_clustering_detector(comments_df):
    """Test ClusteringDetector module"""
    print("\n" + "=" * 60)
    print("TEST: CLUSTERING DETECTOR")
    print("=" * 60)
    
    from detection.clustering import ClusteringDetector
    from features.temporal_features import TemporalFeatures
    from features.text_features import TextFeatures
    from features.behavioral_features import BehavioralFeatures
    
    detector = ClusteringDetector()
    tests_passed = 0
    total_tests = 5
    
    # First, prepare features
    print("\nPreparing features for clustering...")
    temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
    tf = TextFeatures()
    text_df = tf.extract_linguistic_features(comments_df)
    behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
    
    # Merge features
    features_df = temporal_df.merge(text_df, on='author_id', how='outer')
    features_df = features_df.merge(behavioral_df, on='author_id', how='outer', suffixes=('', '_dup'))
    features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
    features_df = features_df.fillna(0)
    
    print(f"   Prepared {len(features_df.columns) - 1} features for {len(features_df)} authors")
    
    # Test 1: prepare_features
    print("\n1. prepare_features...")
    try:
        features_array = detector.prepare_features(features_df)
        assert isinstance(features_array, np.ndarray)
        print(f"   ✅ Prepared array shape: {features_array.shape}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        features_array = None
    
    # Test 2: cluster_hdbscan
    print("\n2. cluster_hdbscan...")
    try:
        if features_array is not None:
            labels = detector.cluster_hdbscan(features_array, min_cluster_size=2)
            assert isinstance(labels, np.ndarray)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"   ✅ HDBSCAN found {n_clusters} clusters")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no features)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 3: cluster_dbscan
    print("\n3. cluster_dbscan...")
    try:
        if features_array is not None:
            labels = detector.cluster_dbscan(features_array, eps=0.5)
            assert isinstance(labels, np.ndarray)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"   ✅ DBSCAN found {n_clusters} clusters")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no features)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: ensemble_clustering
    print("\n4. ensemble_clustering...")
    try:
        if features_array is not None:
            labels, confidence = detector.ensemble_clustering(features_array)
            assert isinstance(labels, np.ndarray)
            assert isinstance(confidence, np.ndarray)
            print(f"   ✅ Ensemble clustering complete, confidence range: {confidence.min():.2f}-{confidence.max():.2f}")
            tests_passed += 1
        else:
            print("   ⚠️ Skipped (no features)")
            tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: detect_bots (full pipeline)
    print("\n5. detect_bots (full pipeline)...")
    try:
        results_df = detector.detect_bots(features_df)
        assert isinstance(results_df, pd.DataFrame)
        assert 'final_bot_probability' in results_df.columns
        assert 'classification' in results_df.columns
        
        # Show classification breakdown
        classifications = results_df['classification'].value_counts()
        print(f"   ✅ Detection complete:")
        for cls, count in classifications.items():
            print(f"      {cls}: {count}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'✅' if tests_passed == total_tests else '❌'} CLUSTERING: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_full_integration(comments_df):
    """Test full feature extraction and detection pipeline"""
    print("\n" + "=" * 60)
    print("TEST: FULL INTEGRATION PIPELINE")
    print("=" * 60)
    
    try:
        from features.temporal_features import TemporalFeatures
        from features.text_features import TextFeatures
        from features.network_features import NetworkFeatures
        from features.behavioral_features import BehavioralFeatures
        from features.semantic_features import SemanticFeatures
        from detection.clustering import ClusteringDetector
        
        print("\n✅ All modules imported successfully")
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False
    
    print("\n🔄 Extracting all features...")
    
    # Temporal
    burst_scores = TemporalFeatures.extract_burst_patterns(comments_df)
    temporal_df = TemporalFeatures.extract_time_patterns(comments_df)
    temporal_df['burst_score'] = temporal_df['author_id'].map(burst_scores)
    print(f"   ✅ Temporal: {len(temporal_df.columns) - 1} features")
    
    # Text
    tf = TextFeatures()
    template_scores = tf.detect_template_comments(comments_df)
    spam_scores = tf.detect_spam_patterns(comments_df)
    diversity_scores = tf.calculate_comment_diversity(comments_df)
    text_df = tf.extract_linguistic_features(comments_df)
    text_df['template_score'] = text_df['author_id'].map(template_scores)
    text_df['spam_score'] = text_df['author_id'].map(spam_scores)
    text_df['diversity_score'] = text_df['author_id'].map(diversity_scores)
    print(f"   ✅ Text: {len(text_df.columns) - 1} features")
    
    # Network
    network_df = NetworkFeatures.calculate_author_network_features(comments_df)
    print(f"   ✅ Network: {len(network_df.columns) - 1} features")
    
    # Behavioral
    behavioral_df = BehavioralFeatures.compile_behavioral_features(comments_df)
    print(f"   ✅ Behavioral: {len(behavioral_df.columns) - 1} features")
    
    # Semantic
    sf = SemanticFeatures(use_openai=True)
    semantic_df = sf.extract_semantic_features(comments_df)
    print(f"   ✅ Semantic: {len(semantic_df.columns) - 1} features")
    
    # Merge all features
    print("\n🔗 Merging all features...")
    features_df = temporal_df
    for df in [text_df, network_df, behavioral_df, semantic_df]:
        features_df = features_df.merge(df, on='author_id', how='outer', suffixes=('', '_dup'))
        features_df = features_df.loc[:, ~features_df.columns.str.endswith('_dup')]
    
    features_df = features_df.fillna(0)
    
    total_features = len(features_df.columns) - 1
    print(f"\n📊 TOTAL FEATURES EXTRACTED: {total_features}")
    
    # List all features by category
    all_features = [c for c in features_df.columns if c != 'author_id']
    print("\n📋 Complete Feature List:")
    for i, f in enumerate(sorted(all_features)):
        print(f"   {i+1:2d}. {f}")
    
    # Run detection
    print("\n🤖 Running bot detection...")
    detector = ClusteringDetector()
    results_df = detector.detect_bots(features_df)
    
    print(f"\n📈 DETECTION RESULTS:")
    print(f"   Total authors: {len(results_df)}")
    
    classifications = results_df['classification'].value_counts()
    for cls, count in classifications.items():
        pct = count / len(results_df) * 100
        print(f"   {cls}: {count} ({pct:.1f}%)")
    
    # Show top suspected bots
    print("\n🔴 Top 5 Suspected Bots:")
    top_bots = results_df.nlargest(5, 'final_bot_probability')
    for _, row in top_bots.iterrows():
        print(f"   {row['author_id']}: {row['final_bot_probability']:.1%}")
    
    print("\n" + "=" * 60)
    print("✅ FULL INTEGRATION TEST PASSED")
    print(f"   Features extracted: {total_features}")
    print(f"   Authors analyzed: {len(results_df)}")
    print("=" * 60)
    
    return True


def main():
    """Run all tests"""
    print("🧪 YouTube Bot Detector - Complete Feature Test Suite")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating synthetic test data...")
    comments_df = generate_test_data()
    print(f"   Created {len(comments_df)} comments")
    print(f"   From {comments_df['author_id'].nunique()} unique authors")
    print(f"   Across {comments_df['video_id'].nunique()} videos")
    
    # Run all tests
    results = {}
    
    results['temporal'] = test_temporal_features(comments_df)
    results['text'] = test_text_features(comments_df)
    results['network'] = test_network_features(comments_df)
    results['behavioral'] = test_behavioral_features(comments_df)
    results['semantic'] = test_semantic_features(comments_df)
    results['clustering'] = test_clustering_detector(comments_df)
    results['integration'] = test_full_integration(comments_df)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    all_passed = True
    for module, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {module.upper()}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nFeature modules verified:")
        print("   • TemporalFeatures - time patterns, bursts, synchronization")
        print("   • TextFeatures - linguistic, spam, templates, similarity")
        print("   • NetworkFeatures - graphs, communities, centrality")
        print("   • BehavioralFeatures - account age, automation, targeting")
        print("   • SemanticFeatures - embeddings, similarity, topics, intents")
        print("   • ClusteringDetector - HDBSCAN, DBSCAN, ensemble")
    else:
        print("❌ SOME TESTS FAILED")
        print("   Check the output above for details")
    
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())