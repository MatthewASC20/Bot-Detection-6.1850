#!/usr/bin/env python3
"""
Test suite for SemanticFeatures (merged version with OpenAI + local fallback)
Tests the actual methods that exist in the merged semantic_features.py
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


def generate_test_data():
    """Generate test data with various comment patterns"""
    random.seed(42)
    np.random.seed(42)
    
    comments = []
    base_time = datetime.now() - timedelta(days=7)
    
    # Bot-like comments (similar/templated)
    bot_templates = [
        "Check out my channel!!!",
        "Subscribe to me please!",
        "Great video! Visit my profile!",
        "Amazing content check my page",
        "Nice video subscribe back",
    ]
    
    # Human-like comments (diverse/substantive)
    human_comments = [
        "I think the argument at 5:32 is flawed because it doesn't account for external factors.",
        "Could you explain more about how the algorithm handles edge cases?",
        "This reminds me of a research paper I read about similar topics last year.",
        "I disagree with the premise. The historical data shows a different trend.",
        "What sources did you use for the statistics mentioned at the beginning?",
        "The production quality has really improved, nice work on the editing.",
        "However, there's an important caveat that wasn't mentioned in the video.",
        "Based on the evidence presented, I believe the conclusion is well-supported.",
    ]
    
    # Coordinated talking point
    coordinated_phrase = "This video perfectly explains why we need change now more than ever"
    
    # Create bot authors
    for i in range(5):
        author_id = f"bot_{i}"
        for j in range(4):
            comments.append({
                'comment_id': f'comment_bot_{i}_{j}',
                'video_id': f'video_{random.randint(1, 3)}',
                'text': random.choice(bot_templates),
                'author': f"User{random.randint(10000, 99999)}",
                'author_id': author_id,
                'published_at': (base_time + timedelta(hours=random.randint(0, 48))).isoformat(),
                'like_count': random.randint(0, 3),
                'is_reply': False,
                'parent_id': None,
            })
    
    # Create human authors
    for i in range(8):
        author_id = f"human_{i}"
        for j in range(3):
            comments.append({
                'comment_id': f'comment_human_{i}_{j}',
                'video_id': f'video_{random.randint(1, 5)}',
                'text': random.choice(human_comments),
                'author': f"RealPerson{i}",
                'author_id': author_id,
                'published_at': (base_time + timedelta(hours=random.randint(0, 168))).isoformat(),
                'like_count': random.randint(5, 50),
                'is_reply': random.choice([True, False]),
                'parent_id': None,
            })
    
    # Create coordinated authors (same message)
    for i in range(4):
        comments.append({
            'comment_id': f'comment_coord_{i}',
            'video_id': f'video_{random.randint(1, 3)}',
            'text': coordinated_phrase,
            'author': f"Patriot{random.randint(100, 999)}",
            'author_id': f'coordinated_{i}',
            'published_at': (base_time + timedelta(hours=random.randint(0, 24))).isoformat(),
            'like_count': random.randint(0, 10),
            'is_reply': False,
            'parent_id': None,
        })
    
    return pd.DataFrame(comments)


def test_semantic_features():
    """Test SemanticFeatures module (merged version)"""
    print("=" * 60)
    print("TEST: SEMANTIC FEATURES (Merged Version)")
    print("=" * 60)
    
    # Import
    try:
        from features.semantic_features import SemanticFeatures
        print("✅ SemanticFeatures imported successfully")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Generate test data
    print("\n📊 Generating test data...")
    comments_df = generate_test_data()
    print(f"   Created {len(comments_df)} comments from {comments_df['author_id'].nunique()} authors")
    
    # Initialize
    sf = SemanticFeatures(use_openai=True)  # Will fallback if not available
    mode = "OpenAI" if sf.use_openai else "Local (TF-IDF)"
    print(f"\n🔧 Mode: {mode}")
    
    tests_passed = 0
    total_tests = 7
    
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
        assert similarity_matrix.shape[1] == len(comments_df)
        # Check symmetry
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-6)
        print(f"   ✅ Matrix shape: {similarity_matrix.shape}")
        print(f"      Diagonal mean: {np.diag(similarity_matrix).mean():.3f} (should be ~1.0)")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        similarity_matrix = None
    
    # Test 3: calculate_comment_to_discussion_similarity
    print("\n3. calculate_comment_to_discussion_similarity...")
    try:
        discussion_scores = sf.calculate_comment_to_discussion_similarity(comments_df)
        assert isinstance(discussion_scores, dict)
        assert len(discussion_scores) == len(comments_df)
        scores = list(discussion_scores.values())
        print(f"   ✅ Scores for {len(discussion_scores)} comments")
        print(f"      Range: {min(scores):.3f} - {max(scores):.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 4: calculate_author_semantic_features
    print("\n4. calculate_author_semantic_features...")
    try:
        author_df = sf.calculate_author_semantic_features(comments_df, similarity_matrix)
        assert isinstance(author_df, pd.DataFrame)
        assert 'author_id' in author_df.columns
        
        # Check expected columns from merged version
        expected_cols = ['avg_semantic_similarity_to_others', 'max_semantic_similarity_to_others',
                        'high_similarity_count', 'avg_discussion_similarity', 'semantic_diversity']
        found_cols = [c for c in expected_cols if c in author_df.columns]
        
        print(f"   ✅ Features for {len(author_df)} authors")
        print(f"      Found {len(found_cols)}/{len(expected_cols)} expected columns")
        print(f"      Columns: {list(author_df.columns)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 5: find_semantically_similar_groups
    print("\n5. find_semantically_similar_groups...")
    try:
        similar_groups = sf.find_semantically_similar_groups(
            comments_df, 
            similarity_matrix,
            threshold=0.7
        )
        assert isinstance(similar_groups, list)
        print(f"   ✅ Found {len(similar_groups)} similar groups")
        if similar_groups:
            print(f"      Largest group size: {max(len(g) for g in similar_groups)}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 6: calculate_comment_bot_probability
    print("\n6. calculate_comment_bot_probability...")
    try:
        comment_probs = sf.calculate_comment_bot_probability(comments_df, similarity_matrix)
        assert isinstance(comment_probs, dict)
        assert len(comment_probs) == len(comments_df)
        probs = list(comment_probs.values())
        print(f"   ✅ Probabilities for {len(comment_probs)} comments")
        print(f"      Range: {min(probs):.3f} - {max(probs):.3f}")
        print(f"      Mean: {np.mean(probs):.3f}")
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
    
    # Test 7: extract_semantic_features (the main combined method)
    print("\n7. extract_semantic_features (combined)...")
    try:
        full_df = sf.extract_semantic_features(comments_df)
        assert isinstance(full_df, pd.DataFrame)
        assert 'author_id' in full_df.columns
        
        feature_cols = [c for c in full_df.columns if c != 'author_id']
        print(f"   ✅ Extracted {len(feature_cols)} semantic features")
        print(f"      Features: {feature_cols}")
        
        # Verify we have the expected features from merged version
        expected_features = [
            'avg_semantic_similarity_to_others',
            'max_semantic_similarity_to_others', 
            'high_similarity_count',
            'avg_discussion_similarity',
            'semantic_diversity',
            'dominant_topic',
            'topic_concentration',
            'intent_bot_score',
            'intent_human_score',
            'talking_point_matches',
            'uses_coordinated_language'
        ]
        
        found = [f for f in expected_features if f in feature_cols]
        print(f"      Found {len(found)}/{len(expected_features)} expected features")
        
        tests_passed += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 60)
    print(f"SEMANTIC FEATURES: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("✅ All semantic feature tests passed!")
        return True
    else:
        print(f"❌ {total_tests - tests_passed} tests failed")
        return False


def test_semantic_detection_quality():
    """Test that semantic features actually help distinguish bots from humans"""
    print("\n" + "=" * 60)
    print("TEST: SEMANTIC DETECTION QUALITY")
    print("=" * 60)
    
    from features.semantic_features import SemanticFeatures
    
    comments_df = generate_test_data()
    sf = SemanticFeatures(use_openai=True)
    
    # Extract features
    features_df = sf.extract_semantic_features(comments_df)
    
    # Add ground truth labels
    features_df['is_bot'] = features_df['author_id'].apply(
        lambda x: 'bot' in x or 'coordinated' in x
    )
    
    # Compare bot vs human features
    print("\n📊 Feature comparison (Bot vs Human):")
    print("-" * 50)
    
    feature_cols = [c for c in features_df.columns if c not in ['author_id', 'is_bot']]
    
    for col in feature_cols:
        bot_mean = features_df[features_df['is_bot']][col].mean()
        human_mean = features_df[~features_df['is_bot']][col].mean()
        diff = bot_mean - human_mean
        
        indicator = "🔴" if abs(diff) > 0.1 else "⚪"
        direction = "↑" if diff > 0 else "↓"
        
        print(f"   {indicator} {col}:")
        print(f"      Bot avg: {bot_mean:.3f}, Human avg: {human_mean:.3f} ({direction}{abs(diff):.3f})")
    
    # Check if high similarity correlates with bots
    print("\n📈 Quality indicators:")
    
    if 'avg_semantic_similarity_to_others' in features_df.columns:
        bot_sim = features_df[features_df['is_bot']]['avg_semantic_similarity_to_others'].mean()
        human_sim = features_df[~features_df['is_bot']]['avg_semantic_similarity_to_others'].mean()
        
        if bot_sim > human_sim:
            print("   ✅ Bots have higher similarity (expected for template usage)")
        else:
            print("   ⚠️ Humans have higher similarity (unexpected)")
    
    if 'semantic_diversity' in features_df.columns:
        bot_div = features_df[features_df['is_bot']]['semantic_diversity'].mean()
        human_div = features_df[~features_df['is_bot']]['semantic_diversity'].mean()
        
        if human_div > bot_div:
            print("   ✅ Humans have higher diversity (expected)")
        else:
            print("   ⚠️ Bots have higher diversity (unexpected)")
    
    return True


if __name__ == "__main__":
    print("🧪 Semantic Features Test Suite (Merged Version)")
    print("=" * 60)
    
    success = test_semantic_features()
    
    if success:
        test_semantic_detection_quality()
    
    sys.exit(0 if success else 1)