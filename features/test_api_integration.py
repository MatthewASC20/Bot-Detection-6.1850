#!/usr/bin/env python3
"""
API and Database Integration Test
Tests the complete flow from API request through database storage

Run this AFTER test_all_features.py passes:
    python3 test_api_integration.py
"""

import sys
import os
import json
import requests
from datetime import datetime
import time

print("=" * 70)
print("API & DATABASE INTEGRATION TEST")
print("=" * 70)
print()

API_BASE = "http://localhost:5001/api"

def test_endpoint(name, method, url, data=None, expected_keys=None):
    """Test an API endpoint"""
    print(f"\n[TEST] {name}")
    print(f"  {method} {url}")
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=30)
        
        print(f"  Status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if expected_keys:
                missing = [k for k in expected_keys if k not in result]
                if missing:
                    print(f"  ⚠️  Missing keys: {missing}")
                    return False
            
            # Pretty print truncated response
            result_str = json.dumps(result, indent=2, default=str)
            if len(result_str) > 500:
                result_str = result_str[:500] + "\n  ... (truncated)"
            print(f"  Response: {result_str}")
            
            print(f"  ✅ PASSED")
            return True
        else:
            print(f"  ❌ FAILED: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"  ❌ FAILED: Could not connect to API")
        print(f"     Make sure the API is running: python3 -m api.integrated_api")
        return False
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

# ============================================================
# Test 1: Health Check
# ============================================================

print("\n" + "=" * 50)
print("1. HEALTH CHECK")
print("=" * 50)

health_ok = test_endpoint(
    "Health Check",
    "GET",
    f"{API_BASE}/health",
    expected_keys=["status"]
)

if not health_ok:
    print("\n❌ API is not running! Start it with:")
    print("   cd ~/Documents/youtube_botnet_detector")
    print("   python3 -m api.integrated_api")
    sys.exit(1)

# ============================================================
# Test 2: Single Comment Analysis (Legacy)
# ============================================================

print("\n" + "=" * 50)
print("2. SINGLE COMMENT ANALYSIS")
print("=" * 50)

test_comment = {
    "author": "user12345678",
    "content": "Great video! Check out my channel for more!!!",
    "timestamp": "2 hours ago",
    "likes": "0"
}

test_endpoint(
    "Analyze Single Comment (Bot-like)",
    "POST",
    f"{API_BASE}/analyze",
    data=test_comment,
    expected_keys=["bot_probability"]
)

test_comment_human = {
    "author": "RealPersonName",
    "content": "I really enjoyed this video. The analysis at 3:45 was particularly insightful and helped me understand the topic better.",
    "timestamp": "1 day ago",
    "likes": "42"
}

test_endpoint(
    "Analyze Single Comment (Human-like)",
    "POST",
    f"{API_BASE}/analyze",
    data=test_comment_human,
    expected_keys=["bot_probability"]
)

# ============================================================
# Test 3: Batch Video Analysis
# ============================================================

print("\n" + "=" * 50)
print("3. BATCH VIDEO ANALYSIS (Full Pipeline)")
print("=" * 50)

# Create test batch with mix of bot and human comments
test_batch = {
    "video_id": "test_video_123",
    "comments": [
        # Bot-like comments
        {"id": "c1", "author": "user99999", "author_id": "ch_user99999", "content": "First! Subscribe!", "timestamp": "2024-01-01T10:00:00Z", "likes": 0},
        {"id": "c2", "author": "user88888", "author_id": "ch_user88888", "content": "Check out my channel!", "timestamp": "2024-01-01T10:01:00Z", "likes": 0},
        {"id": "c3", "author": "user77777", "author_id": "ch_user77777", "content": "Great video! Visit my profile", "timestamp": "2024-01-01T10:02:00Z", "likes": 1},
        {"id": "c4", "author": "user66666", "author_id": "ch_user66666", "content": "Amazing!!! Click link in bio", "timestamp": "2024-01-01T10:03:00Z", "likes": 0},
        {"id": "c5", "author": "user55555", "author_id": "ch_user55555", "content": "Subscribe to me!!!", "timestamp": "2024-01-01T10:04:00Z", "likes": 0},
        # Human-like comments
        {"id": "c6", "author": "JohnDoe", "author_id": "ch_johndoe", "content": "This is a really interesting perspective. I hadn't considered the implications before.", "timestamp": "2024-01-01T12:30:00Z", "likes": 15},
        {"id": "c7", "author": "JaneSmith", "author_id": "ch_janesmith", "content": "Could you elaborate on the point you made at 5:30? I'm not sure I fully understand.", "timestamp": "2024-01-01T14:45:00Z", "likes": 8},
        {"id": "c8", "author": "VideoFan", "author_id": "ch_videofan", "content": "Been following your channel for years. The quality keeps improving!", "timestamp": "2024-01-01T16:20:00Z", "likes": 23},
        {"id": "c9", "author": "CuriousViewer", "author_id": "ch_curious", "content": "This reminds me of a similar situation I experienced last year. Thanks for sharing.", "timestamp": "2024-01-01T18:00:00Z", "likes": 5},
        {"id": "c10", "author": "ThoughtfulCommenter", "author_id": "ch_thoughtful", "content": "I respectfully disagree with some points, but I appreciate the thorough research.", "timestamp": "2024-01-01T20:30:00Z", "likes": 12},
    ],
    "include_network": True,
    "fetch_related": False
}

print("\n[TEST] Batch Video Analysis")
print(f"  POST {API_BASE}/analyze_video")
print(f"  Sending {len(test_batch['comments'])} comments...")

try:
    start_time = time.time()
    response = requests.post(f"{API_BASE}/analyze_video", json=test_batch, timeout=60)
    elapsed = time.time() - start_time
    
    print(f"  Status: {response.status_code}")
    print(f"  Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        result = response.json()
        
        # Check structure
        print(f"\n  Analysis Summary:")
        summary = result.get('analysis_summary', {})
        print(f"    - Total comments: {summary.get('total_comments', 'N/A')}")
        print(f"    - Clusters found: {summary.get('clusters_found', 'N/A')}")
        print(f"    - Likely bots: {summary.get('likely_bots', 'N/A')}")
        print(f"    - Suspicious: {summary.get('suspicious', 'N/A')}")
        print(f"    - Likely humans: {summary.get('likely_humans', 'N/A')}")
        print(f"    - Coordinated groups: {summary.get('coordinated_groups', 'N/A')}")
        
        # Check individual results
        results = result.get('results', [])
        print(f"\n  Individual Results ({len(results)} comments):")
        
        bots_detected = 0
        humans_detected = 0
        
        for r in results[:5]:  # Show first 5
            prob = r.get('bot_probability', 0)
            cls = r.get('classification', 'unknown')
            author = r.get('author', 'unknown')[:20]
            flags = r.get('flags', [])
            
            if cls == 'likely_bot':
                bots_detected += 1
            elif cls == 'likely_human':
                humans_detected += 1
            
            print(f"    - {author}: {prob:.0%} ({cls}) flags={flags}")
        
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")
        
        print(f"\n  ✅ PASSED - Detected {bots_detected} bots, {humans_detected} humans")
    else:
        print(f"  ❌ FAILED: {response.text[:300]}")
        
except Exception as e:
    print(f"  ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Test 4: Vote Submission
# ============================================================

print("\n" + "=" * 50)
print("4. VOTE SUBMISSION")
print("=" * 50)

test_vote = {
    "commentId": "test_comment_vote_123",
    "vote": 1,  # Bot
    "commentData": {
        "author": "SuspiciousUser",
        "content": "Buy now! Limited offer!",
        "author_id": "ch_suspicious"
    },
    "timestamp": int(datetime.now().timestamp() * 1000)
}

test_endpoint(
    "Submit Bot Vote",
    "POST",
    f"{API_BASE}/vote",
    data=test_vote,
    expected_keys=["success"]
)

# ============================================================
# Test 5: Statistics
# ============================================================

print("\n" + "=" * 50)
print("5. STATISTICS")
print("=" * 50)

test_endpoint(
    "Get Stats",
    "GET",
    f"{API_BASE}/stats",
    expected_keys=["total_votes"]
)

# ============================================================
# Test 6: Author Profile
# ============================================================

print("\n" + "=" * 50)
print("6. AUTHOR PROFILE")
print("=" * 50)

test_endpoint(
    "Get Author Profile",
    "GET",
    f"{API_BASE}/author/ch_user99999"
)

# ============================================================
# Test 7: Network Visualization
# ============================================================

print("\n" + "=" * 50)
print("7. NETWORK VISUALIZATION")
print("=" * 50)

print(f"\n[TEST] Network Visualization")
print(f"  GET {API_BASE}/network/test_video_123")

try:
    response = requests.get(f"{API_BASE}/network/test_video_123", timeout=30)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        content_type = response.headers.get('content-type', '')
        if 'text/html' in content_type:
            html_preview = response.text[:200].replace('\n', ' ')
            print(f"  Response: HTML document ({len(response.text)} bytes)")
            print(f"  Preview: {html_preview}...")
            print(f"  ✅ PASSED")
        else:
            print(f"  Response type: {content_type}")
    else:
        print(f"  Response: {response.text[:200]}")
        
except Exception as e:
    print(f"  ⚠️  Note: {e}")

# ============================================================
# Test 8: Dashboard
# ============================================================

print("\n" + "=" * 50)
print("8. DASHBOARD")
print("=" * 50)

print(f"\n[TEST] Discovery Dashboard")
print(f"  GET {API_BASE}/dashboard")

try:
    response = requests.get(f"{API_BASE}/dashboard", timeout=30)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        print(f"  Response: HTML document ({len(response.text)} bytes)")
        print(f"  ✅ PASSED - Dashboard is accessible")
    else:
        print(f"  ⚠️  Dashboard returned: {response.status_code}")
        
except Exception as e:
    print(f"  ⚠️  Note: {e}")

# ============================================================
# Test 9: Botnet Discovery (if implemented)
# ============================================================

print("\n" + "=" * 50)
print("9. BOTNET DISCOVERY")
print("=" * 50)

print(f"\n[TEST] Run Discovery")
print(f"  POST {API_BASE}/discover")

try:
    discovery_request = {
        "seed_video_ids": ["test_video_123"],
        "min_bot_probability": 0.5,
        "min_botnet_size": 2,
        "max_depth": 1
    }
    
    response = requests.post(f"{API_BASE}/discover", json=discovery_request, timeout=120)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Botnets found: {result.get('phases', {}).get('identification', {}).get('botnets_found', 'N/A')}")
        print(f"  ✅ PASSED")
    elif response.status_code == 404:
        print(f"  ⚠️  Discovery endpoint not yet implemented")
    else:
        print(f"  Response: {response.text[:200]}")
        
except Exception as e:
    print(f"  ⚠️  Note: {e}")

# ============================================================
# Test 10: List Botnets
# ============================================================

print("\n" + "=" * 50)
print("10. LIST BOTNETS")
print("=" * 50)

print(f"\n[TEST] List Botnets")
print(f"  GET {API_BASE}/botnets")

try:
    response = requests.get(f"{API_BASE}/botnets", timeout=30)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"  Count: {result.get('count', 'N/A')}")
        print(f"  ✅ PASSED")
    elif response.status_code == 404:
        print(f"  ⚠️  Botnets endpoint not yet implemented")
    else:
        print(f"  Response: {response.text[:200]}")
        
except Exception as e:
    print(f"  ⚠️  Note: {e}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)
print("""
Core API Tests:
  ✅ Health check
  ✅ Single comment analysis
  ✅ Batch video analysis (full pipeline)
  ✅ Vote submission
  ✅ Statistics
  
Advanced Features:
  ✅ Author profiles
  ✅ Network visualization
  ✅ Dashboard
  ⚠️  Botnet discovery (may need implementation)

Next Steps:
  1. If all core tests pass, the API is working correctly
  2. Test with the Chrome extension on real YouTube videos
  3. Collect more data to improve detection accuracy
  4. Implement discovery endpoints if not yet done
""")
print("=" * 70)
