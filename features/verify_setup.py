#!/usr/bin/env python3
"""
Setup Verification Script
Run this first to ensure your project is correctly configured

Usage:
    cd ~/Documents/youtube_botnet_detector
    python3 verify_setup.py
"""

import sys
import os

print("=" * 70)
print("YOUTUBE BOTNET DETECTOR - SETUP VERIFICATION")
print("=" * 70)
print()

errors = []
warnings = []

# ============================================================
# Step 1: Check Python Version
# ============================================================

print("1. Checking Python version...")
python_version = sys.version_info
print(f"   Python {python_version.major}.{python_version.minor}.{python_version.micro}")

if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
    errors.append("Python 3.8+ required")
    print("   ❌ Python 3.8+ required")
else:
    print("   ✅ OK")

# ============================================================
# Step 2: Check Required Directories
# ============================================================

print("\n2. Checking directory structure...")

required_dirs = [
    'config',
    'data_collection',
    'features',
    'detection',
    'storage',
    'visualization',
    'utils',
    'api',
    'data',
    'outputs',
]

for d in required_dirs:
    if os.path.isdir(d):
        print(f"   ✅ {d}/")
    else:
        if d in ['api', 'discovery']:
            warnings.append(f"Directory missing (optional): {d}/")
            print(f"   ⚠️  {d}/ (will create)")
            os.makedirs(d, exist_ok=True)
        else:
            errors.append(f"Directory missing: {d}/")
            print(f"   ❌ {d}/ MISSING")

# ============================================================
# Step 3: Check Required Files
# ============================================================

print("\n3. Checking required files...")

required_files = {
    'config/config.py': 'Configuration settings',
    'config/__init__.py': 'Config package init',
    'features/temporal_features.py': 'Temporal feature extraction',
    'features/text_features.py': 'Text feature extraction',
    'features/network_features.py': 'Network feature extraction',
    'features/behavioral_features.py': 'Behavioral feature extraction',
    'features/__init__.py': 'Features package init',
    'detection/clustering.py': 'Clustering detector',
    'detection/__init__.py': 'Detection package init',
    'storage/database.py': 'Database handler',
    'storage/__init__.py': 'Storage package init',
    'data_collection/youtube_api.py': 'YouTube API wrapper',
    'data_collection/data_collector.py': 'Data collector',
    'data_collection/__init__.py': 'Data collection package init',
    'visualization/network_viz.py': 'Network visualization',
    'visualization/__init__.py': 'Visualization package init',
    'main.py': 'Main entry point',
}

for filepath, description in required_files.items():
    if os.path.isfile(filepath):
        print(f"   ✅ {filepath}")
    else:
        errors.append(f"File missing: {filepath} ({description})")
        print(f"   ❌ {filepath} - {description}")

# ============================================================
# Step 4: Check Required Packages
# ============================================================

print("\n4. Checking required packages...")

packages = {
    'pandas': 'Data processing',
    'numpy': 'Numerical computing',
    'sklearn': 'Machine learning (scikit-learn)',
    'hdbscan': 'Clustering algorithm',
    'networkx': 'Network analysis',
    'community': 'Community detection (python-louvain)',
    'nltk': 'Natural language processing',
    'textstat': 'Text statistics',
    'Levenshtein': 'String similarity',
    'plotly': 'Visualization',
    'flask': 'Web API',
    'flask_cors': 'CORS support',
    'sqlalchemy': 'Database ORM',
}

missing_packages = []
for package, description in packages.items():
    try:
        __import__(package)
        print(f"   ✅ {package}")
    except ImportError:
        missing_packages.append(package)
        errors.append(f"Package missing: {package} ({description})")
        print(f"   ❌ {package} - {description}")

if missing_packages:
    print(f"\n   Install missing packages:")
    print(f"   pip install {' '.join(missing_packages)}")

# ============================================================
# Step 5: Check NLTK Data
# ============================================================

print("\n5. Checking NLTK data...")

try:
    import nltk
    nltk_data = ['punkt', 'stopwords', 'vader_lexicon']
    
    for data in nltk_data:
        try:
            nltk.data.find(f'tokenizers/{data}' if data == 'punkt' else 
                          f'corpora/{data}' if data == 'stopwords' else
                          f'sentiment/{data}')
            print(f"   ✅ nltk.{data}")
        except LookupError:
            warnings.append(f"NLTK data missing: {data}")
            print(f"   ⚠️  nltk.{data} (downloading...)")
            try:
                nltk.download(data, quiet=True)
                print(f"   ✅ nltk.{data} (downloaded)")
            except:
                print(f"   ❌ Failed to download {data}")
except ImportError:
    print("   ❌ NLTK not installed")

# ============================================================
# Step 6: Check Environment Variables
# ============================================================

print("\n6. Checking environment configuration...")

if os.path.isfile('.env'):
    print("   ✅ .env file exists")
    
    # Check for API key
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        api_key = os.getenv('YOUTUBE_API_KEY')
        if api_key:
            print(f"   ✅ YOUTUBE_API_KEY configured ({api_key[:10]}...)")
        else:
            warnings.append("YOUTUBE_API_KEY not set in .env")
            print("   ⚠️  YOUTUBE_API_KEY not set (needed for data collection)")
    except ImportError:
        warnings.append("python-dotenv not installed")
        print("   ⚠️  python-dotenv not installed")
else:
    warnings.append(".env file not found")
    print("   ⚠️  .env file not found (copy from .env.template)")

# ============================================================
# Step 7: Test Module Imports
# ============================================================

print("\n7. Testing module imports...")

imports_to_test = [
    ('config.config', 'Config'),
    ('features.temporal_features', 'TemporalFeatures'),
    ('features.text_features', 'TextFeatures'),
    ('features.network_features', 'NetworkFeatures'),
    ('features.behavioral_features', 'BehavioralFeatures'),
    ('detection.clustering', 'ClusteringDetector'),
    ('storage.database', 'DatabaseHandler'),
    ('visualization.network_viz', 'NetworkVisualizer'),
]

for module_path, class_name in imports_to_test:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"   ✅ from {module_path} import {class_name}")
    except Exception as e:
        errors.append(f"Import failed: {module_path}.{class_name}")
        print(f"   ❌ from {module_path} import {class_name}")
        print(f"      Error: {e}")

# ============================================================
# Step 8: Test Database
# ============================================================

print("\n8. Testing database initialization...")

try:
    from storage.database import DatabaseHandler
    
    db = DatabaseHandler()
    print(f"   ✅ Database initialized at {db.db_path}")
    
    # Check tables exist
    import sqlite3
    with sqlite3.connect(db.db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"   ✅ Tables: {', '.join(tables)}")
        
except Exception as e:
    errors.append(f"Database initialization failed: {e}")
    print(f"   ❌ Database error: {e}")

# ============================================================
# Step 9: Test Feature Extraction (Quick)
# ============================================================

print("\n9. Quick feature extraction test...")

try:
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create minimal test data
    test_data = pd.DataFrame([
        {'comment_id': '1', 'video_id': 'v1', 'text': 'Test comment', 
         'author': 'user1', 'author_id': 'ch1', 
         'published_at': datetime.now().isoformat(), 'like_count': 0,
         'is_reply': False, 'parent_id': None, 'channel_title': 'Test'}
    ])
    
    from features.temporal_features import TemporalFeatures
    burst = TemporalFeatures.extract_burst_patterns(test_data)
    print(f"   ✅ TemporalFeatures working")
    
    from features.text_features import TextFeatures
    tf = TextFeatures()
    spam = tf.detect_spam_patterns(test_data)
    print(f"   ✅ TextFeatures working")
    
    from features.network_features import NetworkFeatures
    G = NetworkFeatures.build_co_occurrence_network(test_data)
    print(f"   ✅ NetworkFeatures working")
    
    from features.behavioral_features import BehavioralFeatures
    behavior = BehavioralFeatures.compile_behavioral_features(test_data)
    print(f"   ✅ BehavioralFeatures working")
    
    from detection.clustering import ClusteringDetector
    detector = ClusteringDetector()
    print(f"   ✅ ClusteringDetector working")
    
except Exception as e:
    errors.append(f"Feature extraction test failed: {e}")
    print(f"   ❌ Feature test error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ {len(errors)} ERROR(S) FOUND:")
    for e in errors:
        print(f"   - {e}")

if warnings:
    print(f"\n⚠️  {len(warnings)} WARNING(S):")
    for w in warnings:
        print(f"   - {w}")

if not errors:
    print("""
✅ ALL CHECKS PASSED!

Your setup is ready. Next steps:

1. Run the feature tests:
   python3 test_all_features.py

2. Start the API:
   python3 -m api.integrated_api

3. Test the API:
   python3 test_api_integration.py

4. Load the Chrome extension and browse YouTube!
""")
else:
    print("""
❌ SETUP INCOMPLETE

Fix the errors above before proceeding. Common fixes:

1. Missing packages:
   pip install -r requirements.txt

2. Missing NLTK data:
   python3 -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('vader_lexicon')"

3. Missing .env file:
   cp .env.template .env
   # Then edit .env and add your YouTube API key

4. Missing __init__.py files:
   touch features/__init__.py
   touch api/__init__.py
   touch discovery/__init__.py
""")

print("=" * 70)
