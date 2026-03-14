import os
import boto3
import sys
from botocore.client import Config
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
S3_ENDPOINT = os.environ.get('S3_ENDPOINT')
S3_BUCKET = os.environ.get('S3_BUCKET')
AWS_ACCESS_KEY_ID = os.environ.get('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.environ.get('AWS_SECRET_ACCESS_KEY')

# S3 Paths
S3_CSV_KEY = 'Ostatni/Pollen_viability/detected_images/pollen_counts.csv'
S3_MEASUREMENTS_KEY = 'Ostatni/Pollen_viability/detected_images/particle_measurements.csv'
LOCAL_RESULT_DIR = 'pollen_counting_results'
LOCAL_CSV_PATH = os.path.join(LOCAL_RESULT_DIR, 'pollen_counts.csv')
LOCAL_MEASUREMENTS_PATH = os.path.join(LOCAL_RESULT_DIR, 'particle_measurements.csv')

def setup_s3():
    if not all([S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY]):
        print("❌ Error: Missing S3 credentials.")
        print("   Please create a .env file with S3_ENDPOINT, S3_BUCKET, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY.")
        sys.exit(1)
    
    return boto3.resource('s3',
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        config=Config(signature_version='s3v4')
    )

def sync_results():
    print("🔄 Syncing results from S3...")
    
    # ensure local directory exists
    if not os.path.exists(LOCAL_RESULT_DIR):
        os.makedirs(LOCAL_RESULT_DIR)
        print(f"   Created directory: {LOCAL_RESULT_DIR}")

    s3 = setup_s3()
    bucket = s3.Bucket(S3_BUCKET)

    try:
        # 1. Sync Summary
        print(f"⬇️  Downloading {S3_CSV_KEY}...")
        bucket.download_file(S3_CSV_KEY, LOCAL_CSV_PATH)
        print(f"✅ Downloaded summary to {LOCAL_CSV_PATH}")
        
        # 2. Sync Detailed Measurements
        print(f"⬇️  Downloading {S3_MEASUREMENTS_KEY}...")
        bucket.download_file(S3_MEASUREMENTS_KEY, LOCAL_MEASUREMENTS_PATH)
        print(f"✅ Downloaded measurements to {LOCAL_MEASUREMENTS_PATH}")
        
    except Exception as e:
        print(f"❌ Failed to download results: {e}")
        # Check if 404
        if "404" in str(e):
             print("   (Files might not exist yet if no detection job has run successfully.)")

if __name__ == "__main__":
    sync_results()
