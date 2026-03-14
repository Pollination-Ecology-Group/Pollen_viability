import boto3
import os

S3_ENDPOINT = "https://s3.cl4.du.cesnet.cz"
S3_BUCKET = "bucket"
AWS_ACCESS_KEY_ID = "1Y920BKC0SAWPNDE8RD6"
AWS_SECRET_ACCESS_KEY = "SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD"

s3 = boto3.resource('s3',
    endpoint_url=S3_ENDPOINT,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

bucket = s3.Bucket(S3_BUCKET)
prefix = 'Ostatni/Pollen_viability/'

print(f"Listing objects in {prefix}...")
objects = bucket.objects.filter(Prefix=prefix)
for obj in objects:
    if 'detected' in obj.key:
        print(f"{obj.key} ({obj.size} bytes)")
