import boto3, os, tempfile
from botocore.client import Config

s3 = boto3.resource('s3', endpoint_url='https://s3.cl4.du.cesnet.cz', 
                    aws_access_key_id='1Y920BKC0SAWPNDE8RD6', 
                    aws_secret_access_key='SnKMQbJ8mRKVboPDymkYFaFTz7VBxysrsWwJRoMD', 
                    config=Config(signature_version='s3v4', s3={'payload_signing_enabled': False}))
bucket = s3.Bucket('bucket')

prefix = 'Ostatni/Pollen_viability/trained_models/pollen_train_20260311_1901/'

keys = [obj.key for obj in bucket.objects.filter(Prefix=prefix)]
print(f"Found {len(keys)} objects for the previous run.")

csv_key = prefix + "results.csv"
if csv_key in keys:
    print("Downloading results.csv...")
    bucket.download_file(csv_key, 'temp_results.csv')
    with open('temp_results.csv', 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
            header = [x.strip() for x in lines[0].split(',')]
            last_line = [x.strip() for x in lines[-1].split(',')]
            print(f"Metrics (Epoch {last_line[header.index('epoch')]}):")
            for h, v in zip(header, last_line):
                if 'mAP' in h:
                    print(f"  {h}: {v}")
else:
    print("results.csv not found.")

