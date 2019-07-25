from .coco import COCO

import boto3
import os

BUCKET_NAME = 'sotabench'
OUTPUT_DIR = './data/'

if not os.path.isdir(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)

s3 = boto3.client('s3')
list = s3.list_objects(Bucket=BUCKET_NAME)['Contents']
files = [i for i in s3.list_objects(Bucket='sotabench')['Contents'] if i['Key'][-1] != '/']

for file in files:
    if 'VOC' not in file['Key']:
        continue
        print(file['Key'])
    output_location = '%s%s' % (OUTPUT_DIR, file['Key'].split('/')[-1])
    s3.download_file(BUCKET_NAME, file['Key'], output_location)