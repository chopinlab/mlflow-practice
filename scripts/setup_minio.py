#!/usr/bin/env python3
"""
MinIO 초기 설정 스크립트
MLflow 아티팩트 저장을 위한 S3 버킷을 생성합니다.
"""

import boto3
import os
from dotenv import load_dotenv

def setup_minio_bucket():
    # 환경변수 파일 로드
    load_dotenv('../mlflow.env')
    
    # MinIO 설정
    endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL')
    access_key = os.getenv('AWS_ACCESS_KEY_ID')
    secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    print(f"🔧 MinIO 연결 중... ({endpoint_url})")
    
    # S3 클라이언트 생성
    s3_client = boto3.client(
        's3',
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key
    )

    try:
        # 버킷 생성
        bucket_name = 'mlflow-artifacts'
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"✅ 버킷 '{bucket_name}' 생성 완료!")
        
        # 버킷 목록 확인
        response = s3_client.list_buckets()
        print("\n📁 현재 버킷 목록:")
        for bucket in response['Buckets']:
            print(f"  - {bucket['Name']}")
            
    except Exception as e:
        if 'BucketAlreadyOwnedByYou' in str(e):
            print(f"ℹ️  버킷 '{bucket_name}'이 이미 존재합니다.")
        else:
            print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    setup_minio_bucket() 