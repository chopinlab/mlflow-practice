"""
MLflow Practice 프로젝트

MLflow와 사이킷런을 함께 사용한 머신러닝 모델 실험 추적 및 분석
"""

__version__ = "0.1.0"
__author__ = "MLflow Practice Team"

# 주요 모듈들을 import하기 쉽게 만들기
from . import basic_example
from . import model_comparison

__all__ = ["basic_example", "model_comparison"] 