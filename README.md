# MLflow Practice 프로젝트

MLflow와 사이킷런을 함께 사용한 머신러닝 모델 실험 추적 및 분석 프로젝트

## 🎯 프로젝트 목적

- MLflow를 사용한 머신러닝 실험 추적
- 사이킷런과 MLflow의 통합 사용법 학습
- 여러 모델 비교 분석 및 성능 평가

## 📋 주요 기능

### 1. 기본 실험 추적 (`src/basic_example.py`)
- 단일 모델(로지스틱 회귀) 학습 및 추적
- 파라미터, 메트릭, 모델 아티팩트 저장

### 2. 모델 비교 분석 (`src/model_comparison.py`)
- 여러 알고리즘 동시 비교 (로지스틱 회귀, 랜덤 포레스트, SVM)
- 교차 검증을 통한 성능 평가
- 시각화를 통한 결과 분석

## 🚀 사용법

### 1. MLflow 서버 시작
```bash

docker-compose --env-file mlflow.env down -v
# Docker Compose로 MLflow 서버 실행
docker-compose --env-file mlflow.env up -d

# 또는 로컬 MLflow 서버 실행
mlflow ui --host 0.0.0.0 --port 8080
```

### 2. 기본 실험 실행
```bash
python -m src.basic_example
```

### 3. 모델 비교 분석 실행
```bash
python -m src.model_comparison
```

### 4. MLflow UI 확인
브라우저에서 `http://localhost:8080` 접속하여 실험 결과 확인

## 🔄 MLflow + 사이킷런 워크플로우

```python
# 1단계: MLflow 설정
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("실험_이름")

# 2단계: 실험 시작 + 사이킷런 모델 학습
with mlflow.start_run():
    # 파라미터 기록 (MLflow)
    mlflow.log_params(model_params)
    
    # 모델 학습 (사이킷런)
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    # 평가 (사이킷런)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 메트릭 기록 (MLflow)
    mlflow.log_metric("accuracy", accuracy)
    
    # 모델 저장 (MLflow + 사이킷런)
    mlflow.sklearn.log_model(model, "model")
```

## 📊 분석 가능한 내용

### 1. 모델 성능 비교
- 정확도, 정밀도, 재현율, F1 점수
- 교차 검증 점수 및 표준편차
- 모델별 학습 시간

### 2. 하이퍼파라미터 최적화
- 파라미터 조합별 성능 추적
- 최적 파라미터 조합 자동 식별

### 3. 실험 버전 관리
- 모델 버전별 성능 변화 추적
- 실험 재현성 보장

### 4. 시각화 분석
- 모델별 성능 비교 차트
- 메트릭 히트맵
- 교차 검증 결과 시각화

## 🛠️ 의존성

- Python >= 3.10
- MLflow >= 3.0.0rc2
- scikit-learn
- pandas
- matplotlib
- seaborn
- psycopg2-binary (PostgreSQL 연결용)

## 📁 프로젝트 구조

```
mlflow-practice/
├── src/                          # Python 소스 코드
│   ├── __init__.py              # 패키지 초기화
│   ├── basic_example.py         # 기본 MLflow 사용 예제
│   └── model_comparison.py      # 모델 비교 분석 스크립트
├── Dockerfile.mlflow            # MLflow 서버 Docker 이미지
├── docker-compose.yaml          # MLflow 서버 설정
├── pyproject.toml              # 프로젝트 설정
├── uv.lock                     # 의존성 잠금 파일
├── .gitignore                  # Git 제외 파일 설정
├── mlruns/                     # MLflow 실험 데이터 (git 제외)
└── mlartifacts/                # MLflow 아티팩트 저장소 (git 제외)
``` 