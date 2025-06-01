import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1️⃣ MLflow 설정 (가장 먼저!)
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("MLflow Quickstart")

# 2️⃣ 데이터 준비
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ 모델 파라미터 정의
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

# 4️⃣ MLflow 실험 시작 - 이 안에서 학습과 추적을 동시에!
with mlflow.start_run():
    # 파라미터 먼저 기록
    mlflow.log_params(params)
    
    # 모델 학습
    lr = LogisticRegression(**params)
    lr.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = lr.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # 메트릭 기록
    mlflow.log_metric("accuracy", accuracy)
    
    # 태그 설정
    mlflow.set_tag("Training Info", "Basic LR model for iris data")
    
    # 모델 서명 생성 및 저장
    signature = infer_signature(X_train, lr.predict(X_train))
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tracking-quickstart",
    )
    
    # 결과 출력
    print(f"모델 정확도: {accuracy:.4f}")
    # print("MLflow UI에서 실험 결과를 확인하세요: http://localhost:5001")