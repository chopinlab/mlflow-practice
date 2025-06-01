import os
import mlflow
from mlflow.models import infer_signature
from dotenv import load_dotenv

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 환경변수 파일 로드
load_dotenv('mlflow.env')

# 1️⃣ MLflow 설정 - MinIO S3 호환 저장소 사용
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("MLflow S3 Test")

# 2️⃣ 데이터 준비
X, y = datasets.load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ 모델 파라미터 정의
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
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
        name="iris_model",
        signature=signature,
        input_example=X_train[:5]
    )
    
    # 결과 출력
    print(f"✅ 모델 학습 완료!")
    print(f"📊 모델 정확도: {accuracy:.4f}")
    print(f"🔗 모델 URI: {model_info.model_uri}")

print("\n" + "="*60)
print("🔄 MLflow에서 저장된 모델 로드 및 예측 테스트")
print("="*60)


# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

# 결과를 DataFrame으로 정리
iris_feature_names = [
    "sepal length (cm)",
    "sepal width (cm)", 
    "petal length (cm)",
    "petal width (cm)"
]
iris_target_names = ["setosa", "versicolor", "virginica"]

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

result[:4]
