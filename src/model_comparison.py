import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# MLflow 설정
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("Model Comparison Analysis")

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """MLflow와 사이킷런을 함께 사용한 모델 평가 함수"""
    
    with mlflow.start_run(run_name=model_name):
        # 1️⃣ 모델 파라미터 기록 (MLflow)
        mlflow.log_params(model.get_params())
        
        # 2️⃣ 모델 학습 (사이킷런)
        model.fit(X_train, y_train)
        
        # 3️⃣ 예측 및 평가 (사이킷런)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # 4️⃣ 메트릭 계산 (사이킷런)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # 5️⃣ Cross-validation 점수 (사이킷런)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 6️⃣ 메트릭 기록 (MLflow)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_score_mean", cv_mean)
        mlflow.log_metric("cv_score_std", cv_std)
        
        # 7️⃣ 태그 설정 (MLflow)
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset", "iris")
        
        # 8️⃣ 모델 저장 (MLflow)
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=f"iris_{model_name.lower()}"
        )
        
        # 9️⃣ 분류 리포트 저장 (MLflow + 사이킷런)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv("classification_report.csv")
        mlflow.log_artifact("classification_report.csv")
        
        print(f"\n=== {model_name} 결과 ===")
        print(f"정확도: {accuracy:.4f}")
        print(f"정밀도: {precision:.4f}")
        print(f"재현율: {recall:.4f}")
        print(f"F1 점수: {f1:.4f}")
        print(f"CV 점수: {cv_mean:.4f} (±{cv_std:.4f})")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }

def main():
    """메인 실험 함수 - MLflow와 사이킷런 교차 사용"""
    
    # 1️⃣ 데이터 준비 (사이킷런)
    print("📊 데이터 로딩 중...")
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 2️⃣ 여러 모델 정의 (사이킷런)
    models = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100),
        "SVM": SVC(random_state=42, probability=True)
    }
    
    # 3️⃣ 각 모델에 대해 실험 실행 (MLflow + 사이킷런)
    results = []
    print("\n🔬 모델 실험 시작...")
    
    for model_name, model in models.items():
        result = evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
        results.append(result)
    
    # 4️⃣ 결과 비교 분석 (pandas + MLflow)
    results_df = pd.DataFrame(results)
    print("\n📈 모델 비교 결과:")
    print(results_df.to_string(index=False))
    
    # 5️⃣ 최고 성능 모델 찾기
    best_model = results_df.loc[results_df['accuracy'].idxmax()]
    print(f"\n🏆 최고 성능 모델: {best_model['model_name']} (정확도: {best_model['accuracy']:.4f})")
    
    # 6️⃣ 비교 차트 생성 및 저장 (MLflow)
    with mlflow.start_run(run_name="Model_Comparison_Summary"):
        # 결과 테이블 저장
        results_df.to_csv("model_comparison.csv", index=False)
        mlflow.log_artifact("model_comparison.csv")
        
        # 비교 차트 생성
        plt.figure(figsize=(12, 8))
        
        # 서브플롯 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 정확도 비교
        axes[0, 0].bar(results_df['model_name'], results_df['accuracy'])
        axes[0, 0].set_title('모델별 정확도 비교')
        axes[0, 0].set_ylabel('정확도')
        
        # F1 점수 비교
        axes[0, 1].bar(results_df['model_name'], results_df['f1_score'])
        axes[0, 1].set_title('모델별 F1 점수 비교')
        axes[0, 1].set_ylabel('F1 점수')
        
        # CV 점수 비교 (에러바 포함)
        axes[1, 0].bar(results_df['model_name'], results_df['cv_mean'], 
                       yerr=results_df['cv_std'], capsize=5)
        axes[1, 0].set_title('모델별 Cross-Validation 점수')
        axes[1, 0].set_ylabel('CV 점수')
        
        # 종합 메트릭 히트맵
        metrics_data = results_df[['accuracy', 'precision', 'recall', 'f1_score']].T
        metrics_data.columns = results_df['model_name']
        sns.heatmap(metrics_data, annot=True, fmt='.3f', ax=axes[1, 1])
        axes[1, 1].set_title('모델별 메트릭 히트맵')
        
        plt.tight_layout()
        plt.savefig("model_comparison_charts.png", dpi=300, bbox_inches='tight')
        mlflow.log_artifact("model_comparison_charts.png")
        plt.close()
        
        # 요약 메트릭 기록
        mlflow.log_metric("best_accuracy", best_model['accuracy'])
        mlflow.log_metric("model_count", len(models))
        mlflow.set_tag("experiment_type", "model_comparison")
        mlflow.set_tag("best_model", best_model['model_name'])
    
    print(f"\n✅ 실험 완료! MLflow UI에서 결과를 확인하세요: http://localhost:8080")

if __name__ == "__main__":
    main() 