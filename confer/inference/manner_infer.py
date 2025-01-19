import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def load_svm_model_scaler_thresholds(
    model_path: str,
    scaler_path: str,
    thresholds_path: str
):
    """
    SVM 모델, 스케일러, Threshold (lower_threshold, upper_threshold) 로드.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일이 없습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] 스케일러 파일이 없습니다: {scaler_path}")
    if not os.path.exists(thresholds_path):
        raise FileNotFoundError(f"[ERROR] thresholds 파일이 없습니다: {thresholds_path}")

    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    thresholds = joblib.load(thresholds_path)

    lower_threshold = thresholds['lower_threshold']
    upper_threshold = thresholds['upper_threshold']

    return svm_model, scaler, lower_threshold, upper_threshold


def infer_with_thresholds(df_user: pd.DataFrame, svm_model, scaler, lower_thr, upper_thr):
    """
    df_user(1행 또는 소량의 행)에 대해:
      1) 파생변수(team_kill_ratio, road_kill_ratio, vehicle_destroy_ratio) 생성
      2) SVM 결정 함수 계산
      3) lower_thr, upper_thr에 의해 라벨링 ('none-manner', 'manner', 'unlabeled')

    - df_user에는 'kills', 'team_kills', 'road_kills', 'vehicle_destroys' 필드가 있어야 함.
    """
    needed_cols = ['kills', 'team_kills', 'road_kills', 'vehicle_destroys']
    for col in needed_cols:
        if col not in df_user.columns:
            raise KeyError(f"[ERROR] '{col}' 컬럼이 없습니다: {col}")

    # 파생변수 생성
    df_user = df_user.copy()  # 원본 df 보호
    df_user['team_kill_ratio'] = df_user['team_kills'] / (df_user['kills'] + 1)
    df_user['road_kill_ratio'] = df_user['road_kills'] / (df_user['kills'] + 1)
    df_user['vehicle_destroy_ratio'] = df_user['vehicle_destroys'] / (df_user['kills'] + 1)

    features = ['team_kill_ratio', 'road_kill_ratio', 'vehicle_destroy_ratio']
    X = df_user[features].to_numpy()

    # 만약 파생변수의 값이 모두 0이면 'manner'로 일괄 분류
    if np.all(X == 0):
        decision_scores = np.zeros(len(df_user))
        final_labels = np.full(len(df_user), 'manner')
        result_df = df_user.copy()
        result_df['Decision_Score'] = decision_scores
        result_df['Final_Label'] = final_labels
        return result_df

    # 스케일링
    X_scaled = scaler.transform(X)

    # 결정 함수 계산
    decision_scores = svm_model.decision_function(X_scaled)

    # 최종 라벨링
    # scores > upper_thr => 'manner'
    # scores < lower_thr => 'none-manner'
    # 그 외 => 'unlabeled'
    final_labels = np.where(
        decision_scores > upper_thr, 'manner',
        np.where(decision_scores < lower_thr, 'no-manner', 'unlabeled')
    )

    # 결과 df
    result_df = df_user.copy()
    result_df['Decision_Score'] = decision_scores
    result_df['Final_Label'] = final_labels

    return result_df
