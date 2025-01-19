import pandas as pd
import joblib
import numpy as np

def load_model_and_scaler(model_path: str, scaler_path: str):
    """
    SVM 모델과 Scaler를 로드합니다.
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception as e:
        raise RuntimeError(f"[ERROR] 모델 또는 스케일러 로드 실패: {e}")


def infer_aggression(df_user: pd.DataFrame, model, scaler):
    """
    개별 행 데이터를 사용해 공격성(Aggression)을 예측합니다.
    """
    selected_columns = ['kills', 'kill_streaks', 'headshot_kills', 'damage_dealt', 'time_spent_in_combat_sec']
    
    for col in selected_columns:
        if col not in df_user.columns:
            raise KeyError(f"[ERROR] '{col}' 컬럼이 없습니다.")

    try:
        data_scaled = scaler.transform(df_user[selected_columns])

        predicted_labels = model.predict(data_scaled)
        predicted_probabilities = model.predict_proba(data_scaled)

        result_df = df_user.copy()
        result_df['Predicted_Label'] = np.where(predicted_labels == 0, 'Passive', 'Aggressive')
        result_df['Aggressive_Probability'] = predicted_probabilities[:, 1]
        result_df['Passive_Probability'] = predicted_probabilities[:, 0]

        return result_df
    except Exception as e:
        raise RuntimeError(f"[ERROR] 예측 중 오류 발생: {e}")