import numpy as np
import os
import joblib

def load_collector_model_scaler(scaler_path: str):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] 스케일러 파일이 없습니다: {scaler_path}")
    scaler = joblib.load(scaler_path)
    return scaler

def classify_item(data_point, scaler):
    # items_carried 문자열에서 item_count 계산
    items_carried = data_point.get('items_carried', '')
    if isinstance(items_carried, str):
        # 쉼표로 구분된 아이템 수 계산
        item_count = len(items_carried.split(',')) - 1
    else:
        item_count = 0
    data_point['item_count'] = item_count

    # LogItemUse_count 값 가져오기
    log_item_use_count = data_point.get('LogItemUse_count', 0)

    # win_place 값 가져오기 (스케일링 제외)
    win_place = data_point.get('win_place', 0)

    # item_count와 LogItemUse_count만 스케일러를 통해 정규화
    fields = np.array([[float(item_count), float(log_item_use_count)]])
    try:
        scaled = scaler.transform(fields)[0]
    except Exception as e:
        raise ValueError(f"Scaler 변환 오류: {e}")

    scaled_item_count = scaled[0]
    scaled_loguse_count = scaled[1]
    # win_place는 스케일링되지 않음

    # 정규화된 값과 로그회귀식 결과 계산
    try:
        result_count = scaled_item_count - (-0.4656 * np.log1p(win_place) + 1.2450)
        result_use = scaled_loguse_count - (-0.4670 * np.log1p(win_place) + 1.2345)
    except Exception as e:
        raise ValueError(f"로그 회귀식 계산 오류: {e}")

    total_result = result_count + result_use
    return total_result
