import os
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm  # tqdm 임포트
from sklearn.ensemble import RandomForestClassifier

# RandomForestClassifier 모델 로드
def load_rf_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] RandomForestClassifier 모델 파일이 없습니다: {model_path}")
    rf_model = joblib.load(model_path)
    return rf_model

# 탐험가 데이터 전처리 함수
def explorer_data_preprocessing(df, location_col='first_location_x', name_col='player_name'):
    df = df[df[location_col] != 'None']
    df = df[~df[name_col].isnull()]
    return df

# 좌표를 분리하는 함수 정의
def split_coordinates(trajectory):
    if not trajectory or not isinstance(trajectory, str):  # 유효성 검사
        return []
    try:
        points = trajectory.split(" -> ")
        coordinates = [tuple(map(float, point.strip("()").split(","))) for point in points]
        return coordinates
    except ValueError:
        return []

# 특성 생성 함수
def feature_generation(df):
    # 좌표 분리 및 확장
    coordinate_list = df["movement_routes"].apply(split_coordinates)
    max_points = max(coordinate_list.apply(len)) if not coordinate_list.empty else 0

    column_names = []
    for i in tqdm(range(1, max_points + 1), desc="Generating column names"):
        column_names.extend([f"x{i}", f"y{i}", f"z{i}"])

    expanded_df = pd.DataFrame(coordinate_list.tolist()).apply(
        lambda row: pd.Series([v for point in row if point for v in point]),
        axis=1
    )
    expanded_df.columns = column_names[:expanded_df.shape[1]]

    # 원본 데이터프레임과 확장된 좌표 데이터 병합
    result = pd.concat([df.reset_index(drop=True), expanded_df.reset_index(drop=True)], axis=1)

    # 거리 합산
    result['total_distance'] = result['walk_distance'] + result['ride_distance'] + result['swim_distance']

    # 좌표를 기반으로 분당 이동 거리 합산 계산
    coordinate_columns = [col for col in result.columns if col.startswith(('x', 'y', 'z'))]
    coords = result[coordinate_columns].values.reshape(len(result), -1, 3)
    coords = np.nan_to_num(coords)
    distances = np.sqrt(np.sum(np.diff(coords, axis=1) ** 2, axis=2))
    result['total_movement_distance'] = distances.sum(axis=1)

    # 플레이어별 탐험한 맵의 개수 계산 및 'unique_maps' 칼럼 생성
    map_diversity = result.groupby('player_name')['map_name'].nunique().reset_index(name='unique_maps')
    result = result.merge(map_diversity, on='player_name', how='left')

    # 'unique_maps' 컬럼이 없는 경우 기본값 0 설정
    if 'unique_maps' not in result.columns:
        result['unique_maps'] = 1

    return result

# RandomForest를 활용한 탐험가 유형 예측 함수
def infer_with_rf(df_user: pd.DataFrame, rf_model):
    needed_cols = ['walk_distance', 'ride_distance', 'swim_distance', 'map_name', 'player_name']
    for col in needed_cols:
        if col not in df_user.columns:
            raise KeyError(f"[ERROR] '{col}' 컬럼이 없습니다: {col}")

    # 파생변수 생성
    df_user = df_user.copy()  # 원본 df 보호
    df_user['total_distance'] = df_user['walk_distance'] + df_user['ride_distance'] + df_user['swim_distance']

    # 플레이어별 탐험한 맵의 개수 계산 및 'unique_maps' 칼럼 생성
    map_diversity = df_user.groupby('player_name')['map_name'].nunique().reset_index(name='unique_maps')
    df_user = df_user.merge(map_diversity, on='player_name', how='left')

    # 'unique_maps' 컬럼이 없는 경우 기본값 0 설정
    if 'unique_maps' not in df_user.columns:
        df_user['unique_maps'] = 0

    # 거리 이동 계산
    coordinate_columns = [col for col in df_user.columns if col.startswith(('x', 'y', 'z'))]
    coords = df_user[coordinate_columns].values.reshape(len(df_user), -1, 3)
    coords = np.nan_to_num(coords)
    distances = np.sqrt(np.sum(np.diff(coords, axis=1) ** 2, axis=2))
    df_user['total_movement_distance'] = distances.sum(axis=1)

    # 분류 예측
    rf_features = ['total_distance', 'unique_maps', 'total_movement_distance']
    X = df_user[rf_features].to_numpy()
    final_labels = rf_model.predict(X)

    # 결과 생성
    result_df = df_user.copy()
    result_df['Final_Label'] = final_labels

    return result_df
