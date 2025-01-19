import os
import joblib
import numpy as np
import pandas as pd
import re
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# SVM 모델과 스케일러 로드 함수
def load_svm_model_scaler(
    model_path: str,
    scaler_path: str,
):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[ERROR] 모델 파일이 없습니다: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"[ERROR] 스케일러 파일이 없습니다: {scaler_path}")

    svm_model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    return svm_model, scaler

def euclidean_distance(player_pos, team_mean_pos):
    """
    두 점 사이의 유클리드 거리 계산.
    """
    return np.sqrt(np.sum((player_pos - team_mean_pos) ** 2))

def calculate_team_distances(team_df):
    """
    팀 데이터프레임을 입력받아 각 플레이어의 이동 거리와 팀 평균 이동 거리 간의 차이를 계산.
    """
    min_length = team_df['x_coords'].apply(len).min()

    team_mean_positions = []
    for i in range(min_length):
        mean_x = np.mean([x[i] for x in team_df['x_coords']])
        mean_y = np.mean([y[i] for y in team_df['y_coords']])
        mean_z = np.mean([z[i] for z in team_df['z_coords']])
        team_mean_positions.append([mean_x, mean_y, mean_z])
    team_mean_positions = np.array(team_mean_positions)

    distances = []
    for _, row in team_df.iterrows():
        player_positions = np.array([
            [row['x_coords'][i], row['y_coords'][i], row['z_coords'][i]]
            for i in range(min_length)
        ])
        distance = np.mean([
            euclidean_distance(player_pos, team_mean_positions[i])
            for i, player_pos in enumerate(player_positions)
        ]) / min_length
        distances.append(distance)

    return distances

def infer_with_model(df_user: pd.DataFrame, telemetry_data, svm_model, scaler):
    """
    단일 유저 데이터프레임 (1행)과 telemetry_data를 사용해 SVM 추론 수행.
    """
    needed_cols = ['match_id', 'team_id', 'assists', 'revives',
                   'first_location_x', 'first_location_y', 'first_location_z', 'movement_routes']
    for col in needed_cols:
        if col not in df_user.columns:
            raise KeyError(f"[ERROR] '{col}' 컬럼이 없습니다: {col}")

    user = df_user.iloc[0]
    match_id = user['match_id']
    team_id = user['team_id']

    # teamId를 활용하여 동일 팀의 telemetry 이벤트 필터링
    team_data = [
        event for event in telemetry_data
        if event.get("character", {}).get("teamId") == team_id
    ]

    if not team_data:
        raise ValueError(f"[ERROR] match_id={match_id}, team_id={team_id}에 해당하는 팀 데이터를 찾을 수 없습니다.")

    team_df = pd.DataFrame([
        {
            "player_id": event["character"]["name"],
            "first_location_x": event["character"]["location"]["x"],
            "first_location_y": event["character"]["location"]["y"],
            "first_location_z": event["character"]["location"]["z"],
            "movement_routes": event["character"].get("movement_routes", "")
        }
        for event in team_data
    ])

    team_avg_x = team_df['first_location_x'].mean()
    team_avg_y = team_df['first_location_y'].mean()
    team_avg_z = team_df['first_location_z'].mean()

    user_position = np.array([user['first_location_x'], user['first_location_y'], user['first_location_z']])
    team_avg_position = np.array([team_avg_x, team_avg_y, team_avg_z])
    first_location_to_team = euclidean_distance(user_position, team_avg_position)

    def extract_coordinates(movement_route):
        pattern = r'\(([-\d\.]+),\s*([-\d\.]+),\s*([-\d\.]+)\)'
        coordinates = re.findall(pattern, movement_route)
        x_coords = [float(x) for x, _, _ in coordinates]
        y_coords = [float(y) for _, y, _ in coordinates]
        z_coords = [float(z) for _, _, z in coordinates]
        return x_coords, y_coords, z_coords

    team_df[['x_coords', 'y_coords', 'z_coords']] = team_df['movement_routes'].apply(
        lambda route: pd.Series(extract_coordinates(route))
    )

    user_coords = extract_coordinates(user['movement_routes'])
    user_df = pd.DataFrame({
        'x_coords': [user_coords[0]],
        'y_coords': [user_coords[1]],
        'z_coords': [user_coords[2]]
    })

    full_team_df = pd.concat([team_df, user_df], ignore_index=True)

    distances = calculate_team_distances(full_team_df)
    location_to_team = distances[-1]

    df_user = df_user.copy()
    df_user['first_location_to_team'] = first_location_to_team
    df_user['location_to_team'] = location_to_team

    svm_features = ['assists', 'revives', 'first_location_to_team', 'location_to_team']
    X = df_user[svm_features].to_numpy()

    # NaN 값을 0으로 대체하여 SVM이 NaN 값을 받지 않도록 처리
    X = np.nan_to_num(X)

    X_scaled = scaler.transform(X)

    final_labels = svm_model.predict(X_scaled)

    result_df = df_user.copy()
    result_df['Final_Label'] = np.where(final_labels == 0, 'individual', 'group')

    return result_df
