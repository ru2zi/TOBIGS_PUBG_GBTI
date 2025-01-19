import os
import numpy as np
import pandas as pd
import plotly.express as px

# 이미 작성된 inference 모듈 임포트 (경로/이름은 상황에 맞게)
import killer_infer
import social_infer
import explorer_infer
import collector_infer
import manner_infer

from tqdm import tqdm

# 1) 각 모델/스케일러/thresholds 파일 경로 지정
KILLER_MODEL_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\killer\svm_model.pkl'
KILLER_SCALER_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\killer\scaler.pkl'

SOCIAL_MODEL_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\social\group_model_svm.pkl'
SOCIAL_SCALER_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\social\group_scaler.pkl'

EXPLORER_MODEL_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\explorer\explorer_model.pkl'

COLLECTOR_SCALER_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\collector\scaler.pkl'

MANNER_MODEL_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\manner\svm_model.pkl'
MANNER_SCALER_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\manner\scaler.pkl'
MANNER_THRESH_PATH = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\confer\model\manner\thresholds.pkl'


def main():
    # 2) player_data_event_details.csv 로드
    csv_path = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\output\player_data_event_details.csv'
    df = pd.read_csv(csv_path, encoding='utf-8', low_memory=False)
    df.drop_duplicates(inplace=True)  # 중복 제거 (선택사항)

    # 3) 각 모델/스케일러 로드
    killer_model, killer_scaler = killer_infer.load_model_and_scaler(
        KILLER_MODEL_PATH, KILLER_SCALER_PATH
    )
    social_model, social_scaler = social_infer.load_svm_model_scaler(
        SOCIAL_MODEL_PATH, SOCIAL_SCALER_PATH
    )
    explorer_rf_model = explorer_infer.load_rf_model(EXPLORER_MODEL_PATH)
    collector_scaler = collector_infer.load_collector_model_scaler(COLLECTOR_SCALER_PATH)
    manner_svm_model, manner_scaler, manner_low_thr, manner_up_thr = (
        manner_infer.load_svm_model_scaler_thresholds(
            MANNER_MODEL_PATH, MANNER_SCALER_PATH, MANNER_THRESH_PATH
        )
    )

    # 결과 저장용 리스트
    killer_types = []
    social_types = []
    explorer_types = []
    collector_types = []
    manner_types = []

    # 4) 모든 행(플레이어 매치 데이터)에 대해 반복
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="GBTI Inference"):
        # row -> DataFrame 형태 (1행)
        single_df = pd.DataFrame([row])  # (1, N) 형태

        # ---- Killer Inference (A/P) ----
        try:
            killer_result = killer_infer.infer_aggression(single_df, killer_model, killer_scaler)
            # 'Predicted_Label': Passive/Aggressive
            k_label = killer_result['Predicted_Label'].iloc[0]
            k_type = "A" if k_label == "Aggressive" else "P"
        except:
            k_type = "P"  # 예외 시 'P' 등으로 처리

        # ---- Social Inference (I/G) ----
        try:
            # 이 예시에서는 telemetry_data 미사용 / 빈 list
            social_inference_df = social_infer.infer_with_model(
                single_df,
                telemetry_data=[],  # 실제 telemetry가 없으니 빈 list
                svm_model=social_model,
                scaler=social_scaler
            )
            s_label = social_inference_df['Final_Label'].iloc[0]  # 'individual' or 'group'
            s_type = "G" if s_label == "group" else "I"
        except:
            s_type = "I"

        # ---- Explorer Inference (E/S) ----
        try:
            single_explorer_df = explorer_infer.feature_generation(single_df)
            explorer_df = explorer_infer.infer_with_rf(single_explorer_df, explorer_rf_model)
            e_label = explorer_df['Final_Label'].iloc[0]  # 1 or 0
            e_type = "E" if e_label == 1 else "S"
        except:
            e_type = "S"

        # ---- Collector Inference (C/M) ----
        try:
            data_point = row.to_dict()
            total_result = collector_infer.classify_item(data_point, collector_scaler)
            if total_result > 0:
                c_type = 'C'
            elif total_result < 0:
                c_type = 'M'
            else:
                c_type = 'X'  # or 'Unlabeled'
        except:
            c_type = "X"

        # ---- Manner Inference (m/n) ----
        try:
            manner_result = manner_infer.infer_with_thresholds(
                single_df, manner_svm_model, manner_scaler,
                manner_low_thr, manner_up_thr
            )
            m_label = manner_result['Final_Label'].iloc[0]  # 'manner', 'no-manner', or 'unlabeled'
            if m_label == "manner":
                m_type = "m"
            elif m_label == "no-manner":
                m_type = "n"
            else:
                m_type = "x"
        except:
            m_type = "x"

        killer_types.append(k_type)
        social_types.append(s_type)
        explorer_types.append(e_type)
        collector_types.append(c_type)
        manner_types.append(m_type)

    # 5) Inference 결과를 df에 붙이기
    df['KillerType'] = killer_types
    df['SocialType'] = social_types
    df['ExplorerType'] = explorer_types
    df['CollectorType'] = collector_types
    df['MannerType'] = manner_types

    # 6) GBTI composite label
    #    ex) "PIGC-n"
    #    killer + social + explorer + collector + "-" + manner
    composite_labels = []
    for k, s, e, c, m in zip(killer_types, social_types, explorer_types, collector_types, manner_types):
        composite_labels.append(f"{k}{s}{e}{c}-{m}")
    df['GBTI'] = composite_labels

    # 7) GBTI 분포 시각화 (Plotly)
    #    => match_id / player_name / GBTI 만을 저장 (CSV)
    freq_series = df['GBTI'].value_counts().reset_index()
    freq_series.columns = ['GBTI_Label', 'count']

    # bar 차트 예시
    fig_bar = px.bar(
        freq_series,
        x='GBTI_Label',
        y='count',
        title="GBTI 유형별 분포 (Bar)",
        text='count'
    )
    fig_bar.update_layout(xaxis={'categoryorder': 'total descending'})
    fig_bar.show()

    # bar 차트를 파일로 저장 (HTML, PNG)
    fig_bar.write_html("gbti_bar_chart.html")
    fig_bar.write_image("gbti_bar_chart.png")

    # pie 차트 예시
    fig_pie = px.pie(
        freq_series,
        names='GBTI_Label',
        values='count',
        title="GBTI 유형별 분포 (Pie)"
    )
    fig_pie.show()

    # pie 차트를 파일로 저장 (HTML, PNG)
    fig_pie.write_html("gbti_pie_chart.html")
    fig_pie.write_image("gbti_pie_chart.png")

    # 8) CSV로 match_id / player_name / GBTI 만 저장
    #    중복 제거 여부는 선택사항
    df_subset = df[['match_id', 'player_name', 'GBTI']].drop_duplicates()

    # 파일 이름 등은 원하는 대로
    out_path = r"C:\Users\inho0\OneDrive\문서\GitHub\PUBG\output\gbti_subset.csv"
    df_subset.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"[INFO] GBTI Inference 완료 -> {out_path} 에 match_id/player_name/GBTI 저장")


if __name__ == "__main__":
    main()
