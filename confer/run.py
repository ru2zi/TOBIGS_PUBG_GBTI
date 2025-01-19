import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st
import aiohttp
import asyncio
import time
from datetime import datetime
import openai

from inference import manner_infer, social_infer, explorer_infer, collector_infer, killer_infer
from inference.killer_infer import load_model_and_scaler, infer_aggression
from chicken_dinner.pubgapi import PUBG
from chicken_dinner.constants import map_dimensions
from data_inmemory import generate_single_csv_line_in_memory
import pubg_fetch

#########################################################
# 1) 환경 설정
#########################################################

# 현재 파일의 디렉토리를 기준으로 BASE_DIR 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# API_KEY = os.getenv("PUBG_API_KEY")
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# st.secrets를 통해 비밀 정보 불러오기
API_KEY = st.secrets["PUBG_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# OpenAI API 키 설정
openai.api_key = OPENAI_API_KEY

if not API_KEY:
    st.error("PUBG API 키를 불러오지 못했습니다.")
    st.stop()

try:
    pubg = PUBG(API_KEY, shard='kakao')
except Exception as e:
    st.error(f"PUBG API 초기화 중 오류 발생: {e}")
    st.stop()

#########################################################
# 2) Streamlit UI 스타일
#########################################################
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f0f5;
        padding: 1rem 2rem;
    }
    .title-font {
        font-size: 42px !important;
        font-weight: 700 !important;
        color: #333333;
        margin-bottom: 1rem;
    }
    .centered {
        text-align: center;
    }
    .button-col {
        border-left: 1px solid #ccc;
        padding-left: 10px;
    }
    /* 각 버튼별 색상 설정 */
    .killer-btn button {
        background-color: #FFA8E0;
        color: #fff;
    }
    .killer-btn button:hover {
        filter: brightness(0.9);
    }
    .social-btn button {
        background-color: #FF6666;
        color: #fff;
    }
    .social-btn button:hover {
        filter: brightness(0.9);
    }
    .explorer-btn button {
        background-color: #66B2FF;
        color: #fff;
    }
    .explorer-btn button:hover {
        filter: brightness(0.9);
    }
    .collector-btn button {
        background-color: #66FF66;
        color: #333;
    }
    .collector-btn button:hover {
        filter: brightness(0.9);
    }
    .manner-btn button {
        background-color: #FFCC66;
        color: #333;
    }
    .manner-btn button:hover {
        filter: brightness(0.9);
    }
    div[data-testid="stProgressBar"] > div > div > div {
        height: 20px;
    }
    .title-container {
        text-align: center;
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# 세션 상태 초기화
for key in [
    'match_list', 'df_result', 'inference_df', 'social_inference',
    'explorer_inference', 'collector_inference', 'killer_inference',
    'telemetry_data', 'roster_data',
    'socialiser_type', 'explorer_type', 'collector_type', 'manner_type', 'killer_type'
]:
    if key not in st.session_state:
        st.session_state[key] = None if 'type' in key or key in ['df_result', 'inference_df'] else []

#########################################################
# 3) 게임 상세/맵 시각화 관련 함수들
#########################################################
def get_death_reason(death_type: str) -> str:
    mapping = {
        'byplayer': '다른 플레이어에 의해 사망',
        'byzone': '안전 구역 밖에 나가서 사망',
        'suicide': '자살',
        'other': '기타'
    }
    return mapping.get(death_type.lower(), '알 수 없는 사망 원인')

def extract_landing_locations(telemetry, mapy):
    try:
        start_time_str = telemetry.started()
        start_time = pd.to_timedelta(start_time_str[start_time_str.find('T') + 1 : -5])
        unequips = telemetry.filter_by('log_item_unequip')

        landing_data = {}
        for unequip in unequips:
            if unequip['item']['item_id'] == 'Item_Back_B_01_StartParachutePack_C':
                char = unequip['character']
                name_ = char['name']
                x_ = char['location']['x']
                y_ = mapy - char['location']['y']
                t_ = pd.to_timedelta(unequip.timestamp[unequip.timestamp.find('T') + 1 : -5]) - start_time
                team_id_ = char['team_id']
                landing_data[name_] = (x_, y_, t_.total_seconds(), team_id_)

        df_landing = pd.DataFrame(landing_data).T.reset_index()
        df_landing.columns = ['name','x','y','time','teamId']
        df_landing['teamId'] = df_landing['teamId'].astype('int64')
        df_landing['teamName'] = df_landing['name'].str.extract(r'([0-9A-Za-z]+)_')
        return df_landing

    except Exception as e:
        st.warning(f"낙하 위치 추출 오류: {e}")
        return pd.DataFrame(columns=['name','x','y','time','teamId','teamName'])

async def fetch_data(session, url, headers):
    try:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                return await response.json()
            else:
                st.error(f"데이터 요청 실패: {url}, 상태코드={response.status}")
                return None
    except Exception as e:
        st.error(f"데이터 요청 중 오류: {e}")
        return None

async def fetch_match_ids_with_details(session, player_name):
    base_url = "https://api.pubg.com/shards/kakao/players"
    url = f"{base_url}?filter[playerNames]={player_name}"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Accept": "application/vnd.api+json"
    }
    user_data = await fetch_data(session, url, headers)
    if not user_data:
        return []

    try:
        data_block = user_data["data"][0]
        matches_ = data_block["relationships"]["matches"]["data"]
        details = []
        for m in matches_:
            mid = m["id"]
            match_url = f"https://api.pubg.com/shards/kakao/matches/{mid}"
            match_obj = await fetch_data(session, match_url, headers)
            if match_obj:
                created_at = match_obj["data"]["attributes"]["createdAt"]
                details.append({"id": mid, "time": created_at})
        return details
    except:
        return []

def visualize_pubg_map(user: str, match_id: str):
    from dateutil.parser import parse as parse_iso8601
    try:
        with st.spinner("Loading match data..."):
            progress_bar = st.progress(0)
            time.sleep(0.3)
            progress_bar.progress(10)

            current_match = pubg.match(match_id)
            telemetry = current_match.get_telemetry()

            user_kills = user_assists = user_deaths = 0
            death_reason = '알 수 없는 사망 원인'
            for participant in current_match.participants:
                if participant.name.lower() == user.lower():
                    stt = participant.stats
                    user_kills = stt.get('kills', 0)
                    user_assists = stt.get('assists', 0)
                    dt = stt.get('death_type','alive')
                    user_deaths = 1 if dt != 'alive' else 0
                    death_reason = get_death_reason(dt)
                    break
            user_kda = (user_kills + user_assists) / max(1, user_deaths)
            progress_bar.progress(30)

            map_id = telemetry.map_id()
            map_name = telemetry.map_name()
            mapx, mapy = map_dimensions[map_id]
            circles = telemetry.circle_positions()
            positions = telemetry.player_positions()

            landing_locations = extract_landing_locations(telemetry, mapy)
            winners = telemetry.winner()

            USER_COLORS = {winner: 'bgcmyk'[i % 6] for i, winner in enumerate(winners)}
            USER_COLORS[user] = 'r'

            whites = np.array(circles['white'])
            whites[:,2] = mapy - whites[:,2]
            phases = np.where(whites[1:,4] - whites[:-1,4] != 0)[0] + 1

            num_teams = telemetry.num_teams()
            num_players = telemetry.num_players()
            match_length = telemetry.match_length()
            match_start_time = telemetry.started()
            rankings = telemetry.rankings()

            user_team_rank = None
            for rank, p_list in rankings.items():
                if user in p_list:
                    user_team_rank = rank
                    break

            progress_bar.progress(50)

            match_start_dt = parse_iso8601(match_start_time.split('+')[0])
            formatted_time = match_start_dt.strftime("%Y년 %m월 %d일, %H시 %M분 %S초")

            st.write(f"**맵 이름**: {map_name}")
            st.write(f"**총 플레이어 수**: {num_players}")
            st.write(f"**매치 시작 시간**: {formatted_time}")
            st.write(f"**매치 길이**: {match_length//60}분 {match_length%60}초")

            if user_team_rank:
                st.write(f"**{user}의 팀 랭킹**: {user_team_rank}/{num_teams}")

            st.write(f"**{user}의 KDA**:")
            st.write(f"- 킬: {user_kills}")
            st.write(f"- 어시스트: {user_assists}")
            st.write(f"- 데스: {user_deaths}")
            st.write(f"- KDA: {user_kda:.2f}")
            st.write(f"- 사망 원인: {death_reason}")

            progress_bar.progress(70)

            fig = plt.figure(figsize=(10,10), dpi=100)
            ax = fig.add_axes([0,0,1,1])
            ax.axis('off')

            # 상대 경로로 이미지 파일 위치 지정
            img_path = os.path.join(BASE_DIR, "map_image", f"{map_id}_High_Res.png")
            if not os.path.isfile(img_path):
                st.warning(f"맵 이미지 파일이 없음: {img_path}")
                return

            img = mpimg.imread(img_path)
            ax.imshow(img, extent=[0, mapx, 0, mapy])

            for i, ph in enumerate(phases):
                white_circle = plt.Circle(
                    (whites[ph][1], whites[ph][2]),
                    whites[ph][4],
                    edgecolor="w",
                    linewidth=0.7,
                    fill=False,
                    zorder=5
                )
                ax.add_patch(white_circle)
                progress_bar.progress(int(70 + (i+1)/len(phases)*20))
                time.sleep(0.1)

            for user_name, color_ in USER_COLORS.items():
                if user_name not in positions:
                    st.warning(f"{user_name} position data 없음. pass..")
                    continue
                cpos = np.array(positions[user_name])
                cpos[:,2] = mapy - cpos[:,2]
                if user_name in landing_locations['name'].values:
                    row_df = landing_locations[landing_locations['name'] == user_name]
                    if row_df.empty:
                        st.warning(f"Landing data for {user_name} is missing. skip")
                        continue
                    t0 = row_df['time'].values[0]
                    cpos = cpos[cpos[:,0] > t0]
                ax.plot(cpos[:,1], cpos[:,2], '--', c=color_, linewidth=2, zorder=20, label=user_name)
            ax.legend(loc='upper right', title="Users")
            progress_bar.progress(100)
            st.pyplot(fig)

    except Exception as ex:
        st.error(f"시각화 중 오류: {ex}")

#########################################################
# 4) main() 함수
#########################################################
async def main():
    st.markdown(
        '<div class="title-container"><h1 class="title-font centered">PUBG Match Data Viewer</h1></div>',
        unsafe_allow_html=True
    )

    player_name = st.text_input("플레이어 이름을 입력하세요 (예: coffee1000won):")
    if not player_name:
        st.stop()

    if st.button("매치 목록 가져오기"):
        async with aiohttp.ClientSession() as session:
            match_details = await fetch_match_ids_with_details(session, player_name)
        if not match_details:
            st.error("플레이어 매치를 못찾음.")
            return
        st.session_state['match_list'] = match_details
        st.success("매치 목록 가져옴.")

    if 'match_list' not in st.session_state or not st.session_state['match_list']:
        st.session_state['match_list'] = []
    match_options = [f"{m['id']} ({m['time']})" for m in st.session_state['match_list']]
    selected_match_str = st.selectbox("매치를 선택하세요:", match_options)

    if selected_match_str:
        match_id = selected_match_str.split(" ")[0]

        if st.button("Visualize Match"):
            visualize_pubg_map(user=player_name, match_id=match_id)

        if st.button("이 매치를 CSV로 저장"):
            with st.spinner("메모리에 match 데이터 로딩중..."):
                async with aiohttp.ClientSession() as session:
                    match_obj = await pubg_fetch.get_match_in_memory(session, match_id, API_KEY)
                if not match_obj:
                    st.error("match data 불러오기 실패.")
                    return

                meta_data = match_obj["meta_data"]
                players_data = match_obj["players_data"]
                roster_data = match_obj["roster_data"]
                telemetry_data = match_obj["telemetry_data"]

                st.session_state.telemetry_data = telemetry_data
                st.session_state.roster_data = roster_data

                df_result, msg = generate_single_csv_line_in_memory(
                    meta_data=meta_data,
                    players_data=players_data,
                    telemetry_data=telemetry_data,
                    roster_data=roster_data,
                    user_name=player_name,
                    match_id=match_id
                )
            if df_result is not None:
                st.success(msg)
                st.dataframe(df_result)
                st.session_state.df_result = df_result
            else:
                st.error(msg)

    if st.session_state.df_result is not None:
        st.subheader("유저의 해당 매치 Data")
        st.dataframe(st.session_state.df_result)

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("<div class='killer-btn centered button-col'>", unsafe_allow_html=True)
            if st.button("Killer(A/P)"):
                model_dir = os.path.join(BASE_DIR, "model", "killer")
                KILLER_MODEL_PATH = os.path.join(model_dir, "svm_model.pkl")
                KILLER_SCALER_PATH = os.path.join(model_dir, "scaler.pkl")

                try:
                    killer_model, killer_scaler = load_model_and_scaler(KILLER_MODEL_PATH, KILLER_SCALER_PATH)
                    killer_result = infer_aggression(st.session_state.df_result.iloc[[0]], killer_model, killer_scaler)

                    killer_label = killer_result['Predicted_Label'].iloc[0]
                    st.markdown(f"<div class='centered'><strong>{killer_label}</strong></div>", unsafe_allow_html=True)

                    st.session_state.killer_type = "A" if killer_label == "Aggressive" else "P"
                except Exception as e:
                    st.error(f"Killer Inference 오류: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("<div class='social-btn centered button-col'>", unsafe_allow_html=True)
            if st.button("Social(I/G)"):
                model_dir = os.path.join(BASE_DIR, "model", "social")
                SOCIAL_MODEL_PATH = os.path.join(model_dir, "group_model_svm.pkl")
                SOCIAL_SCALER_PATH = os.path.join(model_dir, "group_scaler.pkl")
                try:
                    social_model, social_scaler = social_infer.load_svm_model_scaler(
                        SOCIAL_MODEL_PATH, SOCIAL_SCALER_PATH
                    )
                    social_inference_df = social_infer.infer_with_model(
                        st.session_state.df_result,
                        st.session_state.telemetry_data,
                        social_model,
                        social_scaler
                    )
                    social_inference_df['Final_Label'] = social_inference_df['Final_Label'].map({
                        'individual': 'Individualism',
                        'group': 'Group'
                    })
                    st.session_state.social_inference = social_inference_df
                    social_label = social_inference_df['Final_Label'].iloc[0]
                    st.markdown(f"<div class='centered'><strong>{social_label}</strong></div>", unsafe_allow_html=True)
                    st.session_state.socialiser_type = "G" if social_label == "Group" else "I"
                except Exception as e:
                    st.error(f"Social Inference 오류: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col3:
            st.markdown("<div class='explorer-btn centered button-col'>", unsafe_allow_html=True)
            if st.button("Explorer(E/S)"):
                try:
                    model_dir = os.path.join(BASE_DIR, "model", "explorer")
                    EXPLORER_MODEL_PATH = os.path.join(model_dir, "explorer_model.pkl")
                    rf_model = explorer_infer.load_rf_model(EXPLORER_MODEL_PATH)
                    processed_df = explorer_infer.feature_generation(st.session_state.df_result)
                    explorer_df = explorer_infer.infer_with_rf(processed_df, rf_model)
                    st.session_state.explorer_inference = explorer_df
                    explorer_label = explorer_df['Final_Label'].iloc[0]
                    explorer_full = "Explorer" if explorer_label == 1 else "Settler"
                    st.markdown(f"<div class='centered'><strong>{explorer_full}</strong></div>", unsafe_allow_html=True)
                    st.session_state.explorer_type = "E" if explorer_label == 1 else "S"
                except Exception as e:
                    st.error(f"Explorer Inference 오류: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col4:
            st.markdown("<div class='collector-btn centered button-col'>", unsafe_allow_html=True)
            if st.button("Collector(C/M)"):
                model_dir = os.path.join(BASE_DIR, "model", "collector")
                COLLECTOR_SCALER_PATH = os.path.join(model_dir, "scaler.pkl")
                try:
                    collector_scaler = collector_infer.load_collector_model_scaler(COLLECTOR_SCALER_PATH)
                    data_point = st.session_state.df_result.iloc[0].to_dict()
                    total_result = collector_infer.classify_item(data_point, collector_scaler)

                    if total_result > 0:
                        collector_label = 'Collector'
                    elif total_result < 0:
                        collector_label = 'Minimalist'
                    else:
                        collector_label = 'Unlabeled'

                    st.markdown(f"<div class='centered'><strong>{collector_label}</strong></div>", unsafe_allow_html=True)
                    st.session_state.collector_type = "C" if collector_label == "Collector" else "M"
                except Exception as e:
                    st.error(f"Collector Inference 오류: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        with col5:
            st.markdown("<div class='manner-btn centered'>", unsafe_allow_html=True)
            if st.button("Manner(m/n)"):
                model_dir = os.path.join(BASE_DIR, "model", "manner")
                MODEL_PATH = os.path.join(model_dir, "svm_model.pkl")
                SCALER_PATH = os.path.join(model_dir, "scaler.pkl")
                THRESH_PATH = os.path.join(model_dir, "thresholds.pkl")
                try:
                    svm_model, scaler, low_thr, up_thr = manner_infer.load_svm_model_scaler_thresholds(
                        MODEL_PATH, SCALER_PATH, THRESH_PATH
                    )
                    inference_df = manner_infer.infer_with_thresholds(
                        st.session_state.df_result, svm_model, scaler, low_thr, up_thr
                    )
                    st.session_state.inference_df = inference_df
                    manner_label = inference_df['Final_Label'].iloc[0]
                    manner_full = "Manner" if manner_label == "manner" else "No-Manner"
                    st.markdown(f"<div class='centered'><strong>{manner_full}</strong></div>", unsafe_allow_html=True)
                    st.session_state.manner_type = "m" if manner_label == "manner" else "n"
                except Exception as e:
                    st.error(f"Inference 오류: {e}")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")

        with st.container():
            st.markdown(
                """
                <style>
                .center-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                    text-align: center;
                }
                .centered {
                    text-align: center;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="center-container">', unsafe_allow_html=True)
            if st.button("당신의 GBTI 유형 보기"):
                required_keys = ['socialiser_type', 'explorer_type', 'collector_type', 'manner_type', 'killer_type']
                if all(key in st.session_state and st.session_state[key] is not None for key in required_keys):
                    type_mapping = {
                        "G": "Group",
                        "I": "Individualism",
                        "E": "Explorer",
                        "S": "Settler",
                        "C": "Collector",
                        "M": "Minimalist",
                        "A": "Aggressive",
                        "P": "Passive",
                        "m": "Manner",
                        "n": "No-Manner"
                    }
                    social_full = type_mapping.get(st.session_state.socialiser_type, "Unknown")
                    explorer_full = type_mapping.get(st.session_state.explorer_type, "Unknown")
                    collector_full = type_mapping.get(st.session_state.collector_type, "Unknown")
                    manner_full = type_mapping.get(st.session_state.manner_type, "Unknown")
                    killer_full = type_mapping.get(st.session_state.killer_type, "Unknown")
                    composite = (
                        (st.session_state.killer_type or "")
                        + (st.session_state.socialiser_type or "")
                        + (st.session_state.explorer_type or "")
                        + (st.session_state.collector_type or "")
                        + "-"
                        + (st.session_state.manner_type or "")
                    )
                    st.markdown(f"<div class='centered' style='font-size: 24px; font-weight: bold;'>{composite}</div>", unsafe_allow_html=True)
                    prompt = (
                        f"사용자 분류 결과는 다음과 같습니다: "
                        f"공격형 유형: {killer_full}. "
                        f"사회적 유형: {social_full}, "
                        f"탐험가 유형: {explorer_full}, "
                        f"수집가 유형: {collector_full}, "
                        f"매너 유형: {manner_full}. "
                        "이 사용자에 대한 간단한 설명을 한국어로 제공해주세요."
                    )
                    try:
                        response = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "당신은 사용자의 게임유형을 분석하는 AI입니다."},
                                {"role": "user", "content": prompt}
                            ],
                            max_tokens=500,
                            n=1,
                            stop=None,
                            temperature=0.7,
                        )
                        suggestion = response.choices[0].message['content'].strip()
                        st.markdown(f"<div class='centered'><strong>{suggestion}</strong></div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"OpenAI 요청 오류: {e}")
                else:
                    st.markdown("<div class='centered'>모든 유형이 아직 계산되지 않았습니다.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(main())
