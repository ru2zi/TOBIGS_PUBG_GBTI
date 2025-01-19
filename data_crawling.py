import os
import json
import pandas as pd
from datetime import datetime
from tqdm import tqdm  # tqdm 임포트

# 설정
DATA_DIR = r'C:\Users\inho0\OneDrive\문서\GitHub\Tobigs_PUBG\PUBG_data'
OUTPUT_DIR = 'output'

# 출력 디렉토리 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_json(file_path):
    """JSON 파일을 로드하여 파이썬 객체로 반환."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except UnicodeDecodeError:
        # utf-8 디코딩 실패 시 다른 인코딩으로 다시 시도
        try:
            with open(file_path, 'r', encoding='latin1') as f:
                print(f"경고: {file_path} 파일이 utf-8이 아닌 다른 인코딩으로 처리되었습니다.")
                return json.load(f)
        except Exception as e:
            print(f"JSON 파일 로드 오류: {file_path}\n{e}")
            return None
    except Exception as e:
        print(f"JSON 파일 로드 오류: {file_path}\n{e}")
        return None

def parse_iso8601(timestamp_str):
    """ISO 8601 형식의 타임스탬프 문자열을 datetime 객체로 변환."""
    if not timestamp_str:
        return None
    try:
        # 마이크로초 부분 정밀도 조정 (최대 6자리로 제한)
        if '.' in timestamp_str:
            dot_index = timestamp_str.find('.')
            z_index = timestamp_str.find('Z', dot_index)
            if z_index == -1:
                z_index = len(timestamp_str)
            microseconds = timestamp_str[dot_index + 1 : z_index]
            if len(microseconds) > 6:
                microseconds = microseconds[:6]
            else:
                microseconds = microseconds.ljust(6, '0')
            timestamp_str = (
                timestamp_str[:dot_index + 1]
                + microseconds
                + timestamp_str[z_index:]
            )
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        print(f"타임스탬프 파싱 오류: {timestamp_str}")
        return None

def extract_item_usage(telemetry_data, account_id):
    """
    텔레메트리 데이터에서 아이템 사용 정보를 추출.
    """
    primary_weapon = None
    secondary_weapon = None
    armor_type = "None"
    health_items_used = 0
    boost_items_used = 0

    for event in telemetry_data:
        if event.get('_T') == 'LogItemEquip' and event.get('character', {}).get('accountId') == account_id:
            item_category = event.get('item', {}).get('category')
            item_id = event.get('item', {}).get('itemId')
            if item_category == 'Weapon':
                if primary_weapon is None:
                    primary_weapon = item_id
                elif secondary_weapon is None:
                    secondary_weapon = item_id
            elif item_category == 'Armor':
                armor_type = item_id

        if event.get('_T') == 'LogItemUse' and event.get('character', {}).get('accountId') == account_id:
            item_category = event.get('item', {}).get('category')
            if item_category == 'Healing':
                health_items_used += 1
            elif item_category == 'Boost':
                boost_items_used += 1

    return {
        'primary_weapon': primary_weapon,
        'secondary_weapon': secondary_weapon,
        'armor_type': armor_type,
        'use_of_health_items': health_items_used,
        'use_of_boost_items': boost_items_used
    }

def extract_movement_routes(telemetry_data, account_id, match_start_time):
    """
    텔레메트리 데이터에서 특정 플레이어의 이동 경로를 추출.
    """
    movement_routes = []
    player_position_events = [
        event for event in telemetry_data
        if event.get('_T') == 'LogPlayerPosition'
        and event.get('character', {}).get('accountId') == account_id
    ]
    for event in player_position_events:
        loc = event.get('character', {}).get('location', {})
        x = loc.get('x')
        y = loc.get('y')
        z = loc.get('z')
        timestamp_str = event.get('_D')
        timestamp_dt = parse_iso8601(timestamp_str)
        if timestamp_dt and match_start_time:
            relative_seconds = (timestamp_dt - match_start_time).total_seconds()
            if x is not None and y is not None and z is not None:
                movement_routes.append((relative_seconds, x, y, z))

    movement_routes.sort(key=lambda x: x[0])
    return movement_routes

def extract_login_logout_times(telemetry_data, account_id):
    """
    LogPlayerLogin / LogPlayerLogout을 통해 플레이어의 세션 시간 측정에 활용.
    - 가장 이른 로그인 시간(login_time)과 가장 늦은 로그아웃 시간(logout_time)을 추출.
    """
    login_time = None
    logout_time = None

    for event in telemetry_data:
        e_type = event.get('_T')
        e_id = event.get('accountId')  # LogPlayerLogin/Logout의 accountId는 루트에 존재
        if e_id == account_id:
            time_dt = parse_iso8601(event.get('_D'))
            if e_type == 'LogPlayerLogin':
                # 가장 이른 로그인 시간
                if login_time is None or (time_dt and time_dt < login_time):
                    login_time = time_dt
            elif e_type == 'LogPlayerLogout':
                # 가장 늦은 로그아웃 시간
                if logout_time is None or (time_dt and time_dt > logout_time):
                    logout_time = time_dt

    return login_time, logout_time

def extract_additional_data(telemetry_data, account_id, players_stats):
    """
    텔레메트리 데이터에서 추가 정보를 추출.
    """
    items_carried = []
    loot_events = []
    combat_events = []

    for event in telemetry_data:
        if event.get('_T') == 'LogItemPickup':
            if event.get('character', {}).get('accountId') == account_id:
                item_id = event.get('item', {}).get('itemId')
                if item_id:
                    items_carried.append(item_id)
                    loot_events.append(event.get('_D'))

        if event.get('_T') == 'LogPlayerAttack':
            if event.get('attacker', {}).get('accountId') == account_id:
                combat_events.append(event.get('_D'))

    # 로팅(looting) 시간 계산
    time_spent_looting = 0
    if loot_events:
        loot_timestamps = [parse_iso8601(ts) for ts in loot_events if ts]
        loot_timestamps = [ts for ts in loot_timestamps if ts is not None]
        if len(loot_timestamps) > 1:
            time_spent_looting = (max(loot_timestamps) - min(loot_timestamps)).total_seconds()

    # 전투(combat) 시간 계산
    time_spent_in_combat = 0
    if combat_events:
        combat_timestamps = [parse_iso8601(ts) for ts in combat_events if ts]
        combat_timestamps = [ts for ts in combat_timestamps if ts is not None]
        if len(combat_timestamps) > 1:
            time_spent_in_combat = (max(combat_timestamps) - min(combat_timestamps)).total_seconds()

    kills = players_stats.get('kills', 0)
    damage_dealt = players_stats.get('damageDealt', 0)

    return {
        'items_carried': ', '.join(items_carried) if items_carried else "None",
        'time_spent_looting_sec': time_spent_looting,
        'time_spent_in_combat_sec': time_spent_in_combat,
        'kills': kills,
        'damage_dealt': damage_dealt
    }

def extract_team_info(roster_data):
    """
    roster.json 데이터에서 팀 정보를 추출.
    """
    team_info = {}
    for roster in roster_data:
        if roster.get('type') != 'roster':
            continue
        attributes = roster.get('attributes', {})
        stats = attributes.get('stats', {})
        rank = stats.get('rank')
        team_id = stats.get('teamId')
        won = attributes.get('won', "false")

        team_info[team_id] = {
            'team_rank': rank,
            'team_won': won
        }
    return team_info

def extract_event_details(telemetry_data, account_id):
    """
    특정 플레이어의 이벤트 타입별 상세 정보를 추출.
    - LogSwimStart / LogSwimEnd는 필요 없으므로 제거.
    - LogPlayerMakeGroggy 처리 추가.
    - LogPlayerKillV2에서 killer/victim이 None일 수 있으니 안전 처리.
    """
    event_details = {}

    for event in telemetry_data:
        event_type = event.get('_T', 'Unknown')

        # ---- LogPlayerKillV2 ----
        if event_type == 'LogPlayerKillV2':
            # killer / victim / finishDamageInfo 가 None일 수 있으니 or {}로 안전처리
            killer_dict = event.get('killer') or {}
            victim_dict = event.get('victim') or {}
            finish_info = event.get('finishDamageInfo') or {}

            killer_id = killer_dict.get('accountId')
            victim_id = victim_dict.get('accountId')
            weapon_used = finish_info.get('damageCauserName')
            damage_reason = finish_info.get('damageReason')
            headshot = (damage_reason == "HeadShot")

            if killer_id == account_id or victim_id == account_id:
                kill_time = parse_iso8601(event.get('_D'))
                victim_loc = victim_dict.get('location', {}) or {}

                kill_details = {
                    'killer_account_id': killer_id,
                    'victim_account_id': victim_id,
                    'weapon_used': weapon_used,
                    'damage_reason': damage_reason,
                    'headshot': headshot,
                    'kill_time': kill_time.isoformat() if kill_time else "None",
                    'victim_location_x': victim_loc.get('x'),
                    'victim_location_y': victim_loc.get('y'),
                    'victim_location_z': victim_loc.get('z')
                }
                event_details.setdefault('LogPlayerKillV2_details', []).append(kill_details)

        # ---- LogPlayerMakeGroggy ----
        elif event_type == 'LogPlayerMakeGroggy':
            attacker_dict = event.get('attacker') or {}
            victim_dict = event.get('victim') or {}
            attacker_id = attacker_dict.get('accountId')
            victim_id = victim_dict.get('accountId')

            damage_reason = event.get('damageReason')
            damage_type_category = event.get('damageTypeCategory')
            distance = event.get('distance')
            is_through_wall = event.get('isThroughPenetrableWall', False)

            groggy_time = parse_iso8601(event.get('_D'))
            if attacker_id == account_id or victim_id == account_id:
                victim_loc = victim_dict.get('location', {}) or {}
                groggy_details = {
                    'attacker_id': attacker_id,
                    'victim_id': victim_id,
                    'damage_reason': damage_reason,
                    'damage_type_category': damage_type_category,
                    'distance': distance,
                    'is_through_wall': is_through_wall,
                    'groggy_time': groggy_time.isoformat() if groggy_time else "None",
                    'victim_x': victim_loc.get('x'),
                    'victim_y': victim_loc.get('y'),
                    'victim_z': victim_loc.get('z'),
                }
                event_details.setdefault('LogPlayerMakeGroggy_details', []).append(groggy_details)

        # ---- LogPlayerRevive ----
        elif event_type == 'LogPlayerRevive':
            reviver_dict = event.get('reviver') or {}
            victim_dict = event.get('victim') or {}

            reviver_id = reviver_dict.get('accountId')
            victim_id = victim_dict.get('accountId')

            if reviver_id == account_id or victim_id == account_id:
                revive_time = parse_iso8601(event.get('_D'))
                reviver_loc = reviver_dict.get('location', {}) or {}
                victim_loc = victim_dict.get('location', {}) or {}

                revive_details = {
                    'reviver_account_id': reviver_id,
                    'victim_account_id': victim_id,
                    'revive_time': revive_time.isoformat() if revive_time else "None",
                    'reviver_loc_x': reviver_loc.get('x'),
                    'reviver_loc_y': reviver_loc.get('y'),
                    'reviver_loc_z': reviver_loc.get('z'),
                    'victim_loc_x': victim_loc.get('x'),
                    'victim_loc_y': victim_loc.get('y'),
                    'victim_loc_z': victim_loc.get('z'),
                }
                event_details.setdefault('LogPlayerRevive_details', []).append(revive_details)

        # ---- LogPlayerAttack ----
        elif event_type == 'LogPlayerAttack':
            attacker_dict = event.get('attacker') or {}
            attacker_id = attacker_dict.get('accountId')
            if attacker_id == account_id:
                attack_time = parse_iso8601(event.get('_D'))
                weapon_item_id = (event.get('weapon') or {}).get('itemId')
                attack_type = event.get('attackType')
                fire_count = event.get('fireWeaponStackCount')
                attack_details = {
                    'attacker_account_id': attacker_id,
                    'attack_time': attack_time.isoformat() if attack_time else "None",
                    'weapon_item_id': weapon_item_id,
                    'attack_type': attack_type,
                    'fireWeaponStackCount': fire_count
                }
                event_details.setdefault('LogPlayerAttack_details', []).append(attack_details)

        # ---- LogVehicleRide ----
        elif event_type == 'LogVehicleRide':
            character_dict = event.get('character') or {}
            character_id = character_dict.get('accountId')
            if character_id == account_id:
                ride_time = parse_iso8601(event.get('_D'))
                vehicle_dict = event.get('vehicle') or {}
                vehicle_id = vehicle_dict.get('vehicleId')
                vehicle_type = vehicle_dict.get('vehicleType')
                seat_index = vehicle_dict.get('seatIndex')
                health_percent = vehicle_dict.get('healthPercent')
                fuel_percent = vehicle_dict.get('feulPercent')
                velocity = vehicle_dict.get('velocity')
                ride_details = {
                    'character_account_id': character_id,
                    'ride_time': ride_time.isoformat() if ride_time else "None",
                    'vehicle_id': vehicle_id,
                    'vehicle_type': vehicle_type,
                    'seat_index': seat_index,
                    'health_percent': health_percent,
                    'fuel_percent': fuel_percent,
                    'velocity': velocity
                }
                event_details.setdefault('LogVehicleRide_details', []).append(ride_details)

        # ---- LogItemUse ----
        elif event_type == 'LogItemUse':
            character_dict = event.get('character') or {}
            character_id = character_dict.get('accountId')
            if character_id == account_id:
                use_time = parse_iso8601(event.get('_D'))
                item_dict = event.get('item') or {}
                item_id = item_dict.get('itemId')
                effect = event.get('effect')
                item_use_details = {
                    'character_account_id': character_id,
                    'item_id': item_id,
                    'use_time': use_time.isoformat() if use_time else "None",
                    'effect': effect
                }
                event_details.setdefault('LogItemUse_details', []).append(item_use_details)

        # (LogSwimStart / LogSwimEnd 제거)

    return event_details

def compile_player_data(data_dir):
    """
    모든 플레이어 정보를 추출하여 데이터프레임 생성.
    """
    player_data = []

    # 모든 매치에서 발견된 _T 값을 미리 정의 (원하는 이벤트만 세어도 무방)
    unique_T_values = [
        'LogArmorDestroy', 'LogBlackZoneEnded', 'LogCarePackageLand', 'LogCarePackageSpawn',
        'LogCharacterCarry', 'LogEmPickupLiftOff', 'LogGameStatePeriodic', 'LogHeal',
        'LogItemAttach', 'LogItemDetach', 'LogItemDrop', 'LogItemEquip', 'LogItemPickup',
        'LogItemPickupFromCarepackage', 'LogItemPickupFromLootBox', 'LogItemPickupFromVehicleTrunk',
        'LogItemPutToVehicleTrunk', 'LogItemUnequip', 'LogItemUse', 'LogMatchDefinition',
        'LogMatchEnd', 'LogMatchStart', 'LogObjectDestroy', 'LogObjectInteraction',
        'LogParachuteLanding', 'LogPhaseChange', 'LogPlayerAttack', 'LogPlayerCreate',
        'LogPlayerDestroyBreachableWall', 'LogPlayerDestroyProp', 'LogPlayerKillV2',
        'LogPlayerLogin', 'LogPlayerLogout', 'LogPlayerMakeGroggy', 'LogPlayerPosition',
        'LogPlayerRevive', 'LogPlayerTakeDamage', 'LogPlayerUseFlareGun',
        'LogPlayerUseThrowable', 'LogRedZoneEnded', 'LogSwimEnd', 'LogSwimStart',
        'LogVaultStart', 'LogVehicleDamage', 'LogVehicleDestroy', 'LogVehicleLeave',
        'LogVehicleRide', 'LogWeaponFireCount', 'LogWheelDestroy'
    ]

    user_ids = [
        user_id for user_id in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, user_id))
    ]

    for user_id in tqdm(user_ids, desc="사용자 처리 중", unit="사용자"):
        user_path = os.path.join(data_dir, user_id)
        match_dirs = os.listdir(user_path)

        for match_id in tqdm(match_dirs, desc=f"{user_id}의 매치 처리 중", unit="매치", leave=False):
            match_path = os.path.join(user_path, match_id)
            if not os.path.isdir(match_path):
                continue

            meta_path = os.path.join(match_path, 'meta.json')
            players_path = os.path.join(match_path, 'players.json')
            telemetry_path = os.path.join(match_path, 'telemetry.json')
            roster_path = os.path.join(match_path, 'rosters.json')

            meta_data = load_json(meta_path)
            players_data = load_json(players_path)
            telemetry_data = load_json(telemetry_path)
            roster_data = load_json(roster_path)

            if not meta_data or not players_data or not telemetry_data or not roster_data:
                print(f"Missing data for match {match_id}.")
                continue

            team_info = extract_team_info(roster_data)

            # 매치 시작 시간(가장 이른 _D를 매치 시작 시점으로 가정)
            match_start_time = None
            for event in telemetry_data:
                t_dt = parse_iso8601(event.get('_D'))
                if t_dt:
                    match_start_time = t_dt
                    break
            if not match_start_time:
                print(f"Could not determine match start time for match {match_id}.")
                continue

            # 모든 플레이어 처리
            for player in players_data:
                players_stats = player.get('attributes', {}).get('stats', {})
                account_id = players_stats.get('playerId')
                if not account_id:
                    continue

                # 팀 정보 추출
                player_team_id = None
                for event in telemetry_data:
                    if event.get('character', {}).get('accountId') == account_id:
                        player_team_id = event.get('character', {}).get('teamId')
                        if player_team_id is not None:
                            break
                if player_team_id is None:
                    team_details = {'team_rank': "None", 'team_won': "false"}
                else:
                    team_details = team_info.get(player_team_id, {'team_rank': "None", 'team_won': "false"})

                # 아이템 사용 정보
                item_usage = extract_item_usage(telemetry_data, account_id)
                # 이동 경로
                movement_routes = extract_movement_routes(telemetry_data, account_id, match_start_time)
                # 추가 정보(아이템 픽업/전투 등)
                additional_data = extract_additional_data(telemetry_data, account_id, players_stats)
                # 이벤트별 상세
                event_details = extract_event_details(telemetry_data, account_id)

                # 로그인/로그아웃 시간
                login_time, logout_time = extract_login_logout_times(telemetry_data, account_id)
                session_time = None
                if login_time and logout_time and logout_time > login_time:
                    session_time = (logout_time - login_time).total_seconds()

                # 이동 경로 문자열
                movement_routes_str = "None"
                if movement_routes:
                    movement_routes_str = " -> ".join(
                        f"({x},{y},{z})" for _, x, y, z in movement_routes
                    )
                # 첫/마지막 위치
                first_loc = (None, None, None, None)
                if movement_routes:
                    first_loc = movement_routes[0]
                last_loc = (None, None, None, None)
                if movement_routes:
                    last_loc = movement_routes[-1]

                # 첫 LogPlayerPosition 이벤트에서 elapsedTime, numAlivePlayers
                elapsed_time = None
                num_alive_players = None
                for event in telemetry_data:
                    if (event.get('_T') == 'LogPlayerPosition'
                            and event.get('character', {}).get('accountId') == account_id):
                        elapsed_time = event.get('elapsedTime')
                        num_alive_players = event.get('numAlivePlayers')
                        break

                # _T 이벤트 카운트 (기본 0으로 초기화)
                T_counts = {f"{t}_count": 0 for t in unique_T_values}
                # LogPlayerLogin / Logout은 root level에 accountId가 있음
                for event in telemetry_data:
                    event_type = event.get('_T', 'Unknown')
                    char_acc = (event.get('character') or {}).get('accountId')
                    root_acc = event.get('accountId')  # LogPlayerLogin/Logout 같은 경우
                    if (char_acc == account_id) or (root_acc == account_id):
                        col_name = f"{event_type}_count"
                        if col_name in T_counts:
                            T_counts[col_name] += 1
                        else:
                            T_counts[col_name] = 1

                # DataFrame에 담을 row 하나 구성
                row_data = {
                    'match_id': match_id,
                    'map_name': meta_data.get('mapName'),
                    'game_mode': meta_data.get('gameMode'),
                    'player_id': player.get('id'),
                    'player_name': players_stats.get('name'),
                    'player_account_id': account_id,

                    # 접속(로그인/로그아웃) 관련
                    'login_time': login_time.isoformat() if login_time else None,
                    'logout_time': logout_time.isoformat() if logout_time else None,
                    'session_time_sec': session_time if session_time else 0.0,

                    # 아이템 사용, 추가데이터 등
                    **item_usage,
                    **additional_data,

                    # 이동 경로
                    'movement_routes': movement_routes_str,
                    'first_location_x': first_loc[1] if first_loc[1] else "None",
                    'first_location_y': first_loc[2] if first_loc[2] else "None",
                    'first_location_z': first_loc[3] if first_loc[3] else "None",
                    'final_location_x': last_loc[1] if last_loc[1] else "None",
                    'final_location_y': last_loc[2] if last_loc[2] else "None",
                    'final_location_z': last_loc[3] if last_loc[3] else "None",

                    # players.json에서 가져온 스탯
                    'walk_distance': players_stats.get('walkDistance', 0),
                    'swim_distance': players_stats.get('swimDistance', 0),
                    'ride_distance': players_stats.get('rideDistance', 0),
                    'road_kills': players_stats.get('roadKills', 0),
                    'vehicle_destroys': players_stats.get('vehicleDestroys', 0),
                    'weapons_acquired': players_stats.get('weaponsAcquired', 0),
                    'boosts': players_stats.get('boosts', 0),
                    'heals': players_stats.get('heals', 0),
                    'kill_streaks': players_stats.get('killStreaks', 0),
                    'headshot_kills': players_stats.get('headshotKills', 0),
                    'assists': players_stats.get('assists', 0),
                    'revives': players_stats.get('revives', 0),
                    'team_kills': players_stats.get('teamKills', 0),
                    'win_place': players_stats.get('winPlace', None),

                    # 팀 정보
                    'team_id': player_team_id if player_team_id else "None",
                    'team_rank': team_details.get('team_rank', "None"),
                    'team_won': team_details.get('team_won', "false"),

                    'elapsedTime': elapsed_time,
                    'numAlivePlayers': num_alive_players,

                    # _T 이벤트 카운트
                    **T_counts,

                    # 이벤트별 상세 목록 (JSON 형태)
                    'event_details': json.dumps(event_details) if event_details else "None"
                }
                player_data.append(row_data)

    return pd.DataFrame(player_data)

def main():
    """
    메인 함수: 모든 플레이어 데이터를 추출하고 CSV로 저장.
    """
    df_player_data = compile_player_data(DATA_DIR)

    if df_player_data.empty:
        print("No player data available.")
    else:
        output_csv = os.path.join(OUTPUT_DIR, 'player_data_with_event_details.csv')
        df_player_data.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Player data saved to '{output_csv}'.")
        print(df_player_data.head())

if __name__ == "__main__":
    main()
