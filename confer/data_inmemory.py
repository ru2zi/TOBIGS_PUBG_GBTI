# data_inmemory.py

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime

###################################################
# 헬퍼 함수들
###################################################

def parse_iso8601_custom(timestamp_str):
    """
    ISO8601 형식의 타임스탬프 문자열을 datetime으로 변환하는 헬퍼 함수.
    마이크로초 부분을 6자리로 제한하여 파싱.
    """
    if timestamp_str is None:
        return None
    try:
        if '.' in timestamp_str:
            dot_index = timestamp_str.find('.')
            z_index = timestamp_str.find('Z', dot_index)
            if z_index == -1:
                z_index = len(timestamp_str)
            microseconds = timestamp_str[dot_index+1:z_index]
            if len(microseconds) > 6:
                microseconds = microseconds[:6]
            else:
                microseconds = microseconds.ljust(6, '0')
            timestamp_str = (
                timestamp_str[:dot_index+1] + microseconds + timestamp_str[z_index:]
            )
        return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError:
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return None

def extract_team_info_gpti(roster_data):
    """
    roster_data(list/dict)에서 팀ID -> (team_rank, team_won) 정보를 추출.
    """
    team_info = {}
    for roster_item in roster_data:
        if roster_item.get('type') != 'roster':
            continue
        attributes = roster_item.get('attributes', {})
        stats = attributes.get('stats', {})
        rank = stats.get('rank', None)
        team_id = stats.get('teamId', None)
        won = attributes.get('won', "false")
        team_info[team_id] = {
            'team_rank': rank,
            'team_won': won
        }
    return team_info


def extract_item_usage_gpti(telemetry_data, account_id):
    """
    텔레메트리 LogItemUse, LogItemEquip 분석 -> 무기, 방어구, 힐/부스트 사용량
    """
    primary_weapon = None
    secondary_weapon = None
    armor_type = "None"
    health_items_used = 0
    boost_items_used = 0

    for event in telemetry_data:
        if event.get('_T') == 'LogItemEquip':
            cdict = event.get('character', {})
            if cdict.get('accountId') == account_id:
                item_dict = event.get('item', {})
                cat = item_dict.get('category', None)
                item_id = item_dict.get('itemId', None)
                if cat == 'Weapon':
                    if primary_weapon is None:
                        primary_weapon = item_id
                    elif secondary_weapon is None:
                        secondary_weapon = item_id
                elif cat == 'Armor':
                    armor_type = item_id

        if event.get('_T') == 'LogItemUse':
            cdict = event.get('character', {})
            if cdict.get('accountId') == account_id:
                item_dict = event.get('item', {})
                cat = item_dict.get('category', None)
                if cat == 'Healing':
                    health_items_used += 1
                elif cat == 'Boost':
                    boost_items_used += 1

    usage_dict = {
        'primary_weapon': primary_weapon,
        'secondary_weapon': secondary_weapon,
        'armor_type': armor_type,
        'use_of_health_items': health_items_used,
        'use_of_boost_items': boost_items_used
    }
    return usage_dict

def extract_movement_routes_gpti(telemetry_data, account_id, match_start_time):
    """
    LogPlayerPosition -> 이동경로 (relative_time, x, y, z)
    """
    movement_routes = []
    for event in telemetry_data:
        if event.get('_T') == 'LogPlayerPosition':
            cdict = event.get('character', {})
            if cdict.get('accountId') == account_id:
                loc = cdict.get('location', {})
                x = loc.get('x', None)
                y = loc.get('y', None)
                z = loc.get('z', None)
                dt = parse_iso8601_custom(event.get('_D', None))
                if dt and match_start_time:
                    rel_sec = (dt - match_start_time).total_seconds()
                    if x is not None and y is not None and z is not None:
                        movement_routes.append((rel_sec, x, y, z))
    movement_routes.sort(key=lambda x: x[0])
    return movement_routes

def extract_additional_data_gpti(telemetry_data, account_id, players_stats):
    """
    아이템 픽업, 공격 이벤트 -> looting/combat 시간 계산, + kills/damageDealt
    """
    items_carried = []
    loot_events = []
    combat_events = []

    for event in telemetry_data:
        if event.get('_T') == 'LogItemPickup':
            cdict = event.get('character', {})
            if cdict.get('accountId') == account_id:
                item_id = event.get('item', {}).get('itemId', None)
                if item_id:
                    items_carried.append(item_id)
                    loot_events.append(event.get('_D', None))

        if event.get('_T') == 'LogPlayerAttack':
            atk_dict = event.get('attacker', {})
            if atk_dict.get('accountId') == account_id:
                combat_events.append(event.get('_D', None))

    def calc_time_spent(time_list):
        if not time_list:
            return 0.0
        dt_list = []
        for ts_str in time_list:
            dt = parse_iso8601_custom(ts_str)
            if dt:
                dt_list.append(dt)
        if len(dt_list) < 2:
            return 0.0
        return (max(dt_list) - min(dt_list)).total_seconds()

    time_spent_looting = calc_time_spent(loot_events)
    time_spent_in_combat = calc_time_spent(combat_events)

    kills = players_stats.get('kills', 0)
    damage_dealt = players_stats.get('damageDealt', 0)

    additional_dict = {
        'items_carried': ', '.join(items_carried) if items_carried else "None",
        'time_spent_looting_sec': time_spent_looting,
        'time_spent_in_combat_sec': time_spent_in_combat,
        'kills': kills,
        'damage_dealt': damage_dealt
    }
    return additional_dict


###################################################
# 핵심: 한 행 DF -> CSV 저장
###################################################

def generate_single_csv_line_in_memory(
    meta_data,
    players_data,
    telemetry_data,
    roster_data,
    user_name,
    match_id
):
    """
    이미 메모리에 로드된 JSON 구조(meta/players/telemetry/roster)와
    user_name, match_id를 받아,
    특정 유저 한 명에 대한 한 행 DataFrame -> CSV 저장.
    """

    # (A) 추가: 분석하려는 _T 이벤트 목록
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

    # 1) 팀 정보
    team_info = extract_team_info_gpti(roster_data)

    # 2) 매치 시작 시간
    match_start_time = None
    for evt in telemetry_data:
        dt = parse_iso8601_custom(evt.get('_D'))
        if dt:
            match_start_time = dt
            break
    if not match_start_time:
        return None, "[ERROR] 매치 시작 시간을 결정할 수 없습니다."

    # 3) players_data에서 user_name 찾기(대소문자 무시)
    target_account_id = None
    target_player_data = None
    for p in players_data:
        stt = p.get('attributes', {}).get('stats', {})
        p_name_lower = stt.get('name', '').lower()
        if p_name_lower == user_name.lower():
            target_account_id = stt.get('playerId', None)
            target_player_data = p
            break
    if not target_account_id or not target_player_data:
        return None, f"[ERROR] user_name={user_name} 해당하는 player가 없음."

    players_stats = target_player_data.get('attributes', {}).get('stats', {})

    # 4) teamId 확인
    player_team_id = None
    for ev in telemetry_data:
        cdict = ev.get('character', {})
        if cdict.get('accountId') == target_account_id:
            player_team_id = cdict.get('teamId', None)
            if player_team_id is not None:
                break
    if player_team_id is None:
        team_details = {'team_rank': "None", 'team_won': "false"}
    else:
        team_details = team_info.get(player_team_id, {'team_rank': "None", 'team_won': "false"})

    # 5) 아이템 사용, 이동경로, 추가데이터
    usage_dict = extract_item_usage_gpti(telemetry_data, target_account_id)
    movement_routes = extract_movement_routes_gpti(telemetry_data, target_account_id, match_start_time)
    additional_dict = extract_additional_data_gpti(telemetry_data, target_account_id, players_stats)

    # 이동경로 문자열
    if movement_routes:
        movement_str = ' -> '.join([f"({x:.1f},{y:.1f},{z:.1f})" for _, x, y, z in movement_routes])
    else:
        movement_str = "None"

    # 첫 위치, 마지막 위치
    first_loc = movement_routes[0] if movement_routes else (None, None, None, None)
    last_loc = movement_routes[-1] if movement_routes else (None, None, None, None)

    # (B) 로그인/로그아웃, etc. - 여기서는 생략 가능
    # ...

    # (C) 이벤트 카운트 딕셔너리 (T_counts)
    T_counts = {f"{t}_count": 0 for t in unique_T_values}
    # LogPlayerLogin / Logout은 root level에 accountId가 있음
    for e in telemetry_data:
        event_type = e.get('_T', 'Unknown')
        char_acc = (e.get('character') or {}).get('accountId')
        root_acc = e.get('accountId')  # LogPlayerLogin/Logout 같은 경우
        if (char_acc == target_account_id) or (root_acc == target_account_id):
            col_name = f"{event_type}_count"
            if col_name in T_counts:
                T_counts[col_name] += 1
            else:
                # 만약 unique_T_values 목록에 없는 event_type이면?
                # 필요시 아래처럼 누적 가능 (주석해제)
                # T_counts[col_name] = T_counts.get(col_name, 0) + 1
                pass
                # 여기서는 unique_T_values 안의 이벤트만 카운트

    # elapsedTime, num_alive_players
    elapsed_time = None
    num_alive_players = None
    for ev in telemetry_data:
        if ev.get('_T') == 'LogPlayerPosition':
            cdict = ev.get('character', {})
            if cdict.get('accountId') == target_account_id:
                elapsed_time = ev.get('elapsedTime', None)
                num_alive_players = ev.get('numAlivePlayers', None)
                break

    # row_data
    row_data = {
        'match_id': match_id,
        'map_name': meta_data.get('mapName', None),
        'game_mode': meta_data.get('gameMode', None),
        'player_name': players_stats.get('name', None),
        'player_account_id': target_account_id,

        # 아이템 사용
        **usage_dict,
        # looting/combat + kills/damage
        **additional_dict,

        'movement_routes': movement_str,
        'first_location_x': first_loc[1] if first_loc[1] is not None else "None",
        'first_location_y': first_loc[2] if first_loc[2] is not None else "None",
        'first_location_z': first_loc[3] if first_loc[3] is not None else "None",
        'final_location_x': last_loc[1] if last_loc[1] is not None else "None",
        'final_location_y': last_loc[2] if last_loc[2] is not None else "None",
        'final_location_z': last_loc[3] if last_loc[3] is not None else "None",

        # players_stats에서 추가로
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

        'team_id': player_team_id if player_team_id else "None",
        'team_rank': team_details.get('team_rank', "None"),
        'team_won': team_details.get('team_won', "false"),

        'elapsedTime': elapsed_time,
        'numAlivePlayers': num_alive_players,

        # _T 이벤트 카운트
        **T_counts,
    }

    # DataFrame (1행)
    df = pd.DataFrame([row_data])

    # CSV 저장
    output_dir = 'output_inmemory_updated'
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"{user_name}_{match_id}_player_data.csv")

    if os.path.exists(output_csv):
        return df, f"[WARN] 이미 CSV가 존재합니다: {output_csv}"

    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    return df, f"[INFO] CSV 생성 완료: {output_csv}"
