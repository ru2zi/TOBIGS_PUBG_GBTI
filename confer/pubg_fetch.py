# pubg_fetch.py

import aiohttp
import asyncio
import os
from dotenv import load_dotenv

# (헬퍼) fetch_data
async def fetch_data(session, url, headers=None):
    """비동기 GET -> JSON 응답"""
    if headers is None:
        headers = {}
    try:
        async with session.get(url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                print(f"[WARN] {url} 응답코드={resp.status}")
                return None
    except Exception as e:
        print(f"[ERROR] fetch_data 오류: {e}")
        return None

async def fetch_match_json(session, match_id, api_key):
    """
    match_id에 대한 meta/rosters/participants/asset(telemetry URL) JSON 구조를 반환
    (디스크 저장 없이 메모리에서만)
    """
    url = f"https://api.pubg.com/shards/kakao/matches/{match_id}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/vnd.api+json"
    }
    match_data = await fetch_data(session, url, headers)
    return match_data if match_data else None

async def fetch_telemetry_json(session, telemetry_url):
    """
    텔레메트리 URL -> telemetry_data (list or dict)
    """
    telemetry_data = await fetch_data(session, telemetry_url, headers={})
    return telemetry_data if telemetry_data else None


async def get_match_in_memory(session, match_id, api_key):
    """
    match_id에 대한
    meta_data, players_data, roster_data, telemetry_data
    4개를 모두 메모리에 불러와 dict 형태로 반환
    """
    match_obj = await fetch_match_json(session, match_id, api_key)
    if not match_obj:
        return None

    data_block = match_obj.get("data", {})
    included = match_obj.get("included", [])

    # meta_data
    meta_data = data_block.get("attributes", {})

    # roster_data
    roster_data = [inc for inc in included if inc.get("type") == "roster"]

    # players_data
    players_data = [inc for inc in included if inc.get("type") == "participant"]

    # telemetry_url
    telemetry_url = None
    for inc in included:
        if inc.get("type") == "asset":
            telemetry_url = inc.get("attributes", {}).get("URL")
            break
    if not telemetry_url:
        print("[WARN] telemetry_url이 없음.")
        telemetry_data = []
    else:
        telemetry_data = await fetch_telemetry_json(session, telemetry_url)

    # 최종
    return {
        "meta_data": meta_data,
        "players_data": players_data,
        "roster_data": roster_data,
        "telemetry_data": telemetry_data
    }
