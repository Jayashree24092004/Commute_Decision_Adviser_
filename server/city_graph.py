"""
city_graph.py
=============
Realistic synthetic city — Synthcity.

Modelled loosely like a real Indian city layout:
  - A dense Downtown core
  - Residential suburbs (Home, Suburb_North, Suburb_South)
  - Key destinations (Office, School, Mall, Hospital, University, Airport)
  - Service stops (Pharmacy, Fuel_Stop, Park)
  - Transit hubs (Station_A, Station_B)

Every pair of important nodes has AT LEAST 3 distinct path options:
  1. A fast highway/express route (may have toll)
  2. A medium ring road route (no toll, moderate speed)
  3. A slow but fully local route (no toll, no highway, always exists)

This guarantees Option C (toll-free, no highway) is NEVER 999 minutes.
"""

import heapq
from typing import Dict, List, Tuple

CITY = {
    "name": "Synthcity",

    "nodes": [
        # Residential
        "Home", "Suburb_North", "Suburb_South",
        # Work & Education
        "Office", "School", "University",
        # Commercial & Services
        "Mall", "Pharmacy", "Fuel_Stop",
        # Infrastructure
        "Station_A", "Station_B", "Downtown",
        # Medical & Leisure
        "Hospital", "Park",
        # Transport
        "Airport",
        # Extra connector nodes — these are what fix the 999 problem
        # They act like real city junction points / neighbourhoods
        "Junction_East",   # east side connector
        "Junction_West",   # west side connector
        "Junction_North",  # north side connector
        "Crossroads",      # central crossroads (toll-free alternative to highway)
        "Old_Town",        # historic area with only local roads — always toll-free
    ],

    # ─────────────────────────────────────────────────────────────
    # EDGES
    # Format: (from, to, distance_km, base_time_min, has_toll, road_type)
    #
    # road_type choices:
    #   "highway"   — fast, often tolled, carries traffic multipliers
    #   "express"   — fast, no toll, moderate traffic
    #   "ring_road" — medium speed, no toll, bypass routes
    #   "local"     — slow, no toll, always available
    #
    # DESIGN RULE:
    #   Every major destination is reachable via local roads ONLY.
    #   So even with avoid_toll + avoid_highway, a path always exists.
    # ─────────────────────────────────────────────────────────────
    "edges": [

        # ── HOME CONNECTIONS ──────────────────────────────────────
        ("Home", "School",          2.1,  6,  False, "local"),
        ("Home", "Station_A",       3.2,  8,  False, "local"),
        ("Home", "Suburb_North",    4.5, 11,  False, "local"),
        ("Home", "Fuel_Stop",       2.0,  5,  False, "local"),
        ("Home", "Pharmacy",        1.5,  4,  False, "local"),
        ("Home", "Park",            3.9,  9,  False, "local"),
        ("Home", "Junction_West",   2.8,  7,  False, "local"),     # NEW
        ("Home", "Old_Town",        3.5,  9,  False, "local"),     # NEW toll-free path

        # ── SCHOOL CONNECTIONS ───────────────────────────────────
        ("School", "Office",        5.8, 14,  False, "ring_road"),
        ("School", "Station_A",     1.9,  5,  False, "local"),
        ("School", "Pharmacy",      0.8,  3,  False, "local"),
        ("School", "Junction_East", 2.2,  6,  False, "local"),     # NEW

        # ── STATION CONNECTIONS ──────────────────────────────────
        ("Station_A", "Downtown",   6.1, 12,  True,  "highway"),   # tolled highway
        ("Station_A", "Station_B",  3.3,  8,  False, "express"),
        ("Station_A", "Crossroads", 4.0, 10,  False, "ring_road"), # NEW toll-free alt
        ("Station_A", "Fuel_Stop",  1.1,  3,  False, "local"),
        ("Station_A", "Old_Town",   3.8,  9,  False, "local"),     # NEW

        ("Station_B", "Office",     2.7,  7,  False, "local"),
        ("Station_B", "Mall",       4.2, 10,  False, "ring_road"),
        ("Station_B", "Suburb_North", 6.3, 15, False, "local"),
        ("Station_B", "Junction_North", 3.1, 8, False, "local"),   # NEW

        # ── DOWNTOWN CONNECTIONS ─────────────────────────────────
        ("Downtown", "Airport",    18.0, 28,  True,  "highway"),   # tolled highway
        ("Downtown", "University",  4.4, 10,  False, "local"),
        ("Downtown", "Office",      1.8,  5,  False, "local"),
        ("Downtown", "Mall",        2.5,  6,  False, "ring_road"),
        ("Downtown", "Crossroads",  2.0,  5,  False, "local"),     # NEW toll-free alt
        ("Downtown", "Old_Town",    1.5,  4,  False, "local"),     # NEW

        # ── OFFICE CONNECTIONS ───────────────────────────────────
        ("Office", "Hospital",      3.0,  8,  False, "local"),
        ("Office", "Junction_East", 2.5,  6,  False, "local"),     # NEW

        # ── MALL CONNECTIONS ─────────────────────────────────────
        ("Mall", "Suburb_South",    5.0, 12,  False, "local"),
        ("Mall", "Junction_West",   3.2,  8,  False, "local"),     # NEW

        # ── HOSPITAL / UNIVERSITY / PARK ─────────────────────────
        ("Hospital", "University",  2.2,  6,  False, "local"),
        ("Hospital", "Suburb_South", 3.8, 9,  False, "local"),
        ("Hospital", "Junction_East", 1.8, 5, False, "local"),     # NEW
        ("University", "Park",      1.5,  4,  False, "local"),
        ("University", "Junction_North", 2.0, 5, False, "local"),  # NEW
        ("Park", "Junction_West",   2.1,  5,  False, "local"),     # NEW

        # ── SUBURB CONNECTIONS ───────────────────────────────────
        ("Suburb_North", "Junction_North", 3.5, 9, False, "local"),# NEW
        ("Suburb_South", "Junction_West",  4.0, 10, False, "local"),# NEW

        # ── AIRPORT CONNECTIONS ──────────────────────────────────
        # Airport now has a slow but toll-free local road
        ("Airport", "Junction_North", 14.0, 25, False, "ring_road"), # NEW key fix
        ("Airport", "Crossroads",     16.0, 28, False, "ring_road"), # NEW key fix

        # ── JUNCTION / CONNECTOR ROADS ───────────────────────────
        # These are the backbone of toll-free routing
        ("Junction_East",  "Junction_North", 3.0,  8, False, "ring_road"),
        ("Junction_East",  "Crossroads",     2.5,  6, False, "local"),
        ("Junction_West",  "Crossroads",     2.8,  7, False, "local"),
        ("Junction_West",  "Old_Town",       2.0,  5, False, "local"),
        ("Junction_North", "Crossroads",     3.5,  9, False, "ring_road"),
        ("Junction_North", "Downtown",       5.0, 12, False, "ring_road"), # toll-free to downtown
        ("Crossroads",     "Old_Town",       1.8,  5, False, "local"),
        ("Crossroads",     "Downtown",       2.0,  5, False, "local"),     # NEW
        ("Old_Town",       "Downtown",       1.5,  4, False, "local"),     # always open local
        ("Old_Town",       "University",     3.2,  8, False, "local"),     # NEW
    ],

    # ─────────────────────────────────────────────
    # TRAFFIC MULTIPLIERS
    # Edge key → {time_slot: multiplier}
    # Only highway and express roads have significant traffic impact
    # ─────────────────────────────────────────────
    "traffic": {
        # Highway routes — heavy congestion
        "Station_A→Downtown":      {"morning": 2.1, "midday": 1.2, "evening": 1.8, "night": 1.0},
        "Downtown→Station_A":      {"morning": 1.4, "midday": 1.2, "evening": 2.2, "night": 1.0},
        "Downtown→Airport":        {"morning": 1.3, "midday": 1.1, "evening": 1.6, "night": 1.0},
        "Airport→Downtown":        {"morning": 1.5, "midday": 1.1, "evening": 1.4, "night": 1.0},

        # Express and ring roads — moderate congestion
        "Station_A→Station_B":     {"morning": 1.5, "midday": 1.0, "evening": 1.4, "night": 1.0},
        "School→Office":           {"morning": 1.9, "midday": 1.0, "evening": 1.3, "night": 1.0},
        "Station_B→Office":        {"morning": 1.7, "midday": 1.0, "evening": 1.5, "night": 1.0},
        "Mall→Downtown":           {"morning": 1.2, "midday": 1.4, "evening": 1.9, "night": 1.0},
        "Downtown→Mall":           {"morning": 1.1, "midday": 1.3, "evening": 1.7, "night": 1.0},

        # Local roads near home — mild morning congestion
        "Home→Station_A":          {"morning": 1.6, "midday": 1.0, "evening": 1.2, "night": 1.0},
        "Suburb_North→Station_B":  {"morning": 1.5, "midday": 1.0, "evening": 1.4, "night": 1.0},

        # Junction roads — mild congestion only in peak hours
        "Junction_North→Downtown": {"morning": 1.3, "midday": 1.0, "evening": 1.4, "night": 1.0},
        "Crossroads→Downtown":     {"morning": 1.2, "midday": 1.0, "evening": 1.3, "night": 1.0},
    },

    # ─────────────────────────────────────────────
    # POINTS OF INTEREST — opening hours
    # ─────────────────────────────────────────────
    "pois": {
        "Pharmacy":     {"open": "08:00", "close": "21:00", "type": "service"},
        "Fuel_Stop":    {"open": "06:00", "close": "23:00", "type": "service"},
        "Mall":         {"open": "10:00", "close": "22:00", "type": "shopping"},
        "Hospital":     {"open": "00:00", "close": "23:59", "type": "emergency"},
        "Park":         {"open": "06:00", "close": "20:00", "type": "leisure"},
        "University":   {"open": "08:00", "close": "22:00", "type": "education"},
        "School":       {"open": "07:30", "close": "17:00", "type": "education"},
        "Airport":      {"open": "00:00", "close": "23:59", "type": "transport"},
        "Old_Town":     {"open": "00:00", "close": "23:59", "type": "area"},
        "Crossroads":   {"open": "00:00", "close": "23:59", "type": "junction"},
    }
}


# ─────────────────────────────────────────────
# HELPER: TIME SLOT FROM HOUR
# ─────────────────────────────────────────────

def get_time_slot(hour: int) -> str:
    if 7 <= hour < 10:   return "morning"
    elif 10 <= hour < 16: return "midday"
    elif 16 <= hour < 20: return "evening"
    else:                 return "night"


# ─────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────

def build_adjacency(time_slot: str) -> Dict[str, List[Tuple]]:
    adj: Dict[str, List] = {n: [] for n in CITY["nodes"]}
    for (frm, to, dist, base_time, toll, road) in CITY["edges"]:
        key     = f"{frm}→{to}"
        rev_key = f"{to}→{frm}"
        fwd_mult = CITY["traffic"].get(key,     {}).get(time_slot, 1.0)
        rev_mult = CITY["traffic"].get(rev_key, {}).get(time_slot, 1.0)
        fwd_time = round(base_time * fwd_mult, 1)
        rev_time = round(base_time * rev_mult, 1)
        adj[frm].append((to,  dist, fwd_time, toll, road))
        adj[to].append((frm, dist, rev_time, toll, road))
    return adj


# ─────────────────────────────────────────────
# POI OPEN CHECK
# ─────────────────────────────────────────────

def is_poi_open(poi_name: str, hour: int, minute: int = 0) -> bool:
    poi = CITY["pois"].get(poi_name)
    if not poi: return True
    open_h,  open_m  = map(int, poi["open"].split(":"))
    close_h, close_m = map(int, poi["close"].split(":"))
    current = hour * 60 + minute
    return (open_h * 60 + open_m) <= current <= (close_h * 60 + close_m)


# ─────────────────────────────────────────────
# CITY SUMMARY
# ─────────────────────────────────────────────

def get_city_summary() -> dict:
    return {
        "city_name":              CITY["name"],
        "total_nodes":            len(CITY["nodes"]),
        "total_edges":            len(CITY["edges"]),
        "nodes":                  CITY["nodes"],
        "pois":                   list(CITY["pois"].keys()),
        "traffic_affected_edges": list(CITY["traffic"].keys()),
    }