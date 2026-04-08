"""
tasks.py
========
The full task dataset for the Commute Decision Advisor environment.

Structure:
  - EASY_TASKS    (3 tasks)  : Choose the best of 3 routes; no hard constraints
  - MEDIUM_TASKS  (3 tasks)  : Route + mandatory constraints (toll-free, time windows)
  - HARD_TASKS    (3 tasks)  : Multi-stop itinerary planning with time windows

Every task is a dict with:
  task_id       : Unique identifier
  difficulty    : "easy" | "medium" | "hard"
  scenario      : Natural language description of the commuter's situation
  start         : Starting node
  end           : Destination node
  depart_time   : "HH:MM" string
  constraints   : List of constraint dicts (for medium/hard)
  stops         : List of mandatory intermediate stops (for hard)
  correct_route : "A" | "B" | "C" | None (None = judged by constraint satisfaction)
  grading_keys  : What the grader checks

CONSTRAINT TYPES:
  {"type": "avoid_toll"}               → chosen route must not use tolls
  {"type": "avoid_road", "road": "highway"}  → no highways
  {"type": "max_time", "minutes": 45}  → total trip ≤ N minutes
  {"type": "arrive_before", "node": "School", "time": "08:45"}  → stop arrives before time
  {"type": "poi_open", "node": "Pharmacy"}  → POI must be open at arrival
  {"type": "avoid_node", "node": "Mall"}    → must not pass through this node
"""

EASY_TASKS = [
    {
        "task_id": "E001",
        "difficulty": "easy",
        "scenario": (
            "It's 9:00 AM on a weekday. Alex needs to get from Home to Office "
            "as quickly as possible. He has no specific constraints — just wants "
            "the fastest route for the morning commute."
        ),
        "start": "Home",
        "end": "Office",
        "depart_time": "09:00",
        "constraints": [],
        "stops": [],
        "correct_route": "A",   # Fastest = Option A
        "grading_keys": ["chosen_route", "reasoning_mentions_time"],
    },
    {
        "task_id": "E002",
        "difficulty": "easy",
        "scenario": (
            "It's 2:00 PM (midday traffic). Priya needs to travel from "
            "Suburb_North to Mall for a meeting. She prefers to minimise "
            "fuel cost, so she wants the route covering the fewest kilometres."
        ),
        "start": "Suburb_North",
        "end": "Mall",
        "depart_time": "14:00",
        "constraints": [],
        "stops": [],
        "correct_route": "B",   # Shortest distance = Option B
        "grading_keys": ["chosen_route", "reasoning_mentions_distance"],
    },
    {
        "task_id": "E003",
        "difficulty": "easy",
        "scenario": (
            "It's 11:30 AM. Ravi is driving from Station_B to Downtown "
            "for a client lunch. He has a company fuel card so cost doesn't matter. "
            "He only cares about arriving as fast as possible."
        ),
        "start": "Station_B",
        "end": "Downtown",
        "depart_time": "11:30",
        "constraints": [],
        "stops": [],
        "correct_route": "A",
        "grading_keys": ["chosen_route", "reasoning_mentions_time"],
    },
]

MEDIUM_TASKS = [
    {
        "task_id": "M001",
        "difficulty": "medium",
        "scenario": (
            "It's 8:15 AM. Sunita needs to drop her child at School "
            "before heading to Office. She must: (1) drop the child at School "
            "by 8:45 AM, (2) avoid using any toll roads because her FASTag "
            "is not loaded, (3) total trip must finish within 45 minutes. "
            "Which route satisfies ALL constraints?"
        ),
        "start": "Home",
        "end": "Office",
        "depart_time": "08:15",
        "via_stop": "School",   # mandatory waypoint in route generation
        "constraints": [
            {"type": "avoid_toll"},
            {"type": "arrive_before", "node": "School", "time": "08:45"},
            {"type": "max_time", "minutes": 45},
        ],
        "stops": [],
        "correct_route": None,   # Graded on constraint satisfaction, not fixed answer
        "grading_keys": [
            "constraint_avoid_toll",
            "constraint_arrive_before_school",
            "constraint_max_time",
            "reasoning_identifies_violations",
        ],
    },
    {
        "task_id": "M002",
        "difficulty": "medium",
        "scenario": (
            "It's 5:30 PM (evening rush). Karan is returning from Office to Home. "
            "He needs to stop at the Pharmacy (which closes at 9 PM but he needs "
            "to reach before 8:30 PM due to prescription pickup). "
            "He must avoid highways because of an accident closure. "
            "Suggest the best route and explain any trade-offs."
        ),
        "start": "Office",
        "end": "Home",
        "depart_time": "17:30",
        "via_stop": "Pharmacy",
        "constraints": [
            {"type": "avoid_road", "road": "highway"},
            {"type": "arrive_before", "node": "Pharmacy", "time": "20:30"},
            {"type": "poi_open", "node": "Pharmacy"},
        ],
        "stops": [],
        "correct_route": None,
        "grading_keys": [
            "constraint_avoid_highway",
            "constraint_pharmacy_time",
            "constraint_poi_open",
            "reasoning_mentions_tradeoffs",
        ],
    },
    {
        "task_id": "M003",
        "difficulty": "medium",
        "scenario": (
            "It's 7:45 AM. Meera needs to travel from Suburb_South to University. "
            "She has a hospital appointment at Hospital on the way (must arrive "
            "before 8:30 AM). Total trip budget is 50 minutes. She wants "
            "toll-free roads only. Which option satisfies all three constraints?"
        ),
        "start": "Suburb_South",
        "end": "University",
        "depart_time": "07:45",
        "via_stop": "Hospital",
        "constraints": [
            {"type": "avoid_toll"},
            {"type": "arrive_before", "node": "Hospital", "time": "08:30"},
            {"type": "max_time", "minutes": 50},
        ],
        "stops": [],
        "correct_route": None,
        "grading_keys": [
            "constraint_avoid_toll",
            "constraint_hospital_time",
            "constraint_max_time",
            "reasoning_identifies_violations",
        ],
    },
]

HARD_TASKS = [
    {
        "task_id": "H001",
        "difficulty": "hard",
        "scenario": (
            "It's 8:00 AM. David needs to complete a full morning run from Home. "
            "He must: (1) drop kids at School (school opens 7:30 AM), "
            "(2) fill fuel at Fuel_Stop (open 6 AM–11 PM), "
            "(3) pick up medicine at Pharmacy (opens 8 AM), "
            "and finally reach Office. "
            "He wants to finish all stops AND reach Office by 9:30 AM. "
            "Propose the optimal stop ordering, give estimated arrival times for each stop, "
            "and explain your reasoning. Total trip must be ≤ 90 minutes."
        ),
        "start": "Home",
        "end": "Office",
        "depart_time": "08:00",
        "constraints": [
            {"type": "max_time", "minutes": 90},
            {"type": "arrive_before", "node": "Office", "time": "09:30"},
            {"type": "poi_open", "node": "School"},
            {"type": "poi_open", "node": "Fuel_Stop"},
            {"type": "poi_open", "node": "Pharmacy"},
        ],
        "stops": ["School", "Fuel_Stop", "Pharmacy"],
        "correct_route": None,
        "grading_keys": [
            "all_stops_included",
            "arrival_times_provided",
            "no_open_violations",
            "office_arrival_before_0930",
            "total_time_within_limit",
            "reasoning_explains_ordering",
        ],
    },
    {
        "task_id": "H002",
        "difficulty": "hard",
        "scenario": (
            "It's 9:30 AM. An event coordinator needs to make stops across the city "
            "starting from Downtown. Must visit: Mall (opens 10 AM), "
            "University (opens 8 AM, close 10 PM), Park (opens 6 AM, closes 8 PM), "
            "and end at Airport for a 2:00 PM flight. "
            "Tolls are acceptable. Must reach Airport by 1:00 PM at the latest. "
            "Order the stops optimally, provide all arrival times, and justify the plan."
        ),
        "start": "Downtown",
        "end": "Airport",
        "depart_time": "09:30",
        "constraints": [
            {"type": "arrive_before", "node": "Airport", "time": "13:00"},
            {"type": "poi_open", "node": "Mall"},
            {"type": "poi_open", "node": "University"},
            {"type": "poi_open", "node": "Park"},
        ],
        "stops": ["Mall", "University", "Park"],
        "correct_route": None,
        "grading_keys": [
            "all_stops_included",
            "arrival_times_provided",
            "no_open_violations",
            "airport_arrival_before_1300",
            "reasoning_explains_ordering",
        ],
    },
    {
        "task_id": "H003",
        "difficulty": "hard",
        "scenario": (
            "It's 7:30 AM. A delivery driver starts from Station_A with 4 stops: "
            "Hospital (emergency delivery — must arrive FIRST, as early as possible), "
            "School (drop-off supplies, must arrive before 8:45 AM), "
            "Mall (retail delivery, opens 10 AM, must NOT arrive before it opens), "
            "then return to Station_A. Must avoid toll roads. "
            "Total round-trip must be ≤ 120 minutes. Plan the stop order and "
            "provide arrival times. Identify any constraint violations in other orderings."
        ),
        "start": "Station_A",
        "end": "Station_A",
        "depart_time": "07:30",
        "constraints": [
            {"type": "avoid_toll"},
            {"type": "max_time", "minutes": 120},
            {"type": "arrive_before", "node": "School", "time": "08:45"},
            {"type": "arrive_after",  "node": "Mall",   "time": "10:00"},
        ],
        "stops": ["Hospital", "School", "Mall"],
        "correct_route": None,
        "grading_keys": [
            "all_stops_included",
            "arrival_times_provided",
            "school_arrives_before_0845",
            "mall_arrives_after_1000",
            "constraint_avoid_toll",
            "total_time_within_limit",
            "reasoning_explains_ordering",
            "identifies_alternative_violations",
        ],
    },
]

ALL_TASKS = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS

TASKS_BY_DIFFICULTY = {
    "easy":   EASY_TASKS,
    "medium": MEDIUM_TASKS,
    "hard":   HARD_TASKS,
}