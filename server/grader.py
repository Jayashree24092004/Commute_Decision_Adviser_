"""
grader.py
=========
Programmatic constraint checker — the CORE of the reward function.

Philosophy:
  Most reward must be deterministic and unchallengeable.
  An LLM judge is used ONLY for reasoning quality (20-30% of total reward).
  The remaining 70-80% comes from hard constraint checks here.

  This prevents reward hacking: the model can't get a high score just by
  sounding confident or writing a long answer. It must satisfy measurable
  constraints.

Reward formula:
  Easy:    0.60 × correct_choice + 0.40 × reasoning_quality
  Medium:  0.70 × constraints_satisfied + 0.30 × reasoning_quality
  Hard:    0.60 × constraints_satisfied + 0.20 × all_stops_present + 0.20 × reasoning_quality

All sub-scores are 0.0–1.0. Final reward is always 0.0–1.0.
"""

import re
from typing import Optional
from city_graph import CITY, get_time_slot, is_poi_open
from route_engine import dijkstra, build_adjacency


# ─────────────────────────────────────────────
# 1. TIME UTILITIES
# ─────────────────────────────────────────────

def parse_time(t: str) -> int:
    """Convert 'HH:MM' string to total minutes since midnight."""
    h, m = map(int, t.strip().split(":"))
    return h * 60 + m


def add_minutes(base_time_str: str, delta_minutes: float) -> str:
    """Add delta_minutes to a 'HH:MM' string and return new 'HH:MM'."""
    base = parse_time(base_time_str)
    total = int(base + delta_minutes)
    h = (total // 60) % 24
    m = total % 60
    return f"{h:02d}:{m:02d}"


# ─────────────────────────────────────────────
# 2. EXTRACT ROUTE CHOICE FROM AGENT ACTION
# ─────────────────────────────────────────────

def extract_route_choice(agent_response: str) -> Optional[str]:
    """
    Parse which route option (A/B/C) the agent chose.
    Looks for explicit mentions like 'Route A', 'Option B', 'I choose C', etc.
    Returns "A", "B", "C", or None if unparseable.
    """
    text = agent_response.upper()
    # Patterns: "ROUTE A", "OPTION A", "CHOOSE A", standalone "A" near decision words
    patterns = [
        r'\bROUTE\s+([ABC])\b',
        r'\bOPTION\s+([ABC])\b',
        r'\bCHOOSE\s+([ABC])\b',
        r'\bSELECT\s+([ABC])\b',
        r'\bPICK\s+([ABC])\b',
        r'\bGO\s+WITH\s+([ABC])\b',
        r'\bRECOMMEND\s+([ABC])\b',
        r'"chosen_route"\s*:\s*"([ABC])"',
        r"'chosen_route'\s*:\s*'([ABC])'",
    ]
    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1)
    return None


def extract_arrival_times(agent_response: str) -> dict:
    """
    Extract mentioned arrival times from agent response.
    Looks for patterns like 'School: 08:22' or 'arrive at School at 8:22 AM'.
    Returns {node_name: "HH:MM"} dict.
    """
    arrivals = {}
    nodes = CITY["nodes"]

    for node in nodes:
        # Pattern: NodeName ... HH:MM  or  NodeName at H:MM AM/PM
        patterns = [
            rf'{node}[^.]*?(\d{{1,2}}:\d{{2}})',
            rf'(\d{{1,2}}:\d{{2}})[^.]*?{node}',
        ]
        for p in patterns:
            m = re.search(p, agent_response, re.IGNORECASE)
            if m:
                time_str = m.group(1)
                # Normalise to HH:MM
                parts = time_str.split(":")
                arrivals[node] = f"{int(parts[0]):02d}:{parts[1]}"
                break

    return arrivals


def extract_stop_list(agent_response: str, expected_stops: list) -> list:
    """Check which expected stops are mentioned in the response."""
    mentioned = []
    for stop in expected_stops:
        if stop.lower() in agent_response.lower():
            mentioned.append(stop)
    return mentioned


# ─────────────────────────────────────────────
# 3. ROUTE CONSTRAINT CHECKER
# ─────────────────────────────────────────────

def check_route_constraints(
    chosen_route: dict,  # Full route dict from generate_route_options()
    constraints: list,
    depart_time: str,    # "HH:MM"
    via_stop: Optional[str] = None,
) -> dict:
    """
    Check all constraints against a chosen route.

    Returns:
        {
          "results": {"constraint_type_key": True/False, ...},
          "score": 0.0–1.0,   # fraction of constraints satisfied
          "violations": ["description of each violation", ...]
        }
    """
    results = {}
    violations = []
    depart_min = parse_time(depart_time)
    time_slot = get_time_slot(depart_min // 60)

    for c in constraints:
        ctype = c["type"]
        key = f"constraint_{ctype}"

        if ctype == "avoid_toll":
            passed = not chosen_route.get("has_toll", False)
            key = "constraint_avoid_toll"
            if not passed:
                violations.append("Route uses a toll road but avoid_toll constraint set.")

        elif ctype == "avoid_road":
            road_to_avoid = c["road"]
            used = chosen_route.get("road_types_used", [])
            passed = road_to_avoid not in used
            key = f"constraint_avoid_road_{road_to_avoid}"
            if not passed:
                violations.append(f"Route uses {road_to_avoid} which is prohibited.")

        elif ctype == "max_time":
            passed = chosen_route["total_time_min"] <= c["minutes"]
            key = "constraint_max_time"
            if not passed:
                violations.append(
                    f"Route takes {chosen_route['total_time_min']} min "
                    f"but limit is {c['minutes']} min."
                )

        elif ctype == "arrive_before" and via_stop == c.get("node"):
            # Find time to reach the via_stop (first segment)
            adj = build_adjacency(time_slot)
            start_in_path = chosen_route["path"][0]
            leg = dijkstra(start_in_path, c["node"], adj, weight="time")
            if leg:
                arrival_min = depart_min + leg["total_time_min"]
                arrival_str = f"{int(arrival_min//60)%24:02d}:{int(arrival_min%60):02d}"
                deadline_min = parse_time(c["time"])
                passed = arrival_min <= deadline_min
                key = f"constraint_arrive_before_{c['node'].lower()}"
                if not passed:
                    violations.append(
                        f"Arrives at {c['node']} at {arrival_str}, "
                        f"but deadline is {c['time']}."
                    )
            else:
                passed = False
                key = f"constraint_arrive_before_{c['node'].lower()}"
                violations.append(f"Cannot reach {c['node']}.")

        elif ctype == "poi_open" and via_stop == c.get("node"):
            adj = build_adjacency(time_slot)
            leg = dijkstra(chosen_route["path"][0], c["node"], adj, weight="time")
            if leg:
                arrival_min = depart_min + leg["total_time_min"]
                arr_h = int(arrival_min // 60) % 24
                arr_m = int(arrival_min % 60)
                passed = is_poi_open(c["node"], arr_h, arr_m)
                key = f"constraint_poi_open_{c['node'].lower()}"
                if not passed:
                    poi = CITY["pois"].get(c["node"], {})
                    violations.append(
                        f"{c['node']} is closed at {arr_h:02d}:{arr_m:02d} "
                        f"(hours: {poi.get('open','?')}–{poi.get('close','?')})."
                    )
            else:
                passed = False
                key = f"constraint_poi_open_{c['node'].lower()}"
                violations.append(f"Cannot reach {c['node']}.")

        else:
            # Constraint not applicable to this route type — skip
            continue

        results[key] = passed

    if not results:
        return {"results": {}, "score": 1.0, "violations": []}

    score = sum(results.values()) / len(results)
    return {"results": results, "score": round(score, 3), "violations": violations}


# ─────────────────────────────────────────────
# 4. MULTI-STOP CONSTRAINT CHECKER (Hard tasks)
# ─────────────────────────────────────────────

def check_multistop_constraints(
    agent_response: str,
    plan: dict,             # From route_engine.plan_multi_stop()
    task: dict,
) -> dict:
    """
    Grade a hard multi-stop task.

    Checks:
      - All expected stops are mentioned in the response
      - Arrival times are provided for each stop
      - No POI open-hour violations
      - Specific time constraints (arrive_before, arrive_after)
      - Total trip within max_time
    """
    results = {}
    violations = []
    constraints = task["constraints"]
    expected_stops = task["stops"]
    depart_time = task["depart_time"]
    depart_min = parse_time(depart_time)

    # Check 1: All stops present in response
    mentioned = extract_stop_list(agent_response, expected_stops)
    all_stops = len(mentioned) == len(expected_stops)
    results["all_stops_included"] = all_stops
    if not all_stops:
        missing = set(expected_stops) - set(mentioned)
        violations.append(f"Missing stops in response: {missing}")

    # Check 2: Arrival times provided
    arrivals = extract_arrival_times(agent_response)
    arrivals_provided = len(arrivals) >= len(expected_stops)
    results["arrival_times_provided"] = arrivals_provided
    if not arrivals_provided:
        violations.append("Not all arrival times provided.")

    # Check 3: No POI open-hour violations in the optimal plan
    results["no_open_violations"] = len(plan.get("open_violations", [])) == 0
    for v in plan.get("open_violations", []):
        violations.append(f"POI timing issue: {v}")

    # Check 4: Per-constraint checks
    for c in constraints:
        ctype = c["type"]

        if ctype == "max_time":
            passed = plan["total_time_min"] <= c["minutes"]
            results["total_time_within_limit"] = passed
            if not passed:
                violations.append(
                    f"Total trip {plan['total_time_min']} min exceeds {c['minutes']} min limit."
                )

        elif ctype == "arrive_before":
            node = c["node"]
            arr_str = plan["arrival_times"].get(node)
            if arr_str:
                passed = parse_time(arr_str) <= parse_time(c["time"])
                key = f"{node.lower()}_arrives_before_{c['time'].replace(':','')}"
                results[key] = passed
                if not passed:
                    violations.append(f"{node} arrives {arr_str}, deadline {c['time']}.")
            else:
                results[f"{node.lower()}_arrives_before"] = False
                violations.append(f"No arrival time found for {node}.")

        elif ctype == "arrive_after":
            node = c["node"]
            arr_str = plan["arrival_times"].get(node)
            if arr_str:
                passed = parse_time(arr_str) >= parse_time(c["time"])
                key = f"{node.lower()}_arrives_after_{c['time'].replace(':','')}"
                results[key] = passed
                if not passed:
                    violations.append(f"{node} arrives {arr_str}, must be after {c['time']}.")
            else:
                results[f"{node.lower()}_arrives_after"] = False
                violations.append(f"No arrival time found for {node}.")

        elif ctype == "avoid_toll":
            # Check the plan's segments
            used_toll = any(s.get("has_toll", False) for s in plan.get("segments", []))
            results["constraint_avoid_toll"] = not used_toll
            if used_toll:
                violations.append("Plan uses a toll road despite avoid_toll constraint.")

    score = sum(results.values()) / len(results) if results else 0.0
    return {"results": results, "score": round(score, 3), "violations": violations}


# ─────────────────────────────────────────────
# 5. REASONING QUALITY MARKERS (heuristic, no LLM judge needed for speed)
# ─────────────────────────────────────────────

def score_reasoning_quality(
    agent_response: str,
    difficulty: str,
    constraints: list,
    chosen_route_id: Optional[str],
    violations_in_other_routes: list,
) -> float:
    """
    Heuristic reasoning quality score (0.0–1.0).
    Checks if the agent:
      1. Explains WHY the chosen route is best
      2. Mentions why eliminated routes are worse
      3. References specific numbers (time/distance)
      4. Mentions constraint names or conditions

    NOTE: This is intentionally a lightweight heuristic so inference stays fast.
    For a production environment you'd replace this with LLMJudge.
    """
    score = 0.0
    text = agent_response.lower()

    # Mentions specific numbers (time/distance values)
    has_numbers = bool(re.search(r'\d+\s*(min|km|minute|kilometer)', text))
    if has_numbers:
        score += 0.25

    # Mentions at least one constraint keyword
    constraint_keywords = ["toll", "highway", "time", "minute", "km", "arrive", "open", "close",
                           "constraint", "limit", "window", "deadline", "violation"]
    if any(kw in text for kw in constraint_keywords):
        score += 0.25

    # Mentions eliminated routes (good reasoning explains what's wrong with others)
    other_routes = [r for r in ["a", "b", "c"] if r != (chosen_route_id or "").lower()]
    mentions_others = sum(
        1 for r in other_routes
        if f"route {r}" in text or f"option {r}" in text
    )
    if mentions_others >= 1:
        score += 0.25

    # Response has reasonable length (not a one-liner)
    word_count = len(agent_response.split())
    if word_count >= 50:
        score += 0.25

    return round(min(score, 1.0), 3)


# ─────────────────────────────────────────────
# 6. MASTER GRADER
# ─────────────────────────────────────────────

def grade(
    task: dict,
    agent_response: str,
    route_options: list,     # The 3 options that were shown to the agent
    multistop_plan: Optional[dict] = None,  # Pre-computed optimal plan for hard tasks
) -> dict:
    """
    Master grading function. Returns a full grade report.

    Args:
        task            : Task dict from tasks.py
        agent_response  : The agent's full text response
        route_options   : The 3 route option dicts that were presented
        multistop_plan  : For hard tasks, the pre-computed optimal plan

    Returns:
        {
          "reward": 0.0–1.0,
          "breakdown": {
            "constraint_score": ...,
            "reasoning_score": ...,
          },
          "chosen_route": "A"/"B"/"C"/None,
          "violations": [...],
          "details": {...}
        }
    """
    difficulty = task["difficulty"]
    constraints = task["constraints"]
    correct = task.get("correct_route")
    depart_time = task["depart_time"]
    via_stop = task.get("via_stop")

    chosen_id = extract_route_choice(agent_response)
    chosen_route = next((r for r in route_options if r["id"] == chosen_id), None)

    violations = []
    constraint_score = 0.0
    route_correct_score = 0.0

    # ── EASY: graded on correct choice + reasoning
    if difficulty == "easy":
        route_correct_score = 1.0 if chosen_id == correct else 0.0
        if chosen_id != correct:
            violations.append(
                f"Chose route {chosen_id} but optimal was {correct}."
            )
        reasoning_score = score_reasoning_quality(
            agent_response, difficulty, constraints, chosen_id, []
        )
        reward = round(0.60 * route_correct_score + 0.40 * reasoning_score, 3)
        return {
            "reward": reward,
            "breakdown": {
                "route_correct_score": route_correct_score,
                "reasoning_score": reasoning_score,
            },
            "chosen_route": chosen_id,
            "violations": violations,
        }

    # ── MEDIUM: graded on constraint satisfaction + reasoning
    elif difficulty == "medium":
        if chosen_route:
            check = check_route_constraints(
                chosen_route, constraints, depart_time, via_stop
            )
            constraint_score = check["score"]
            violations = check["violations"]
        else:
            violations.append("Could not determine which route was chosen.")

        reasoning_score = score_reasoning_quality(
            agent_response, difficulty, constraints, chosen_id, violations
        )
        reward = round(0.70 * constraint_score + 0.30 * reasoning_score, 3)
        return {
            "reward": reward,
            "breakdown": {
                "constraint_score": constraint_score,
                "reasoning_score": reasoning_score,
                "constraint_details": check["results"] if chosen_route else {},
            },
            "chosen_route": chosen_id,
            "violations": violations,
        }

    # ── HARD: graded on multi-stop constraints + stop coverage + reasoning
    elif difficulty == "hard":
        if multistop_plan:
            check = check_multistop_constraints(agent_response, multistop_plan, task)
            constraint_score = check["score"]
            violations = check["violations"]
        else:
            violations.append("No multi-stop plan provided to grader.")

        mentioned_stops = extract_stop_list(agent_response, task["stops"])
        stops_score = len(mentioned_stops) / len(task["stops"]) if task["stops"] else 1.0

        reasoning_score = score_reasoning_quality(
            agent_response, difficulty, constraints, None, violations
        )
        reward = round(
            0.50 * constraint_score +
            0.20 * stops_score +
            0.30 * reasoning_score,
            3
        )
        return {
            "reward": reward,
            "breakdown": {
                "constraint_score": constraint_score,
                "stops_coverage_score": stops_score,
                "reasoning_score": reasoning_score,
                "constraint_details": check["results"] if multistop_plan else {},
            },
            "chosen_route": None,
            "violations": violations,
        }

    return {"reward": 0.0, "breakdown": {}, "chosen_route": None, "violations": ["Unknown difficulty."]}