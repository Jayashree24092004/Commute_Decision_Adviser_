"""
main.py
=======
FastAPI server implementing the full OpenEnv spec.

Required endpoints:
  POST /reset          → Start a new episode, return initial observation
  POST /step           → Submit agent action, return observation + reward
  GET  /state          → Return current environment state
  GET  /health         → Health check (must return 200)
  GET  /tasks          → List all available tasks
  GET  /web            → Interactive browser UI

OpenEnv spec compliance:
  - All responses are typed Pydantic models
  - Rewards are always 0.0–1.0
  - reset() starts a new episode
  - step() accepts an action and returns (observation, reward, done, info)
  - state() returns the current episode state without side effects

Episode lifecycle:
  1. Client calls /reset (optionally with difficulty/task_id)
  2. Environment loads a task, generates route options, returns observation
  3. Client calls /step with agent's action (chosen route + reasoning)
  4. Environment grades the action, returns reward + done=True
  5. Episode ends — client calls /reset to start a new one
"""

import uuid
import random
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from models import CommuteAction, CommuteObservation, StepResult, EnvState, RouteOption, Constraint
from tasks import ALL_TASKS, TASKS_BY_DIFFICULTY
from city_graph import CITY, get_time_slot, build_adjacency
from route_engine import generate_route_options, plan_multi_stop
from grader import grade, parse_time

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────

app = FastAPI(
    title="Commute Decision Advisor — OpenEnv",
    description=(
        "An RL environment where an LLM agent reasons about commute route choices "
        "given real-world constraints (time windows, tolls, multi-stop ordering). "
        "Built for the Meta × PyTorch × Scaler OpenEnv Hackathon."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory state (one episode at a time per server instance)
_state: Dict[str, Any] = {
    "episode_id": None,
    "task": None,
    "route_options": None,
    "multistop_plan": None,
    "observation": None,
    "completed": False,
    "reward": 0.0,
    "steps": 0,
    "last_action": None,
    "last_grade": None,
}


# ─────────────────────────────────────────────
# OBSERVATION BUILDER
# ─────────────────────────────────────────────

def build_observation(task: dict, route_options: list) -> CommuteObservation:
    """
    Convert a task + pre-computed routes into a full CommuteObservation.
    This is what the agent sees.
    """
    depart_hour = int(task["depart_time"].split(":")[0])
    time_slot = get_time_slot(depart_hour)

    # Convert constraint dicts to typed Constraint objects with human descriptions
    type_descriptions = {
        "avoid_toll": "Must not use any toll roads",
        "avoid_road": lambda c: f"Must avoid {c['road']} roads",
        "max_time": lambda c: f"Total trip must be ≤ {c['minutes']} minutes",
        "arrive_before": lambda c: f"Must arrive at {c['node']} before {c['time']}",
        "arrive_after": lambda c: f"Must arrive at {c['node']} after {c['time']} (not before it opens)",
        "poi_open": lambda c: f"{c['node']} must be open when you arrive",
        "avoid_node": lambda c: f"Must not pass through {c['node']}",
    }
    constraints_typed = []
    for c in task["constraints"]:
        desc_fn = type_descriptions.get(c["type"], lambda x: str(x))
        if callable(desc_fn):
            desc = desc_fn(c)
        else:
            desc = desc_fn
        constraints_typed.append(Constraint(type=c["type"], description=desc, raw=c))

    # Route options typed
    route_options_typed = [RouteOption(**r) for r in route_options]

    # City context (POI hours, traffic notes)
    city_context = {
        "nodes": CITY["nodes"],
        "poi_hours": {k: f"{v['open']}–{v['close']}" for k, v in CITY["pois"].items()},
        "traffic_note": (
            f"Current time slot: {time_slot}. "
            "Morning (7-10 AM) and Evening (4-8 PM) have heavy traffic on major routes."
        ),
    }

    # Build the task prompt (what you'd pass to an LLM)
    route_lines = []
    for r in route_options:
        toll_str = "⚠️ Uses toll" if r["has_toll"] else "✅ No toll"
        route_lines.append(
            f"  Option {r['id']} — {r['label']}: {r['total_time_min']} min, "
            f"{r['total_distance_km']} km, via {r['via']}. {toll_str}. "
            f"Roads: {', '.join(r['road_types_used'])}."
        )
    routes_text = "\n".join(route_lines)

    constraint_lines = "\n".join(f"  • {c.description}" for c in constraints_typed)
    stops_text = (
        f"Mandatory stops to order: {', '.join(task['stops'])}" if task["stops"]
        else "No intermediate stops."
    )

    task_prompt = f"""You are a commute planning assistant. Analyse the scenario and route options below, then recommend the best route that satisfies ALL constraints.

SCENARIO:
{task['scenario']}

DEPARTURE: {task['start']} at {task['depart_time']} ({time_slot} traffic)
DESTINATION: {task['end']}
{stops_text}

ROUTE OPTIONS:
{routes_text}

CONSTRAINTS (ALL must be satisfied):
{constraint_lines if constraint_lines else '  None — choose the optimal route.'}

CITY CONTEXT:
  POI Opening Hours: {city_context['poi_hours']}
  {city_context['traffic_note']}

REQUIRED RESPONSE FORMAT:
  1. State which route option you choose (A, B, or C). For multi-stop tasks, state the optimal stop order.
  2. Provide estimated arrival times for each stop (for medium/hard tasks).
  3. Explain WHY each other option was eliminated (cite specific constraints violated or numbers).
  4. Confirm which constraints your chosen route satisfies.
"""

    return CommuteObservation(
        task_id=task["task_id"],
        difficulty=task["difficulty"],
        scenario=task["scenario"],
        start=task["start"],
        end=task["end"],
        depart_time=task["depart_time"],
        time_slot=time_slot,
        route_options=route_options_typed,
        constraints=constraints_typed,
        stops=task.get("stops", []),
        city_context=city_context,
        task_prompt=task_prompt,
        done=False,
    )


# ─────────────────────────────────────────────
# RESET REQUEST MODEL
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[str] = None   # "easy" | "medium" | "hard" | None (random)
    task_id: Optional[str] = None       # Specific task to load


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """OpenEnv required health check. Must return 200."""
    return {"status": "ok", "environment": "commute-decision-advisor", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()) -> EnvState:
    """
    Start a new episode.

    Optionally specify difficulty ("easy"/"medium"/"hard") or a specific task_id.
    If neither is given, a random task is selected.

    Returns the initial EnvState including the full observation.
    """
    # Select task
    if req.task_id:
        task = next((t for t in ALL_TASKS if t["task_id"] == req.task_id), None)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task '{req.task_id}' not found.")
    elif req.difficulty:
        pool = TASKS_BY_DIFFICULTY.get(req.difficulty, ALL_TASKS)
        task = random.choice(pool)
    else:
        task = random.choice(ALL_TASKS)

    # Compute routes
    depart_hour = int(task["depart_time"].split(":")[0])
    time_slot = get_time_slot(depart_hour)
    via_stop = task.get("via_stop")

    route_options = generate_route_options(
        start=task["start"],
        end=task["end"],
        time_slot=time_slot,
        via_stop=via_stop,
    )

    # For hard tasks: pre-compute the optimal multi-stop plan
    multistop_plan = None
    if task["difficulty"] == "hard" and task["stops"]:
        adj = build_adjacency(time_slot)
        multistop_plan = plan_multi_stop(
            start=task["start"],
            stops=task["stops"],
            end=task["end"],
            adj=adj,
            depart_hour=depart_hour,
            depart_minute=int(task["depart_time"].split(":")[1]),
        )

    obs = build_observation(task, route_options)
    episode_id = str(uuid.uuid4())[:8]

    # Store state
    _state.update({
        "episode_id": episode_id,
        "task": task,
        "route_options": route_options,
        "multistop_plan": multistop_plan,
        "observation": obs,
        "completed": False,
        "reward": 0.0,
        "steps": 0,
        "last_action": None,
        "last_grade": None,
    })

    return EnvState(
        episode_id=episode_id,
        task_id=task["task_id"],
        difficulty=task["difficulty"],
        started=True,
        completed=False,
        current_reward=0.0,
        total_steps=0,
        observation=obs,
    )


@app.post("/step")
def step(action: CommuteAction) -> StepResult:
    """
    Submit agent's action. Returns observation + reward + done.

    Each episode is single-step: the agent sees the task, responds once,
    and the episode ends. This matches the real-world use case where a
    commuter gets one recommendation per trip.
    """
    if not _state["task"]:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    if _state["completed"]:
        raise HTTPException(status_code=400, detail="Episode already completed. Call /reset.")

    task = _state["task"]
    route_options = _state["route_options"]
    multistop_plan = _state["multistop_plan"]

    # Combine action fields into a single text for the grader
    response_text = action.reasoning
    if action.chosen_route:
        response_text = f"I choose Route {action.chosen_route}. {response_text}"
    if action.stop_order:
        response_text += f"\nStop order: {' → '.join(action.stop_order)}"
    if action.arrival_times:
        times_str = ", ".join(f"{k}: {v}" for k, v in action.arrival_times.items())
        response_text += f"\nArrival times: {times_str}"
    if action.identified_violations:
        response_text += f"\nViolations identified: {'; '.join(action.identified_violations)}"

    # Grade
    grade_result = grade(
        task=task,
        agent_response=response_text,
        route_options=route_options,
        multistop_plan=multistop_plan,
    )

    reward = grade_result["reward"]
    _state.update({
        "completed": True,
        "reward": reward,
        "steps": 1,
        "last_action": action,
        "last_grade": grade_result,
    })

    # Update observation to mark done
    obs = _state["observation"]
    obs.done = True

    return StepResult(
        observation=obs,
        reward=reward,
        done=True,
        info={
            "grade": grade_result,
            "episode_id": _state["episode_id"],
            "task_id": task["task_id"],
            "difficulty": task["difficulty"],
        },
    )


@app.get("/state")
def state() -> EnvState:
    """Return current environment state (no side effects)."""
    if not _state["task"]:
        return EnvState(
            episode_id="none",
            task_id="none",
            difficulty="none",
            started=False,
            completed=False,
        )
    return EnvState(
        episode_id=_state["episode_id"],
        task_id=_state["task"]["task_id"],
        difficulty=_state["task"]["difficulty"],
        started=True,
        completed=_state["completed"],
        current_reward=_state["reward"],
        total_steps=_state["steps"],
        last_action=_state["last_action"],
        last_grade=_state["last_grade"],
        observation=_state["observation"],
    )


@app.get("/tasks")
def list_tasks():
    """List all available tasks with metadata."""
    return {
        "total": len(ALL_TASKS),
        "by_difficulty": {
            k: [{"task_id": t["task_id"], "scenario_preview": t["scenario"][:80] + "..."}
                for t in v]
            for k, v in {"easy": [], "medium": [], "hard": []}.items()
        },
        "tasks": [
            {
                "task_id": t["task_id"],
                "difficulty": t["difficulty"],
                "start": t["start"],
                "end": t["end"],
                "depart_time": t["depart_time"],
                "stops": t.get("stops", []),
                "num_constraints": len(t["constraints"]),
            }
            for t in ALL_TASKS
        ],
    }


# ─────────────────────────────────────────────
# WEB UI
# ─────────────────────────────────────────────

@app.get("/web", response_class=HTMLResponse)
def web_ui():
    """Interactive browser UI for testing the environment."""
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, "web_ui.html")
    try:
        return open(html_path, encoding="utf-8").read()
    except FileNotFoundError:
        return f"<h2>web_ui.html not found at: {html_path}</h2>"
from fastapi.responses import HTMLResponse
import os

@app.get("/", response_class=HTMLResponse)
def home():
    here = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(here, "web_ui.html")
    return open(html_path, encoding="utf-8").read()