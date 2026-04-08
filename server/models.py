"""
models.py
=========
Pydantic data models for the OpenEnv spec.

OpenEnv requires three typed models:
  - Action      : What the agent sends to the environment (step input)
  - Observation : What the environment returns to the agent (step output)
  - EnvState    : Internal environment state (returned by /state endpoint)

Using Pydantic ensures:
  - Automatic JSON serialisation/deserialisation
  - Type validation on every request
  - Auto-generated OpenAPI docs via FastAPI
  - Compatibility with OpenEnv's typed model system
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# ─────────────────────────────────────────────
# ACTION — what the agent sends
# ─────────────────────────────────────────────

class CommuteAction(BaseModel):
    """
    The agent's response to a commute task.

    For easy/medium tasks:  chosen_route is required ("A", "B", or "C")
    For hard tasks:         chosen_route is None; stop_order and arrival_times are used

    reasoning is ALWAYS required — the environment uses it for scoring.
    """
    chosen_route: Optional[str] = Field(
        default=None,
        description="Which route option the agent recommends: 'A', 'B', or 'C'",
        pattern="^[ABC]$"
    )
    reasoning: str = Field(
        description="Full explanation of why this route was chosen, "
                    "including identification of constraint violations in other routes.",
        min_length=10
    )
    stop_order: Optional[List[str]] = Field(
        default=None,
        description="For hard tasks: ordered list of intermediate stops, e.g. ['School', 'Pharmacy']"
    )
    arrival_times: Optional[Dict[str, str]] = Field(
        default=None,
        description="For hard tasks: estimated arrival time at each stop, e.g. {'School': '08:22'}"
    )
    identified_violations: Optional[List[str]] = Field(
        default=None,
        description="Optional: list of constraint violations the agent identified in rejected routes"
    )


# ─────────────────────────────────────────────
# OBSERVATION — what the environment returns
# ─────────────────────────────────────────────

class RouteOption(BaseModel):
    """One of the 3 route choices presented to the agent."""
    id: str
    label: str
    strategy: str
    path: List[str]
    total_time_min: float
    total_distance_km: float
    has_toll: bool
    road_types_used: List[str]
    via: str
    hops: int


class Constraint(BaseModel):
    """A single constraint the agent must satisfy."""
    type: str
    description: str
    raw: Dict[str, Any]


class CommuteObservation(BaseModel):
    """
    Full observation returned to the agent after reset() or step().

    Contains everything the agent needs:
      - The scenario (natural language)
      - Pre-computed route options
      - Constraints to satisfy
      - Traffic context
      - For hard tasks: the list of stops to order
    """
    task_id: str
    difficulty: str
    scenario: str
    start: str
    end: str
    depart_time: str
    time_slot: str                          # "morning" / "midday" / "evening" / "night"
    route_options: List[RouteOption]
    constraints: List[Constraint]
    stops: List[str]                        # Empty for easy; has values for hard
    city_context: Dict[str, Any]            # Node names, POI hours, traffic notes
    task_prompt: str                        # Full prompt string ready for the LLM
    done: bool = False


# ─────────────────────────────────────────────
# STEP RESULT — wraps observation + reward
# ─────────────────────────────────────────────

class StepResult(BaseModel):
    """Returned by the /step endpoint."""
    observation: CommuteObservation
    reward: float = Field(ge=0.0, le=1.0, description="Episode reward, 0.0–1.0")
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)  # Includes grade breakdown


# ─────────────────────────────────────────────
# ENV STATE — internal state for /state endpoint
# ─────────────────────────────────────────────

class EnvState(BaseModel):
    """
    Internal environment state.
    Returned by the /state endpoint.
    Also used for the /reset endpoint response.
    """
    episode_id: str
    task_id: str
    difficulty: str
    started: bool
    completed: bool
    current_reward: float = 0.0
    total_steps: int = 0
    last_action: Optional[CommuteAction] = None
    last_grade: Optional[Dict[str, Any]] = None
    observation: Optional[CommuteObservation] = None