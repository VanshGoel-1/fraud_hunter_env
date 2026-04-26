"""Fraud Hunter Env — EnvClient subclass for training-side use."""

from __future__ import annotations

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import FraudHunterAction, FraudHunterObservation


class FraudHunterEnv(EnvClient[FraudHunterAction, FraudHunterObservation, State]):
    """
    WebSocket client for the Fraud Hunter Env server.

    Example:
        >>> with FraudHunterEnv(base_url="http://localhost:8000") as env:
        ...     r = env.reset()
        ...     r = env.step(FraudHunterAction(kind="query_corporate", entity_name="Acme"))
        ...     print(r.reward, r.observation.tool_output)
    """

    def _step_payload(self, action: FraudHunterAction) -> Dict[str, Any]:
        # Pydantic v2 serialisation, dropping None fields for compact WS frames.
        return action.model_dump(exclude_none=True, mode="json")

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[FraudHunterObservation]:
        obs_data = payload.get("observation", {}) or {}
        observation = FraudHunterObservation(
            case_brief=obs_data.get("case_brief"),
            tool_output=obs_data.get("tool_output"),
            base64_document=obs_data.get("base64_document"),
            grader_feedback=obs_data.get("grader_feedback"),
            evidence_graph=obs_data.get("evidence_graph"),
            step_count=obs_data.get("step_count", 0),
            budget_remaining=obs_data.get("budget_remaining", 0),
            difficulty_tier=obs_data.get("difficulty_tier", 1),
            info=obs_data.get("info"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
