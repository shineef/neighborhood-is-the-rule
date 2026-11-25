"""Uniform random agent.

Samples uniformly over ACTION1 to ACTION5 and ACTION6 with a uniform click on the 64 x 64 grid.
"""
import random
from typing import Any

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState


class RandomAgent(Agent):
    MAX_ACTIONS = float('inf')

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.simple = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                       GameAction.ACTION4, GameAction.ACTION5]

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
            action.reasoning = "Game needs reset."
            return action

        # Restrict to what the level actually offers when it tells us; the
        # field is list[int] in this arcengine build.
        avail = list(getattr(latest_frame, "available_actions", []) or [])
        ids = [a.value if hasattr(a, "value") else a for a in avail]
        if not ids:
            ids = [1, 2, 3, 4, 5, 6]

        choice = random.choice(ids)
        if choice == 6:
            action = GameAction.ACTION6
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = "random click"
        else:
            action = GameAction.from_id(choice)
            action.reasoning = "random"
        return action
