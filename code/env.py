import json
import numpy as np
from typing import Dict, Any, Tuple, Union

# Action space type
Action = Union[str, Tuple[int, int]]
VALID_KEYS = {"UP", "DOWN", "LEFT", "RIGHT", "INTERACT", "UNDO"}

class ARCEnvironment:
    """
    Wrapper around arc_agi.Arcade() that runs one ARC-AGI-3 game offline.
    Action Space: 5 directional/interaction keys + Undo + 1 Click(x, y).
    """
    def __init__(self, level_data: Dict[str, Any], level_id_str: str):
        self.level_data = level_data
        self.level_id_str = level_id_str

        # Hard cutoff at 5x human median steps (5n)
        # self.human_median_steps = self.level_data.get("human_median_steps", 10)
        # self.max_steps = 5 * self.human_median_steps
        #
        # The above collapsed a game's per-level baselines to one median and
        # spent that budget on ALL levels at once, which made every one of the
        # 25 games unwinnable: cd82's baselines are [55,8,41,21,23,23] (171
        # total) but the budget was 5*median = 115. The official rule is "a
        # hard cutoff of 5x human performance *per level*" (ARC-AGI-3 technical
        # report), so the budget resets on every level.
        self.baseline_actions = list(self.level_data.get("baseline_actions", []) or [])
        if not self.baseline_actions:
            self.baseline_actions = [self.level_data.get("human_median_steps", 10)]

        from arc_agi import Arcade
        # Instantiate the arcade wrapper natively to avoid obfuscated arcengine methods
        self._arc = Arcade()
        self.arc_env = self._arc.make(self.level_id_str)

        self.steps_taken = 0
        self.is_done = False
        self.is_success = False

    def _budget_for(self, level_idx: int) -> int:
        """5x this level's own human baseline. Levels past the known baselines
        (shouldn't happen) fall back to the last known one."""
        if level_idx < len(self.baseline_actions):
            return 5 * self.baseline_actions[level_idx]
        return 5 * self.baseline_actions[-1]

    @property
    def human_median_steps(self) -> int:
        """Kept so existing callers/logs don't break; now reports the current
        level's baseline rather than a game-wide median."""
        idx = min(self.current_level, len(self.baseline_actions) - 1)
        return self.baseline_actions[idx]

    @property
    def max_steps(self) -> int:
        return self._budget_for(self.current_level)

    def reset(self) -> Any:
        self.steps_taken = 0
        self.is_done = False
        self.is_success = False

        # Per-level bookkeeping. level_steps counts actions spent on the level
        # currently being played; level_records accumulates (baseline, actions,
        # completed) once per finished level so eval.py can score each one.
        self.current_level = 0
        self.level_steps = 0
        self.level_records = []

        from arcengine import GameAction
        # Issue a reset action to get the initial observation frame
        raw_frame = self.arc_env.step(GameAction.RESET)
        return raw_frame

    def step(self, action: Action) -> Tuple[Any, float, bool, Dict[str, Any]]:
        if self.is_done:
            return None, 0.0, True, {"error": "Environment already done."}
            
        from arcengine import GameAction
        arc_action = None
        data = {}
        
        # Accept arcengine.GameAction directly (for native agents like MyAgent)
        if isinstance(action, GameAction):
            arc_action = action
            if hasattr(action, "action_data") and action.action_data:
                data = action.action_data.model_dump()
            else:
                data = getattr(action, "data", {})
        elif isinstance(action, str):
            if action not in VALID_KEYS:
                raise ValueError(f"Invalid key action: {action}. Must be one of {VALID_KEYS}")
            # Map keys to official GameAction IDs
            if action == "UP": arc_action = GameAction.from_id(1)
            elif action == "DOWN": arc_action = GameAction.from_id(2)
            elif action == "LEFT": arc_action = GameAction.from_id(3)
            elif action == "RIGHT": arc_action = GameAction.from_id(4)
            elif action == "INTERACT": arc_action = GameAction.from_id(5)
            elif action == "UNDO": arc_action = GameAction.from_id(7)
        elif isinstance(action, tuple):
            if len(action) != 2:
                raise ValueError("Click action must be a tuple of (x, y)")
            x, y = action
            arc_action = GameAction.from_id(6)
            data = {"x": x, "y": y}
        else:
            raise ValueError(f"Invalid action type: {type(action)}")
            
        raw_frame = self.arc_env.step(arc_action, data=data)

        self.steps_taken += 1
        self.level_steps += 1

        # Real completion logic using arcengine FrameData Raw
        from arcengine import GameState

        state = getattr(raw_frame, 'state', None)
        completed = int(getattr(raw_frame, 'levels_completed', 0) or 0)

        reward = 0.0

        # Did this action finish a level? levels_completed is the engine's own
        # counter, so this catches the last level (which also sets WIN) and
        # every intermediate one.
        if completed > self.current_level:
            # Credit every level closed out by this step. Normally one.
            for lvl in range(self.current_level, completed):
                self.level_records.append({
                    "level": lvl + 1,  # 1-indexed, this is the RHAE weight w_l
                    "human_steps": self.baseline_actions[lvl] if lvl < len(self.baseline_actions) else self.baseline_actions[-1],
                    "agent_steps": self.level_steps,
                    "completed": True,
                })
            reward = 1.0
            self.current_level = completed
            self.level_steps = 0  # fresh budget for the next level

        if state == GameState.WIN:
            self.is_done = True
            self.is_success = True

        # Hard cutoff enforced - per level, not per game (official rule is
        # "5x human performance per level").
        elif self.level_steps >= self.max_steps:
            # Record the level that ran out of budget as a failure, then stop:
            # levels are sequential, so everything after is unreachable.
            self.level_records.append({
                "level": self.current_level + 1,
                "human_steps": self.human_median_steps,
                "agent_steps": self.level_steps,
                "completed": False,
            })
            self.is_done = True
            self.is_success = False

        info = {
            "steps_taken": self.steps_taken,
            "level_steps": self.level_steps,
            "current_level": self.current_level + 1,
            "total_levels": len(self.baseline_actions),
            "human_median_steps": self.human_median_steps,
            "is_success": self.is_success,
            "level_records": self.level_records,
        }

        return raw_frame, reward, self.is_done, info

import os
import statistics

def load_dataset(dataset_dir: str) -> Dict[str, Any]:
    """Loads the games from the downloaded folder structure."""
    dataset = {}
    if not os.path.exists(dataset_dir):
        print(f"Warning: {dataset_dir} not found. Using mock dataset.")
        return {
            "level_1": {"human_median_steps": 5, "grid": []},
            "level_2": {"human_median_steps": 12, "grid": []},
            "level_3": {"human_median_steps": 8, "grid": []}
        }
        
    for group in os.listdir(dataset_dir):
        group_path = os.path.join(dataset_dir, group)
        if not os.path.isdir(group_path):
            continue
            
        for task_id in os.listdir(group_path):
            task_path = os.path.join(group_path, task_id)
            metadata_file = os.path.join(task_path, "metadata.json")
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                    
                    baseline = meta.get("baseline_actions", [])
                    if baseline:
                        median_steps = int(statistics.median(baseline))
                    else:
                        median_steps = 10

                    game_id = meta.get("game_id", f"{group}-{task_id}")
                    dataset[game_id] = {
                        # The per-level human baselines - this is what the 5x
                        # cutoff and the per-level RHAE term are computed from.
                        # The median is kept only for legacy callers/logging.
                        "baseline_actions": baseline,
                        "human_median_steps": median_steps,
                        "grid": [] # Placeholder for actual grid if needed
                    }
                except Exception as e:
                    print(f"Error loading {metadata_file}: {e}")
                    
    return dataset
