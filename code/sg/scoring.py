"""Scoring that matches arc_agi/scorecard.py.

  1. A level's score is multiplied by 100 and capped at 115.
  2. The sum is divided by the summed weights of the levels reached, not of every level in the game, and is
     capped at the weight fraction that scored above zero.
"""


def game_score(level_records):
    """level_records: list of {level, human_steps, agent_steps, completed}."""
    lv = []
    for rec in level_records:
        h, a = rec["human_steps"], rec["agent_steps"]
        if rec.get("completed") and a > 0:
            s = min((h / a) ** 2 * 100.0, 115.0)
        else:
            s = 0.0
        lv.append((rec["level"], s))
    if not lv:
        return 0.0
    denom = sum(w for w, _ in lv)
    if denom <= 0:
        return 0.0
    numer = sum(w * s for w, s in lv)
    max_w = sum(w for w, s in lv if s > 0)
    return min(numer / denom, max_w / denom * 100.0)
