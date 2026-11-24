"""Single source of truth for scoring, matching arc_agi/scorecard.py.

Our own implementation followed the paper formula and disagreed with the
official scorer in two ways, found by comparing a real Kaggle result (0.19)
with our local number (0.00206) for the identical agent:

  1. official multiplies by 100 and caps a level at 115 (not 1.15)
  2. official divides by the summed weights of the levels REACHED, not of every
     level in the game, and additionally caps the result at the weight fraction
     that scored above zero

(2) is the one that matters for strategy: not reaching a level costs you its
weight but does not dilute what you already earned, so reaching one more level
is worth more than our version implied.
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
