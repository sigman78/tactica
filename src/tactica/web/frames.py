"""Re-simulate a logged game into JSON-friendly board frames for the viewer."""
from __future__ import annotations

from tactica.actions import BOARD_H, BOARD_W, Action, cell_xy
from tactica.battle import Battle, RULES_VERSION
from tactica.scenario import Scenario
from tactica.units import GLYPHS


def _snapshot(battle: Battle, action: str | None, actor: int | None) -> dict:
    active = None
    if not battle.is_terminal():
        active = battle.active_stack().uid
    stacks = []
    for s in battle.stacks.values():
        if not s.alive:
            continue
        x, y = cell_xy(s.cell)
        stacks.append({
            "uid": s.uid, "side": s.side, "unit": s.stats.name,
            "glyph": GLYPHS[s.unit_type], "count": s.count,
            "top_hp": s.top_hp, "max_hp": s.stats.hp,
            "x": x, "y": y, "defending": s.defending,
            "impaired": s.stats.is_ranged and battle._enemy_adjacent(s),
        })
    return {"round": battle.round, "action": action, "actor": actor,
            "active": active, "stacks": stacks}


def game_frames(record: dict) -> dict:
    """One frame per action (plus the initial deployment), with stack
    positions/HP and the acting stack highlighted. Verifies the final
    state hash like ``tactica replay`` does."""
    if int(record.get("rules_version", 1)) != RULES_VERSION:
        raise ValueError(
            f"replay recorded under rules_version "
            f"{record.get('rules_version', 1)}, current is {RULES_VERSION}; "
            f"action ids would mis-decode -- regenerate the replay")
    scenario = Scenario.from_dict(record["scenario"])
    battle = Battle.from_scenario(scenario, int(record["seed"]))
    frames = [_snapshot(battle, None, None)]
    for action_id in record["actions"]:
        actor = battle.active_stack().uid
        action = Action.from_id(int(action_id))
        battle.step(action)
        frames.append(_snapshot(battle, repr(action), actor))
    final_hash = battle.state_hash()
    if final_hash != record["state_hash"]:
        raise AssertionError(
            f"replay diverged: hash {final_hash} != logged {record['state_hash']}")
    return {
        "scenario": record["scenario_name"],
        "specs": list(record["specs"]),
        "seed": record["seed"],
        "winner": record["winner"],
        "rounds": record["rounds"],
        "board": {"w": BOARD_W, "h": BOARD_H},
        "obstacles": [{"x": x, "y": y} for x, y in
                      (cell_xy(c) for c in sorted(scenario.obstacles))],
        "frames": frames,
    }
