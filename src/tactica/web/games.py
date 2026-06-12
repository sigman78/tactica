"""Interactive human-vs-agent battles for the dashboard.

A :class:`GameSession` holds a live :class:`Battle` plus one seeded agent
for the non-human side. The human submits one action id at a time; after
each human action (and right after creation) the agent auto-plays until it
is the human's turn again or the battle ends, so the client only ever sees
two states: "your-turn" or "over".

Sessions are in-memory and single-process, like dashboard jobs.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from tactica.actions import Action, ActionType, BOARD_H, BOARD_W, cell_xy
from tactica.agents import Agent, make_agent
from tactica.battle import (
    DAMAGE_MOD_MAX,
    DAMAGE_MOD_MIN,
    DAMAGE_MOD_PER_POINT,
    RANGED_MELEE_PENALTY,
    Battle,
    Stack,
)
from tactica.eval.runner import GameRecord, derive_seed, write_jsonl
from tactica.scenario import load_scenario
from tactica.units import GLYPHS

MAX_SESSIONS = 50
SIDE_NAME = ("gold", "blue")


def _expected_damage(attacker: Stack, defender: Stack, melee: bool) -> int:
    """Average-roll damage preview. Mirrors Battle.compute_damage but never
    touches the battle RNG, so previews don't perturb the game."""
    stats = attacker.stats
    base = stats.avg_dmg * attacker.count
    diff = stats.attack - defender.effective_defense()
    factor = min(max(1.0 + DAMAGE_MOD_PER_POINT * diff, DAMAGE_MOD_MIN),
                 DAMAGE_MOD_MAX)
    if melee and stats.is_ranged:
        factor *= RANGED_MELEE_PENALTY
    return max(1, int(base * factor))


def _stack_label(s: Stack) -> str:
    return f"{s.stats.name} ×{s.count} ({SIDE_NAME[s.side]})"


@dataclass
class GameSession:
    id: str
    battle: Battle
    agent: Agent
    agent_spec: str
    human_side: int
    scenario_name: str
    seed: int
    actions: list[int] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # structured, for UI animation
    created: float = field(default_factory=time.time)
    saved_to: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ #
    # stepping with a narrated log

    def _narrate_step(self, action: Action) -> None:
        b = self.battle
        actor = b.active_stack()
        actor_label = _stack_label(actor)
        before = {s.uid: (s.count, s.total_hp, _stack_label(s))
                  for s in b.stacks.values() if s.alive}
        rnd = b.round
        event: dict = {"uid": actor.uid, "side": actor.side}

        if action.type == ActionType.WAIT:
            desc = f"{actor_label} waits"
            event["t"] = "wait"
        elif action.type == ActionType.DEFEND:
            desc = f"{actor_label} defends"
            event["t"] = "defend"
        elif action.type == ActionType.MOVE:
            x, y = cell_xy(action.target_cell)
            desc = f"{actor_label} moves to ({x},{y})"
            event["t"] = "move"
        else:
            target = next(s for s in b.stacks.values()
                          if s.alive and s.cell == action.target_cell)
            verb = "shoots" if action.type == ActionType.RANGED_ATTACK else "strikes"
            desc = f"{actor_label} {verb} {_stack_label(target)}"
            event["t"] = "attack"
            event["target"] = target.uid
            event["melee"] = action.type == ActionType.MELEE_ATTACK

        b.step(action)
        self.actions.append(action.id)
        ax, ay = cell_xy(actor.cell)  # post-step: includes the melee approach
        event["to"] = {"x": ax, "y": ay}

        effects: list[dict] = []
        strings: list[str] = []
        for uid, (count0, hp0, label) in before.items():
            s = b.stacks[uid]
            dmg = hp0 - s.total_hp
            if dmg <= 0:
                continue
            killed = count0 - s.count
            effects.append({"uid": uid, "dmg": dmg, "killed": killed,
                            "count": s.count, "top_hp": s.top_hp,
                            "dead": not s.alive})
            strings.append(f"{label.split(' (')[0]} -{dmg}hp"
                           + (f", {killed} slain" if killed else ""))
        if event["t"] == "attack":
            event["effects"] = effects
        self.events.append(event)

        line = f"R{rnd} · {desc}"
        if strings:
            line += ": " + ", ".join(strings)
        if b.is_terminal():
            w = b.winner()
            line += " — " + ("draw" if w is None
                                  else f"{SIDE_NAME[w]} side wins!")
        self.log.append(line)

    def _agent_loop(self) -> None:
        b = self.battle
        while not b.is_terminal() and b.current_player() != self.human_side:
            self._narrate_step(self.agent.act(b))

    def act(self, action_id: int) -> None:
        b = self.battle
        if b.is_terminal():
            raise ValueError("battle is over")
        if b.current_player() != self.human_side:
            raise ValueError("not your turn")
        legal = {a.id for a in b.legal_actions()}
        if action_id not in legal:
            raise ValueError(f"illegal action id {action_id}")
        self._narrate_step(Action.from_id(action_id))
        self._agent_loop()

    # ------------------------------------------------------------------ #
    # state for the client

    def _legal_payload(self) -> list[dict]:
        b = self.battle
        if b.is_terminal() or b.current_player() != self.human_side:
            return []
        actor = b.active_stack()
        out = []
        for a in b.legal_actions():
            entry: dict = {"id": a.id, "type": a.type.name}
            if a.type in (ActionType.MOVE, ActionType.MELEE_ATTACK,
                          ActionType.RANGED_ATTACK):
                x, y = cell_xy(a.target_cell)
                entry.update(x=x, y=y)
            if a.type in (ActionType.MELEE_ATTACK, ActionType.RANGED_ATTACK):
                target = next(s for s in b.stacks.values()
                              if s.alive and s.cell == a.target_cell)
                melee = a.type == ActionType.MELEE_ATTACK
                entry["est"] = _expected_damage(actor, target, melee)
                entry["target_uid"] = target.uid
                if melee:
                    entry["retaliates"] = target.retaliations_left > 0
                    approach = b._melee_approach(actor, target)
                    if approach is not None and approach != actor.cell:
                        ax, ay = cell_xy(approach)
                        entry["from"] = {"x": ax, "y": ay}
            out.append(entry)
        return out

    def _queue_payload(self) -> list[dict]:
        b = self.battle
        if b.is_terminal():
            return []
        rows = []
        for uid, waiting in ([(u, False) for u in b.queue]
                             + [(u, True) for u in b.waiters]):
            s = b.stacks[uid]
            rows.append({"uid": uid, "side": s.side, "unit": s.stats.name,
                         "glyph": GLYPHS[s.unit_type], "count": s.count,
                         "waiting": waiting})
        return rows

    def state(self) -> dict:
        b = self.battle
        over = b.is_terminal()
        return {
            "id": self.id,
            "status": "over" if over else "your-turn",
            "scenario": self.scenario_name,
            "seed": self.seed,
            "agent": self.agent_spec,
            "human_side": self.human_side,
            "round": b.round,
            "winner": b.winner() if over else None,
            "you_won": (b.winner() == self.human_side) if over else None,
            "board": {"w": BOARD_W, "h": BOARD_H},
            "obstacles": [{"x": x, "y": y} for x, y in
                          (cell_xy(c) for c in sorted(b.scenario.obstacles))],
            "stacks": [
                {"uid": s.uid, "side": s.side, "unit": s.stats.name,
                 "glyph": GLYPHS[s.unit_type], "count": s.count,
                 "top_hp": s.top_hp, "max_hp": s.stats.hp,
                 "x": cell_xy(s.cell)[0], "y": cell_xy(s.cell)[1],
                 "defending": s.defending,
                 "speed": s.stats.speed, "initiative": s.stats.initiative,
                 "ranged": s.stats.is_ranged,
                 "impaired": s.stats.is_ranged and b._enemy_adjacent(s)}
                for s in b.stacks.values() if s.alive],
            "active": None if over else b.active_stack().uid,
            "queue": self._queue_payload(),
            "legal": self._legal_payload(),
            "log": self.log,
            "events": self.events,
            "n_actions": len(self.actions),
            "saved_to": self.saved_to,
        }

    def summary(self) -> dict:
        over = self.battle.is_terminal()
        return {"id": self.id, "scenario": self.scenario_name,
                "agent": self.agent_spec, "human_side": self.human_side,
                "seed": self.seed, "round": self.battle.round,
                "status": "over" if over else "your-turn",
                "created": self.created}

    def record(self) -> GameRecord:
        b = self.battle
        specs = ("human", self.agent_spec) if self.human_side == 0 \
            else (self.agent_spec, "human")
        agents = tuple({"type": "human"} if s == "human"
                       else self.agent.config() for s in specs)
        return GameRecord(
            scenario_name=self.scenario_name,
            scenario=b.scenario.to_dict(),
            seed=self.seed,
            specs=specs,
            agents=agents,  # type: ignore[arg-type]
            actions=list(self.actions),
            winner=b.winner(),
            rounds=b.round,
            state_hash=b.state_hash(),
        )


class GameManager:
    def __init__(self, replays_dir: str | Path = "replays") -> None:
        self.sessions: dict[str, GameSession] = {}
        self.replays_dir = Path(replays_dir)
        self._lock = threading.Lock()

    def create(self, agent_spec: str, scenario_name: str, seed: int,
               human_side: int, deterministic: bool = False) -> GameSession:
        if human_side not in (0, 1):
            raise ValueError("human_side must be 0 or 1")
        scenario = load_scenario(scenario_name,
                                 deterministic=deterministic or None)
        battle = Battle.from_scenario(scenario, seed)
        agent = make_agent(agent_spec,
                           seed=derive_seed(seed, "human-game", agent_spec))
        session = GameSession(
            id=uuid.uuid4().hex[:12], battle=battle, agent=agent,
            agent_spec=agent_spec, human_side=human_side,
            scenario_name=scenario.name, seed=seed)
        session._agent_loop()  # agent may open the battle
        with self._lock:
            self.sessions[session.id] = session
            self._prune()
        return session

    def _prune(self) -> None:
        if len(self.sessions) <= MAX_SESSIONS:
            return
        finished = sorted((s for s in self.sessions.values()
                           if s.battle.is_terminal()), key=lambda s: s.created)
        for s in finished[:len(self.sessions) - MAX_SESSIONS]:
            del self.sessions[s.id]

    def get(self, game_id: str) -> GameSession:
        try:
            return self.sessions[game_id]
        except KeyError:
            raise KeyError(f"no such game {game_id!r}") from None

    def list(self) -> list[dict]:
        return [s.summary() for s in
                sorted(self.sessions.values(), key=lambda s: -s.created)]

    def save_replay(self, game_id: str) -> str:
        session = self.get(game_id)
        with session.lock:
            if not session.battle.is_terminal():
                raise ValueError("battle is not over yet")
            self.replays_dir.mkdir(parents=True, exist_ok=True)
            path = self.replays_dir / f"human-{session.id}.jsonl"
            write_jsonl(path, [session.record().to_dict()])
            session.saved_to = path.as_posix()
            return session.saved_to
