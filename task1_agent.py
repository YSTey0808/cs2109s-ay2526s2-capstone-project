import heapq
import itertools
from dataclasses import dataclass, replace
from grid_adventure.grid import GridState
from grid_adventure.entities import (
    AgentEntity, WallEntity, ExitEntity, CoinEntity, GemEntity,
    KeyEntity, LockedDoorEntity, LavaEntity,
    BoxEntity, SpeedPowerUpEntity, ShieldPowerUpEntity, PhasingPowerUpEntity
)
from grid_adventure.step import Action

MOVES = {
    
    Action.RIGHT: (1,  0),
    Action.LEFT:  (-1, 0),
    Action.DOWN:  (0,  1),
    Action.UP:    (0, -1),
}

@dataclass(frozen=True)
class GameState:
    turn: int
    pos: tuple
    gems: frozenset
    coins: frozenset
    keys: int
    doors: frozenset
    keys_on_floor: frozenset 
    boxes: frozenset 
    boots: frozenset
    shields: frozenset
    ghosts: frozenset
    

@dataclass(frozen=True)
class AgentState:
    health: int
    key_count: int
    has_shield: bool
    shield_uses: int
    has_ghost: bool
    ghost_turns: int
    has_boot: bool
    boot_turns: int

class Agent:
    def __init__(self):
        self.plan = []

    def step(self, state: GridState) -> Action:
        if not self.plan:
            self.plan = self._search(state)
        if self.plan:
            return self.plan.pop(0)
        return Action.WAIT
        

    def _search(self, state: GridState) -> list[Action]:
        info = self._parse_gridstate(state)
        counter = itertools.count()
        start_agentstate, start_gamestate = self._make_start_state(info)
        heap = [(0, next(counter), 0, start_gamestate, start_agentstate, [])]
        visited = set()

        while heap:
            f, _, g, s, agent, path = heapq.heappop(heap)

            if s.turn > info['turn_limit']:
                continue
                
            key = replace(s, coins=frozenset())
            state_key = (key, agent)
            if state_key in visited:
                continue
            visited.add(state_key)

            if self._is_goal(s, info):
                return path

            for cost_delta, new_s, new_agent, action in self._get_successors(s, agent, info, state):
                new_g = g + cost_delta
                new_h = self._heuristic(new_s,new_agent,info)
                heapq.heappush(heap, (new_g + new_h, next(counter), new_g, new_s, new_agent, path + [action]))
        
        return []

    def _make_start_state(self, info) -> tuple[AgentState, GameState]:
        agent = info['agent_state']
        return (agent, GameState(
            0,
            pos=info['agent_pos'],
            gems=frozenset(info['gem_pos']),
            coins=frozenset(info['coin_pos']),
            keys=agent.key_count,
            doors=frozenset(info['locked_door_pos']),
            keys_on_floor=frozenset(info['key_pos']),
            boxes=frozenset(info['boxes']),
            boots=frozenset(info['speed_powerup_pos']),
            shields=frozenset(info['shield_powerup_pos']),
            ghosts=frozenset(info['phasing_powerup_pos']),
        ))
    

    def _is_goal(self, s: GameState, info: dict) -> bool:
        return s.pos == info['exit_pos'] and len(s.gems) == 0 and s.turn <= info['turn_limit']

    def _heuristic(self, s: GameState, agent: AgentState, info: dict) -> int:
        if s.pos == info['exit_pos'] and len(s.gems) == 0:
            return 0

        # MST connecting agent pos, all remaining gems, and exit
        nodes = [s.pos] + list(s.gems) + [info['exit_pos']]
        base = self._mst_weight(nodes) + len(s.gems) * 3

        # Powerups that help with movement speed
        if agent.has_boot:
            base = base / 2

        # Time pressure: when remaining turns are tight, heavily penalize
        # being far from goal to discourage any non-essential detours (coins)
        remaining = info['turn_limit'] - s.turn
        if remaining > 0:
            slack = remaining - base
            if slack < 5:
                base += max(0, 5 - slack) * 3

        return max(0, base)


    def _mst_weight(self, positions: list) -> int:
        if len(positions) <= 1:
            return 0
        in_mst = {positions[0]}
        rest = set(positions[1:])
        total = 0
        while rest:
            dist, nearest = min(
                (self._manhattan(u, v), v)
                for u in in_mst for v in rest
            )
            total += dist
            in_mst.add(nearest)
            rest.remove(nearest)
        return total

    def _manhattan(self, a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _apply_lava_damage(self, agent: AgentState):
        """Apply lava damage to agent. Returns updated agent, or None if agent would die."""
        if agent.has_ghost:
            return agent  # phasing prevents damage
        if agent.has_shield:
            new_agent = replace(agent, shield_uses=agent.shield_uses - 1)
            if new_agent.shield_uses <= 0:
                new_agent = replace(new_agent, has_shield=False)
            return new_agent
        if agent.health <= 2:
            return None  # would die
        return replace(agent, health=agent.health - 2)

    def _get_successors(self, s: GameState, agent: AgentState, info: dict, state: GridState):
        successors = []
        #Update agent status for powerup turns
        new_agent = agent


        s = replace(s, turn=s.turn + 1)
        
        # Update powerup turns and expire if needed
        if agent.has_ghost and agent.ghost_turns > 0:
            new_agent = replace(new_agent, ghost_turns=agent.ghost_turns - 1)
            if new_agent.ghost_turns == 0:
                new_agent = replace(new_agent, has_ghost=False)
        if agent.has_boot and agent.boot_turns > 0:
            new_agent = replace(new_agent, boot_turns=agent.boot_turns - 1)
            if new_agent.boot_turns == 0:
                new_agent = replace(new_agent, has_boot=False)  
        


        # Pick up gem
        if s.pos in s.gems:
            successors.append((3, replace(s, gems=s.gems - {s.pos}), new_agent, Action.PICK_UP))

        # Pick up coin
        if s.pos in s.coins:
            successors.append((-2, replace(s, coins=s.coins - {s.pos}), new_agent, Action.PICK_UP))

        if s.pos in s.keys_on_floor:
            new_s = replace(s, keys=s.keys + 1, keys_on_floor=s.keys_on_floor - {s.pos})
            successors.append((3, new_s, new_agent, Action.PICK_UP))
        

        # Use key on adjacent locked door
        if s.keys > 0:
            for door_pos in s.doors:
                if abs(s.pos[0] - door_pos[0]) + abs(s.pos[1] - door_pos[1]) == 1:
                    new_state = replace(s, keys=s.keys - 1, doors=s.doors - {door_pos})
                    successors.append((3, new_state, new_agent, Action.USE_KEY))

        #because of the ticker set to 6
        if s.pos in s.boots:
            pickup_agent = replace(new_agent, has_boot=True, boot_turns= 6)
            new_state = replace(s, boots=s.boots - {s.pos})
            successors.append((3, new_state, pickup_agent, Action.PICK_UP))

        if s.pos in s.shields:
            pickup_agent = replace(new_agent, has_shield=True, shield_uses = new_agent.shield_uses + 5)
            new_state = replace(s, shields=s.shields - {s.pos})
            successors.append((3, new_state , pickup_agent, Action.PICK_UP))

        #because of the ticker set to 6
        if s.pos in s.ghosts:
            pickup_agent = replace(new_agent, has_ghost=True, ghost_turns=6)
            new_state = replace(s, ghosts=s.ghosts - {s.pos})
            successors.append((3, new_state, pickup_agent, Action.PICK_UP))

        # Move
        for action, (dx, dy) in MOVES.items():
            move_agent = new_agent

            if move_agent.has_boot:
                # Boots: engine does 2 sequential substeps, each with damage
                mid = (s.pos[0] + dx, s.pos[1] + dy)
                far = (s.pos[0] + 2*dx, s.pos[1] + 2*dy)

                # Substep 1: mid must be in bounds
                if (mid[0] < 0 or mid[1] < 0 or
                    mid[0] >= state.width or mid[1] >= state.height):
                    continue

                # mid must be passable (walls/doors block, unless ghost)
                if (mid in info['walls'] or mid in s.doors) and not move_agent.has_ghost:
                    continue
                
                if mid in info['lava_pos']:
                    move_agent = self._apply_lava_damage(move_agent)
                    if move_agent is None:
                        continue
                    
                if mid in s.boxes:
                    pushed_to = far
                    if (0 <= far[0] < state.width and 0 <= far[1] < state.height
                        and far not in info['walls'] and far not in s.boxes and far not in s.doors):
                        new_boxes = s.boxes - {mid} | {far}
                        successors.append((3, replace(s, pos=mid, boxes=new_boxes), move_agent, action))
                    continue

                # Substep 2: check if far is reachable
                far_ok = (0 <= far[0] < state.width and 0 <= far[1] < state.height
                          and (move_agent.has_ghost or (far not in info['walls'] and far not in s.doors)))
                
                
                if far_ok:
                    new_pos = far
                    # far lava damage
                    if far in info['lava_pos']:
                        move_agent = self._apply_lava_damage(move_agent)
                        if move_agent is None:
                            continue
                else:
                    new_pos = mid  # blocked at second step, stop at mid

                # Box at final position
                if new_pos in s.boxes:
                    pushed_to = (new_pos[0] + dx, new_pos[1] + dy)
                    if (pushed_to[0] < 0 or pushed_to[1] < 0 or
                        pushed_to[0] >= state.width or pushed_to[1] >= state.height or
                        pushed_to in info['walls'] or pushed_to in s.boxes or pushed_to in s.doors):
                        if far_ok:
                            new_pos = mid  # can't push at far, fall back to mid
                        else:
                            continue
                    else:
                        successors.append((3, replace(s, pos=new_pos, boxes=s.boxes - {new_pos} | {pushed_to}), move_agent, action))
                        continue

                successors.append((3, replace(s, pos=new_pos), move_agent, action))
                continue
            else:
                # Normal movement (no boots)
                new_pos = (s.pos[0] + dx, s.pos[1] + dy)

            if (new_pos[0] < 0 or new_pos[1] < 0 or
                new_pos[0] >= state.width or new_pos[1] >= state.height):
                continue

            if (new_pos in info['walls'] or new_pos in s.doors) and not move_agent.has_ghost:
                continue

            # Lava damage
            if new_pos in info['lava_pos']:
                move_agent = self._apply_lava_damage(move_agent)
                if move_agent is None:
                    continue

            # Box pushing
            if new_pos in s.boxes:
                pushed_to = (new_pos[0] + dx, new_pos[1] + dy)
                if (pushed_to[0] < 0 or pushed_to[1] < 0 or
                    pushed_to[0] >= state.width or pushed_to[1] >= state.height or
                    pushed_to in info['walls'] or pushed_to in s.boxes or pushed_to in s.doors):
                    continue
                successors.append((3, replace(s, pos=new_pos, boxes=s.boxes - {new_pos} | {pushed_to}), move_agent, action))
                continue

            successors.append((3, replace(s, pos=new_pos), move_agent, action))

        return successors

    def _parse_gridstate(self, state: GridState) -> dict:
        info = {
            'agent_pos': None, 'agent_state': None,
            'exit_pos': None, 'turn_limit': state.turn_limit,
            'gem_pos': [], 'coin_pos': [], 'key_pos': [],
            'locked_door_pos': [], 'lava_pos': set(),
            'walls': set(), 'boxes': set(),
            'speed_powerup_pos': [], 'shield_powerup_pos': [], 'phasing_powerup_pos': [],
        }
        for x in range(state.width):
            for y in range(state.height):
                for entity in state.objects_at((x, y)):
                    if isinstance(entity, AgentEntity):
                        info['agent_pos'] = (x, y)
                        inv = entity.inventory_list
                        sts = entity.status_list
                        # shield_uses = next((s.usage_limit.amount for s in sts if isinstance(s, ShieldPowerUpEntity)), 0)
                        # ghost_turns = next((s.time_limit.amount for s in sts if isinstance(s, PhasingPowerUpEntity)), 0)
                        # boot_turns  = next((s.time_limit.amount for s in sts if isinstance(s, SpeedPowerUpEntity)), 0)  
                        shield_uses = 0
                        ghost_turns = 0
                        boot_turns = 0        
                        info['agent_state'] = AgentState(
                            health=entity.health.current_health,
                            key_count=sum(1 for i in inv if isinstance(i, KeyEntity)),
                            has_shield=shield_uses > 0, shield_uses=shield_uses,
                            has_ghost=ghost_turns > 0,   ghost_turns=ghost_turns,
                            has_boot=boot_turns > 0,     boot_turns=boot_turns,
                        )
                    elif isinstance(entity, ExitEntity):          info['exit_pos'] = (x, y)
                    elif isinstance(entity, GemEntity):           info['gem_pos'].append((x, y))
                    elif isinstance(entity, CoinEntity):          info['coin_pos'].append((x, y))
                    elif isinstance(entity, KeyEntity):           info['key_pos'].append((x, y))
                    elif isinstance(entity, LockedDoorEntity):    info['locked_door_pos'].append((x, y))
                    elif isinstance(entity, LavaEntity):          info['lava_pos'].add((x, y))
                    elif isinstance(entity, WallEntity):          info['walls'].add((x, y))
                    elif isinstance(entity, BoxEntity):           info['boxes'].add((x, y))
                    elif isinstance(entity, SpeedPowerUpEntity):  info['speed_powerup_pos'].append((x, y))
                    elif isinstance(entity, ShieldPowerUpEntity): info['shield_powerup_pos'].append((x, y))
                    elif isinstance(entity, PhasingPowerUpEntity):info['phasing_powerup_pos'].append((x, y))
        return info
