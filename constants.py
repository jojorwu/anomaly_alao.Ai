from typing import Dict, Set

# Hot callbacks that run frequently
HOT_CALLBACKS: Set[str] = frozenset({
    'actor_on_update', 'actor_on_first_update',
    'npc_on_update', 'monster_on_update',
    'on_key_press', 'on_key_release', 'on_key_hold',
    'actor_on_weapon_fired', 'actor_on_hud_animation_end',
    'on_before_hit', 'on_hit',
    'physic_object_on_hit_callback',
    'npc_on_before_hit', 'monster_on_before_hit',
    'npc_on_hit_callback', 'monster_on_hit_callback',
    'actor_on_feel_touch',
    'actor_on_item_take', 'actor_on_item_drop',
    'actor_on_item_use',
})

# Per-frame callbacks detection
PER_FRAME_CALLBACKS: Set[str] = frozenset({
    'actor_on_update',
    'npc_on_update',
    'monster_on_update',
    'physic_object_on_update',
})

# Bare globals that benefit from caching
CACHEABLE_BARE_GLOBALS: Set[str] = frozenset({
    'pairs', 'ipairs', 'next', 'type', 'tostring', 'tonumber',
    'unpack', 'select', 'rawget', 'rawset',
    'clamp', 'vector', 'v2d', 'color', 'is_empty',
})

# these are less beneficial to cache (error handling, output)
BARE_GLOBALS_UNSAFE_TO_CACHE: Set[str] = frozenset({
    'pcall', 'xpcall', 'error', 'assert', 'print',
})

# Module functions that benefit from caching
CACHEABLE_MODULE_FUNCS: Dict[str, Set[str]] = {
    'math': frozenset({
        'floor', 'ceil', 'abs', 'min', 'max', 'sqrt', 'sin', 'cos', 'tan',
        'random', 'pow', 'log', 'exp', 'atan2', 'atan', 'asin', 'acos',
        'deg', 'rad', 'fmod', 'modf', 'huge', 'log10', 'sinh', 'cosh', 'tanh',
    }),
    'string': frozenset({
        'find', 'sub', 'gsub', 'match', 'gmatch', 'format',
        'lower', 'upper', 'len', 'rep', 'byte', 'char', 'reverse',
    }),
    'table': frozenset({
        'insert', 'remove', 'concat', 'sort', 'getn', 'unpack', 'clear', 'new',
    }),
    'bit': frozenset({
        'band', 'bor', 'bxor', 'bnot', 'lshift', 'rshift', 'arshift', 'rol', 'ror',
    }),
    'db': frozenset({
        'actor', 'storage',
    }),
    'level': frozenset({
        'name', 'get_target_obj', 'object_by_id', 'id', 'vertex_id',
    }),
}

# Debug/logging function patterns
DEBUG_FUNCTIONS: Set[str] = frozenset({
    'print', 'printf', 'printe', 'printd', 'log',
    'log1', 'log2', 'log3',
    'DebugLog', 'debug_log', 'trace', 'dump',
})

# functions that have direct replacement patterns (not cached)
DIRECT_REPLACEMENT_FUNCS: Set[str] = frozenset({
    'table.insert', 'table.getn', 'string.len',
})

# Functions/properties that can return nil - calling methods on these without
# nil checks can cause CTD (crash to desktop)
# Format: full_name -> description of when it returns nil
NIL_RETURNING_FUNCTIONS: Dict[str, str] = {
    # level functions
    'level.object_by_id': 'object is offline or does not exist',
    'level.get_target_obj': 'nothing under crosshair',
    'level.get_target_element': 'nothing under crosshair',
    'level.get_target_pos': 'nothing under crosshair',
    'level.vertex_position': 'invalid vertex ID',
    'level.get_view_entity': 'no view entity set',

    # alife functions
    'alife': 'called from main menu or during loading',
    'alife().object': 'object does not exist in simulation',
    'alife():object': 'object does not exist in simulation',
    'alife().story_object': 'no object with that story_id',
    'alife():story_object': 'no object with that story_id',
    'alife().actor': 'actor not spawned yet',
    'alife():actor': 'actor not spawned yet',

    # game object methods that can return nil
    ':parent': 'object has no parent (not in inventory)',
    ':best_enemy': 'no enemy detected',
    ':best_item': 'no item of interest found',
    ':best_danger': 'no danger detected',
    ':best_weapon': 'no weapon available',
    ':best_cover': 'no cover available',
    ':active_item': 'no weapon/item currently equipped',
    ':active_detector': 'no detector currently active',
    ':object': 'item not in inventory or index out of bounds',
    ':get_enemy': 'no current enemy target',
    ':get_corpse': 'no corpse being investigated',
    ':get_current_outfit': 'no outfit equipped',
    ':item_in_slot': 'slot is empty',
    ':get_helicopter': 'not a helicopter or no helicopter',
    ':get_car': 'not in a vehicle',
    ':get_campfire': 'not a campfire zone',
    ':get_artefact': 'not an artefact',
    ':get_physics_shell': 'object has no physics shell',
    ':spawn_ini': 'no spawn ini defined',
    ':motivation_action_manager': 'not an NPC with action manager',
    ':get_current_holder': 'not in a vehicle/turret',
    ':get_old_holder': 'was not in a vehicle/turret',
    ':memory_position': 'object never seen',
    ':get_dest_enemy': 'no destination enemy',
    ':bone_id': 'bone name does not exist',
    ':bone_position': 'bone does not exist',
    ':bone_direction': 'bone does not exist',

    # common patterns
    'db.actor': 'called from main menu or during loading',
    'db.storage': 'storage not initialized',  # db.storage[id] can be nil

    # story object helpers
    'get_story_object': 'no object with that story_id',
    'get_object_by_name': 'object not found or offline',
}

# Method patterns that indicate the variable is being nil-checked
# These patterns mean the variable is safe to use after the check
NIL_CHECK_PATTERNS: Set[str] = {
    'if {var} then',
    'if {var} and',
    'if not {var} then return',
    'if not {var} then return end',
    'if {var} == nil then return',
    'if {var} == nil then return end',
    'if {var} ~= nil then',
    '{var} and {var}:',
    '{var} and {var}.',
}

# Callback parameters that are guaranteed non-nil by the engine
# Format: (callback_name, param_index) - 0-indexed
SAFE_CALLBACK_PARAMS: Dict[str, Set[int]] = {
    'actor_on_item_take': {0},      # item
    'actor_on_item_drop': {0},      # item
    'actor_on_item_use': {0},       # item
    'actor_on_trade': {0, 1},       # item, sell_buy
    'npc_on_death_callback': {0, 1}, # npc, killer
    'monster_on_death_callback': {0, 1}, # monster, killer
    'npc_on_hit_callback': {0},     # npc
    'monster_on_hit_callback': {0}, # monster
    'on_before_hit': {0, 1, 2},     # obj, shit, bone_id
    'physic_object_on_hit_callback': {0}, # obj
    'actor_on_before_death': {0, 1}, # who, flags
    'save_state': {0},              # m_data
    'load_state': {0},              # m_data
}

# LuaJIT NYI (Not Yet Implemented) functions that abort JIT compilation
# These should be avoided in hot paths (HOT_CALLBACKS)
LUAJIT_NYI_FUNCS: Set[str] = frozenset({
    'pairs',           # NYI in some versions, prefers next or numeric loop
    'string.format',   # NYI (very common)
    'string.gsub',     # NYI
    'string.match',    # NYI
    'string.gmatch',   # NYI
    'string.dump',     # NYI
    'table.concat',    # NYI
    'table.insert',    # NYI (when used with 3 args)
    'table.remove',    # NYI (when used with 2 args)
    'table.sort',      # NYI
    'unpack',          # NYI
    'pcall',           # NYI
    'xpcall',          # NYI
    'load',            # NYI
    'loadstring',      # NYI
    'next',            # NYI in some contexts
    'math.sinh',       # NYI
    'math.cosh',       # NYI
    'math.tanh',       # NYI
    'math.log10',      # NYI
    'io.open', 'io.read', 'io.write', 'io.close', 'io.flush', 'io.lines',
    'os.execute', 'os.rename', 'os.remove', 'os.getenv', 'os.date', 'os.time', 'os.clock',
})

# Functions that are O(N) and should be avoided in loops
SLOW_LOOP_FUNCS: Set[str] = frozenset({
    'table.insert', # with 3 args (insert at index)
    'table.remove', # with 2 args (remove at index)
})
