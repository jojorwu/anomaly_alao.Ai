"""
Shared data for the Lua analyzer.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Set, Tuple
from pathlib import Path
from luaparser.astnodes import Node


def detect_file_encoding(file_path: Path) -> str:
    """Detect file encoding: UTF-8 BOM -> UTF-8 -> latin-1 fallback."""
    raw = file_path.read_bytes()
    # UTF-8 BOM
    if raw[:3] == b'\xef\xbb\xbf':
        return 'utf-8-sig'
    # try UTF-8
    try:
        raw.decode('utf-8')
        return 'utf-8'
    except UnicodeDecodeError:
        pass
    # fallback to latin-1 (maps bytes 0-255 directly, always succeeds)
    return 'latin-1'


@dataclass
class Finding:
    """Represents a single issue found during analysis."""
    pattern_name: str
    severity: str  # GREEN, YELLOW, RED, DEBUG
    line_num: int
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    source_line: str = ""

    # aliases for compatibility
    @property
    def description(self) -> str:
        return self.message

    @property
    def line_content(self) -> str:
        return self.source_line

@dataclass
class Scope:
    """Represents a variable scope (function, loop, block)."""
    name: str
    start_line: int
    end_line: int = -1
    parent: Optional['Scope'] = None
    scope_type: str = 'block'  # 'function', 'loop', 'block'
    is_hot_callback: bool = False

    # variables declared in this scope
    locals: Set[str] = field(default_factory=set)

    # cached globals in this scope
    cached_globals: Set[str] = field(default_factory=set)

    # function aliases: local_name -> canonical_name (e.g., "tinsert" -> "table.insert")
    func_aliases: Dict[str, str] = field(default_factory=dict)

    def __hash__(self):
        # use object's actual id for hashing
        return id(self)

    def __eq__(self, other):
        if isinstance(other, Scope):
            return self is other
        return False


@dataclass
class CallInfo:
    """Information about a function call."""
    full_name: str          # "table.insert", "db.actor", "pairs"
    module: Optional[str]   # "table", "db", None
    func: str               # "insert", "actor", "pairs"
    args: List[Any]         # AST nodes of arguments
    line: int
    node: Node
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0
    parent_if_node: Optional[Node] = None  # which If statement contains this call
    branch_index: int = -1  # 0=main if, 1=elseif[0], 2=elseif[1], etc., -1=else or not in if


@dataclass
class AssignInfo:
    """Information about an assignment."""
    target: str             # variable name
    value_type: str         # 'call', 'index', 'concat', 'literal', 'other'
    value_repr: str         # string representation
    line: int
    node: Node
    scope: Scope
    is_local: bool = False
    in_loop: bool = False


@dataclass
class ConcatInfo:
    """Information about string concatenation."""
    target: Optional[str]   # variable being assigned to
    left_var: Optional[str]  # left operand if it's a variable
    line: int
    scope: Scope
    in_loop: bool = False
    loop_depth: int = 0
    loop_scope: Optional[Scope] = None  # the innermost loop scope
    right_expr: Optional[str] = None    # string repr of right side of concat


@dataclass
class NilSourceInfo:
    """Information about a variable assigned from a nil-returning function."""
    var_name: str           # variable name
    source_call: str        # the call that might return nil (e.g. "level.object_by_id(id)")
    source_func: str        # just the function name (e.g. "level.object_by_id")
    assign_line: int        # line where assignment happened
    scope: Scope            # scope of the variable
    is_local: bool          # whether it's a local variable
    is_guarded: bool = False  # whether a nil check was found after assignment


@dataclass
class NilAccessInfo:
    """Information about accessing a potentially nil variable."""
    var_name: str           # the variable being accessed
    access_type: str        # 'method' or 'index'
    access_call: str        # full call (e.g. "obj:section()")
    access_line: int
    nil_source: NilSourceInfo  # the nil source info
    is_safe_to_fix: bool = False  # whether this can be auto-fixed


@dataclass
class DeadCodeInfo:
    """Information about dead/unreachable code."""
    dead_type: str          # 'after_return', 'after_break', 'if_false', 'while_false', 'unused_local_var', 'unused_local_func'
    start_line: int
    end_line: int
    scope_name: str
    description: str
    is_safe_to_remove: bool = False  # True only for 100% safe cases
    code_preview: str = ""
    node: Optional[Node] = None


@dataclass
class Assignment:
    """Tracks a single assignment to a variable."""
    line: int
    node: Node
    is_used: bool = False


@dataclass
class LocalVarInfo:
    """Information about a local variable for dead code analysis."""
    name: str
    assign_line: int
    scope: Scope
    is_read: bool = False       # has the variable been read?
    is_function: bool = False   # is it a local function?
    read_lines: List[int] = field(default_factory=list)
    is_loop_var: bool = False   # is it a for loop variable?
    is_param: bool = False      # is it a function parameter?
    assignments: List[Assignment] = field(default_factory=list)


@dataclass
class PerFrameCallbackInfo:
    """Information about a per-frame callback function for performance analysis."""
    name: str
    start_line: int
    end_line: int
    scope: Scope
    # Collected during analysis pass
    expensive_calls: List[CallInfo] = field(default_factory=list)
    loop_count: int = 0
    uncached_globals: List[str] = field(default_factory=list)


@dataclass
class DistanceComparisonInfo:
    """Information about distance_to() used in comparison (can be optimized to distance_to_sqr())."""
    line: int
    source_obj: str          # the object calling distance_to (e.g., "pos", "actor:position()")
    target_obj: str          # the argument to distance_to (e.g., "target_pos")
    comparison_op: str       # '<', '<=', '>', '>='
    threshold_value: float   # the numeric threshold (e.g., 10)
    threshold_node: Node     # the AST node for the threshold (for replacement)
    full_node: Node          # the full comparison node
    invoke_node: Node        # the distance_to invoke node


@dataclass
class VectorAllocationInfo:
    """Information about vector() allocation in a loop."""
    line: int
    call_node: Node
    loop_depth: int
    scope: Scope
    in_per_frame_callback: bool = False
