"""
AST-based Lua code analyzer implemented with https://pypi.org/project/luaparser/
"""

from luaparser import ast
from luaparser.astnodes import (
    Node, Chunk, Block,
    Function, LocalFunction, Method,
    Assign, LocalAssign,
    While, Repeat, Fornum, Forin,
    If, ElseIf,
    Call, Invoke,
    Index, Name, String, Number, Nil, TrueExpr, FalseExpr,
    Table, Field,
    Concat, AddOp, SubOp, MultOp, FloatDivOp, ModOp, ExpoOp,
    Return, Break,
    UMinusOp, UBNotOp, ULNotOp, ULengthOP,
    AndLoOp, OrLoOp,
    LessThanOp, GreaterThanOp, LessOrEqThanOp, GreaterOrEqThanOp, EqToOp, NotEqToOp,
    SemiColon, Comment,
)
from typing import List, Dict, Set, Optional, Tuple, Any
from pathlib import Path
from collections import defaultdict
import sys
import io
import re
import math as py_math

from models import (
    Finding, detect_file_encoding, Scope, CallInfo, AssignInfo,
    ConcatInfo, NilSourceInfo, NilAccessInfo, DeadCodeInfo,
    LocalVarInfo, PerFrameCallbackInfo, DistanceComparisonInfo,
    VectorAllocationInfo, Assignment
)
from utils import node_to_string, iter_children
from constants import (
    HOT_CALLBACKS, PER_FRAME_CALLBACKS, CACHEABLE_BARE_GLOBALS,
    BARE_GLOBALS_UNSAFE_TO_CACHE, CACHEABLE_MODULE_FUNCS,
    DEBUG_FUNCTIONS, DIRECT_REPLACEMENT_FUNCS, NIL_RETURNING_FUNCTIONS,
    NIL_CHECK_PATTERNS, SAFE_CALLBACK_PARAMS, LUAJIT_NYI_FUNCS
)

class ASTAnalyzer:
    """
    AST-based Lua code analyzer.

    This class parses Lua source code into an Abstract Syntax Tree (AST) and
    identifies various patterns that can be optimized or that indicate potential
    bugs (like nil access or dead code). It uses the visitor pattern to traverse
    the AST and collect information about function calls, assignments, and variable scopes.
    """

    def __init__(self, cache_threshold: int = 4, experimental: bool = False):
        self.cache_threshold = cache_threshold
        self.experimental = experimental
        self.reset()

    def reset(self):
        """Reset analyzer state for reuse."""
        self.findings: List[Finding] = []
        self.parent_map: Dict[int, Node] = {}
        self.comparisons: List[Tuple[Node, int]] = []
        self.exposures: List[Tuple[Node, int]] = []
        self.index_accesses: List[Tuple[Node, int, Scope, bool]] = []
        self.active_assignments: Dict[Tuple[int, str], Any] = {}
        self.scopes: List[Scope] = []
        self.current_scope: Optional[Scope] = None
        self.global_scope: Optional[Scope] = None
        self.calls: List[CallInfo] = []
        self.assigns: List[AssignInfo] = []
        self.concats: List[ConcatInfo] = []
        self.global_writes: List[Tuple[str, int]] = []
        
        self.nil_sources: Dict[Tuple[int, str], NilSourceInfo] = {}
        self.nil_accesses: List[NilAccessInfo] = []
        self.nil_guards: Set[Tuple[str, int]] = set()
        
        self.dead_code: List[DeadCodeInfo] = []
        self.local_vars: Dict[Tuple[int, str], LocalVarInfo] = {}
        self.local_funcs: Dict[Tuple[int, str], LocalVarInfo] = {}
        self.callback_registrations: Set[str] = set()
        self.per_frame_callbacks: List[PerFrameCallbackInfo] = []
        self.distance_comparisons: List[DistanceComparisonInfo] = []
        self.vector_allocations: List[VectorAllocationInfo] = []
        self.assignment_target_ids: Set[int] = set()

        self.source_lines: List[str] = []
        self.source: str = ""
        self.file_path: Optional[Path] = None

        self.loop_depth: int = 0
        self.function_depth: int = 0
        
        self.current_if_chain: Optional[Node] = None
        self.current_branch_index: int = -1

    def analyze_file(self, file_path: Path) -> List[Finding]:
        """Analyze a Lua file and return findings."""
        self.reset()
        self.file_path = file_path
        self._ast_tree = None

        try:
            encoding = detect_file_encoding(file_path)
            self.source = file_path.read_text(encoding=encoding)
            self._file_encoding = encoding
        except Exception:
            return []

        self.source_lines = self.source.splitlines()

        try:
            # suppress ANTLR lexer error output during parse
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                tree = ast.parse(self.source)
            finally:
                sys.stderr = old_stderr
        except Exception:
            # parse error, skip
            return []

        # store AST tree for dead code analysis
        self._ast_tree = tree
        from utils import get_parent_map
        self.parent_map = get_parent_map(tree)

        # create global scope
        self.global_scope = Scope(
            name='<global>',
            start_line=1,
            end_line=len(self.source_lines),
            scope_type='global',
        )
        self.current_scope = self.global_scope
        self.scopes.append(self.global_scope)

        # walk AST
        self._visit(tree)

        # analyze collected data
        self._analyze_patterns()

        return self.findings

    def _get_line(self, node: Node) -> int:
        """Extract line number from node."""
        ft = getattr(node, 'first_token', None)
        if ft:
            s = str(ft)
            if ',' in s:
                parts = s.rsplit(',', 1)
                if len(parts) == 2:
                    line_col = parts[1].rstrip(']')
                    if ':' in line_col:
                        try:
                            return int(line_col.split(':')[0])
                        except ValueError:
                            pass
        return 0

    def _get_node_source(self, node: Node) -> str:
        """Get source text for a node (approximate)."""
        line = self._get_line(node)
        if 0 < line <= len(self.source_lines):
            return self.source_lines[line - 1].strip()
        return ""


    def _get_const_val(self, node) -> Any:
        """Recursive helper to get constant value of a node if possible."""
        if isinstance(node, Number):
            return node.n
        if isinstance(node, String):
            s = node.s
            if isinstance(s, bytes):
                return s.decode('utf-8', errors='replace')
            return s
        if isinstance(node, TrueExpr):
            return True
        if isinstance(node, FalseExpr):
            return False
        if isinstance(node, Nil):
            return None

        if isinstance(node, UMinusOp):
            val = self._get_const_val(node.operand)
            return -val if isinstance(val, (int, float)) else None

        if isinstance(node, Index):
            full_name = node_to_string(node)
            if full_name == 'math.pi':
                return py_math.pi
            if full_name == 'math.huge':
                return float('inf')

        if isinstance(node, Call):
            _, _, full_name = self._get_call_name(node)
            args = [self._get_const_val(a) for a in node.args]
            if any(a is None for a in args) and len(node.args) > 0:
                return None

            try:
                def to_int32(n):
                    n = int(n) & 0xFFFFFFFF
                    return n if n < 0x80000000 else n - 0x100000000

                if full_name == 'math.abs': return abs(args[0])
                if full_name == 'math.floor': return py_math.floor(args[0])
                if full_name == 'math.ceil': return py_math.ceil(args[0])
                if full_name == 'math.sqrt' and args[0] >= 0: return py_math.sqrt(args[0])
                if full_name == 'math.deg': return py_math.degrees(args[0])
                if full_name == 'math.rad': return py_math.radians(args[0])
                if full_name == 'math.fmod': return py_math.fmod(args[0], args[1])
                if full_name == 'math.min': return min(args)
                if full_name == 'math.max': return max(args)

                if full_name.startswith('bit.'):
                    if full_name == 'bit.bnot': return to_int32(~int(args[0]))
                    if full_name == 'bit.band':
                        res = -1
                        for a in args: res &= int(a)
                        return to_int32(res)
                    if full_name == 'bit.bor':
                        res = 0
                        for a in args: res |= int(a)
                        return to_int32(res)
                    if full_name == 'bit.bxor':
                        res = 0
                        for a in args: res ^= int(a)
                        return to_int32(res)
                    if full_name == 'bit.lshift': return to_int32(int(args[0]) << (int(args[1]) % 32))
                    if full_name == 'bit.rshift':
                        # Logical shift
                        return to_int32((int(args[0]) & 0xFFFFFFFF) >> (int(args[1]) % 32))
                    if full_name == 'bit.arshift':
                        a = int(args[0]) & 0xFFFFFFFF
                        shift = int(args[1]) % 32
                        if shift == 0: return to_int32(a)
                        if a & 0x80000000:
                            # Sign extend
                            return to_int32((a >> shift) | (0xFFFFFFFF << (32 - shift)))
                        return to_int32(a >> shift)
                    if full_name == 'bit.rol':
                        a = int(args[0]) & 0xFFFFFFFF
                        shift = int(args[1]) % 32
                        if shift == 0: return to_int32(a)
                        return to_int32((a << shift) | (a >> (32 - shift)))
                    if full_name == 'bit.ror':
                        a = int(args[0]) & 0xFFFFFFFF
                        shift = int(args[1]) % 32
                        if shift == 0: return to_int32(a)
                        return to_int32((a >> shift) | (a << (32 - shift)))

                if full_name == 'string.len':
                    s = args[0]
                    if isinstance(s, str): return len(s.encode('utf-8'))
                    if isinstance(s, bytes): return len(s)
                if full_name == 'string.upper': return args[0].upper() if isinstance(args[0], (str, bytes)) else None
                if full_name == 'string.lower': return args[0].lower() if isinstance(args[0], (str, bytes)) else None
                if full_name == 'string.char':
                    # Lua string.char returns bytes
                    return bytes([int(a) for a in args]).decode('latin-1')
                if full_name == 'string.byte':
                    s = args[0]
                    idx = int(args[1]) if len(args) > 1 else 1
                    if isinstance(s, str): s = s.encode('utf-8')
                    return s[idx-1] if 0 < idx <= len(s) else None

                if full_name == 'tonumber':
                    try: return float(args[0])
                    except: return None
                if full_name == 'tostring':
                    if isinstance(args[0], bool): return str(args[0]).lower()
                    if args[0] is None: return "nil"
                    return str(args[0])
            except Exception:
                return None

        if isinstance(node, (AddOp, SubOp, MultOp, FloatDivOp, ModOp, ExpoOp)):
            l = self._get_const_val(node.left)
            r = self._get_const_val(node.right)
            if l is None or r is None: return None
            try:
                if isinstance(node, AddOp): return l + r
                if isinstance(node, SubOp): return l - r
                if isinstance(node, MultOp): return l * r
                if isinstance(node, FloatDivOp): return l / r if r != 0 else None
                if isinstance(node, ModOp): return l % r if r != 0 else None
                if isinstance(node, ExpoOp): return l ** r
            except Exception:
                return None

        if isinstance(node, ULengthOP):
            val = self._get_const_val(node.operand)
            if isinstance(val, (str, bytes)):
                return len(val) if isinstance(val, bytes) else len(val.encode('utf-8'))
            return None

        return None

    def _get_call_name(self, node: Call) -> Tuple[Optional[str], str, str]:
        """Get module, function, and full name from a Call node."""
        func = node.func

        if isinstance(func, Name):
            # bare function: pairs(), time_global()
            return None, func.id, func.id
        elif isinstance(func, Index):
            # module.func: table.insert(), db.actor
            if isinstance(func.value, Name) and isinstance(func.idx, Name):
                module = func.value.id
                fn = func.idx.id
                return module, fn, f"{module}.{fn}"

        return None, "", ""

    def _enter_scope(self, name: str, line: int, scope_type: str = 'block', is_hot: bool = False):
        """Enter a new scope."""
        new_scope = Scope(
            name=name,
            start_line=line,
            parent=self.current_scope,
            scope_type=scope_type,
            is_hot_callback=is_hot or (self.current_scope and self.current_scope.is_hot_callback),
        )

        # inherit cached globals from parent
        if self.current_scope:
            new_scope.cached_globals = set(self.current_scope.cached_globals)
            # inherit function aliases from parent scope
            new_scope.func_aliases = dict(self.current_scope.func_aliases)

        self.scopes.append(new_scope)
        self.current_scope = new_scope
        return new_scope

    def _exit_scope(self, end_line: int):
        """Exit current scope."""
        if self.current_scope:
            self.current_scope.end_line = end_line
            self.current_scope = self.current_scope.parent

    def _is_cached(self, name: str) -> bool:
        """Check if a global is cached in current scope chain."""
        scope = self.current_scope
        while scope:
            if name in scope.cached_globals or name in scope.locals:
                return True
            scope = scope.parent
        return False

    def _resolve_alias(self, name: str) -> Optional[str]:
        """
        Resolve a function alias to its canonical name.
        
        If 'name' is an alias for a stdlib function (e.g., 'tinsert' -> 'table.insert'),
        returns the canonical name. Otherwise returns None.
        """
        scope = self.current_scope
        while scope:
            if name in scope.func_aliases:
                return scope.func_aliases[name]
            scope = scope.parent
        return None

    def _find_local_key(self, name: str) -> Optional[Tuple[int, str]]:
        """Find the key (scope_id, name) for a local variable in scope chain."""
        scope = self.current_scope
        while scope:
            key = (id(scope), name)
            if key in self.local_vars:
                return key
            scope = scope.parent
        return None

    def _register_local(self, var_name: str, line: int, is_param: bool = False, is_loop_var: bool = False):
        """Centralized helper to register a local variable or parameter."""
        if not self.current_scope:
            return

        self.current_scope.locals.add(var_name)

        # Track for diagnostics (shadowing and unused detection)
        if not var_name.startswith('_'):
            key = (id(self.current_scope), var_name)
            info = LocalVarInfo(
                name=var_name,
                assign_line=line,
                scope=self.current_scope,
                is_read=False,
                is_function=False,
                is_loop_var=is_loop_var,
                is_param=is_param,
                assignments=[]
            )
            # Initial assignment (declaration)
            assign = Assignment(line=line, node=None)
            info.assignments.append(assign)
            self.active_assignments[key] = assign
            self.local_vars[key] = info

    def _visit(self, node: Node):
        """Visit a node and dispatch to specific handler."""
        if node is None:
            return

        handler = getattr(self, f'_visit_{type(node).__name__}', None)
        if handler:
            handler(node)
        else:
            self._visit_children(node)

    def _visit_children(self, node: Node):
        """Visit all children of a node using optimized traversal."""
        for child in iter_children(node):
            if isinstance(child, Node):
                self._visit(child)

    def _visit_Chunk(self, node: Chunk):
        self._visit(node.body)

    def _visit_Block(self, node: Block):
        for stmt in node.body:
            self._visit(stmt)

    def _visit_Function(self, node: Function):
        """Handle global function definition."""
        line = self._get_line(node)
        func_name = node_to_string(node.name) if node.name else '<anon>'

        is_hot = func_name in HOT_CALLBACKS
        is_per_frame = func_name in PER_FRAME_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)
        
        # Track per-frame callback for performance analysis
        if is_per_frame:
            self.per_frame_callbacks.append(PerFrameCallbackInfo(
                name=func_name,
                start_line=line,
                end_line=-1,  # Will be set on scope exit
                scope=self.current_scope,
            ))

        # register parameters as locals
        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self._register_local(arg.id, line, is_param=True)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        
        # Update end_line for per-frame callback
        if is_per_frame and self.per_frame_callbacks:
            self.per_frame_callbacks[-1].end_line = end_line
        
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_LocalFunction(self, node: LocalFunction):
        """Handle local function definition."""
        line = self._get_line(node)
        func_name = node.name.id if isinstance(node.name, Name) else '<anon>'

        # register function name in parent scope (before entering function scope)
        if self.current_scope:
            self.current_scope.locals.add(func_name)
            
            # Mark function name as assignment target
            if isinstance(node.name, Name):
                self.assignment_target_ids.add(id(node.name))
                
                # Track for unused function detection (skip _ prefixed)
                if not func_name.startswith('_'):
                    key = (id(self.current_scope), func_name)
                    self.local_funcs[key] = LocalVarInfo(
                        name=func_name,
                        assign_line=line,
                        scope=self.current_scope,
                        is_read=False,
                        is_function=True,
                    )

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)

        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self._register_local(arg.id, line, is_param=True)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _visit_Method(self, node: Method):
        """Handle method definition."""
        line = self._get_line(node)

        # get method name
        if isinstance(node.name, Index):
            func_name = node_to_string(node.name)
        else:
            func_name = node_to_string(node.name) if node.name else '<method>'

        is_hot = func_name in HOT_CALLBACKS

        self.function_depth += 1
        self._enter_scope(func_name, line, 'function', is_hot)

        # 'self' is implicit first param
        self._register_local('self', line, is_param=True)

        if hasattr(node, 'args') and node.args:
            for arg in node.args:
                if isinstance(arg, Name):
                    self._register_local(arg.id, line, is_param=True)

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.function_depth -= 1

    def _get_end_line(self, node: Node) -> int:
        """Try to get end line of a node."""
        lt = getattr(node, 'last_token', None)
        if lt:
            s = str(lt)
            if ',' in s:
                parts = s.rsplit(',', 1)
                if len(parts) == 2:
                    line_col = parts[1].rstrip(']')
                    if ':' in line_col:
                        try:
                            return int(line_col.split(':')[0])
                        except ValueError:
                            pass
        return self._get_line(node)


    def _visit_Forin(self, node: Forin):
        """Handle for-in loop."""
        line = self._get_line(node)

        # visit iterator expression first (outside loop scope)
        for iter_expr in node.iter:
            self._visit(iter_expr)

        self.loop_depth += 1
        self._enter_scope('<forin>', line, 'loop')

        # loop variables are local to loop
        for target in node.targets:
            if isinstance(target, Name):
                var_name = target.id
                self._register_local(var_name, line, is_loop_var=True)
                
                # Mark as assignment target (not a read)
                self.assignment_target_ids.add(id(target))

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_Fornum(self, node: Fornum):
        """Handle numeric for loop."""
        line = self._get_line(node)

        # visit range expressions first
        self._visit(node.start)
        self._visit(node.stop)
        if node.step:
            self._visit(node.step)

        self.loop_depth += 1
        self._enter_scope('<fornum>', line, 'loop')

        if isinstance(node.target, Name):
            var_name = node.target.id
            self._register_local(var_name, line, is_loop_var=True)
            
            # Mark as assignment target (not a read)
            self.assignment_target_ids.add(id(node.target))

        self._visit(node.body)

        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_While(self, node: While):
        """Handle while loop."""
        line = self._get_line(node)

        self._visit(node.test)

        self.loop_depth += 1
        self._enter_scope('<while>', line, 'loop')
        self._visit(node.body)
        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_Repeat(self, node: Repeat):
        """Handle repeat-until loop."""
        line = self._get_line(node)

        self.loop_depth += 1
        self._enter_scope('<repeat>', line, 'loop')
        self._visit(node.body)
        self._visit(node.test)
        end_line = self._get_end_line(node)
        self._exit_scope(end_line)
        self.loop_depth -= 1

    def _visit_If(self, node: If):
        """Handle if statement."""
        # save previous if-chain context
        prev_if_chain = self.current_if_chain
        prev_branch_index = self.current_branch_index
        
        # set this as current if-chain
        self.current_if_chain = node
        
        # visit test condition
        self._visit(node.test)
        
        # visit main if body (branch 0)
        self.current_branch_index = 0
        self._visit(node.body)
        
        # visit elseif/else chain
        if node.orelse:
            self._visit_orelse(node.orelse, 1)
        
        # restore previous context
        self.current_if_chain = prev_if_chain
        self.current_branch_index = prev_branch_index
    
    def _visit_orelse(self, node, branch_idx):
        """Helper to visit elseif/else with branch tracking."""
        if isinstance(node, ElseIf):
            # elseif branch
            self.current_branch_index = branch_idx
            self._visit(node.test)
            self._visit(node.body)
            if node.orelse:
                self._visit_orelse(node.orelse, branch_idx + 1)
        elif isinstance(node, Block):
            # else block
            self.current_branch_index = -1  # -1 for else
            self._visit(node)
        else:
            # fallback
            self.current_branch_index = -1
            self._visit(node)

    def _visit_ElseIf(self, node: ElseIf):
        """Handle elseif clause - this is called from _visit_orelse."""
        # already handled by _visit_orelse
        pass

    def _visit_LocalAssign(self, node: LocalAssign):
        """Handle local assignment."""
        line = self._get_line(node)

        # register targets as locals and track for unused variable detection
        for target in node.targets:
            if isinstance(target, Name):
                var_name = target.id
                self._register_local(var_name, line)
                
                # Mark this Name node as an assignment target (not a read)
                self.assignment_target_ids.add(id(target))

        # check for caching pattern: local xyz = module.func
        is_new_alias = False
        if len(node.targets) == 1 and len(node.values) == 1:
            target = node.targets[0]
            value = node.values[0]

            if isinstance(target, Name):
                target_name = target.id

                # check if caching a module.func
                if isinstance(value, Index):
                    if isinstance(value.value, Name) and isinstance(value.idx, Name):
                        module = value.value.id
                        func = value.idx.id
                        full_name = f"{module}.{func}"

                        if module in CACHEABLE_MODULE_FUNCS:
                            self.current_scope.cached_globals.add(full_name)
                            # Record alias mapping: target_name -> canonical function name
                            self.current_scope.func_aliases[target_name] = full_name
                            is_new_alias = True

                # check if caching a bare global
                elif isinstance(value, Name):
                    if value.id in CACHEABLE_BARE_GLOBALS:
                        self.current_scope.cached_globals.add(value.id)
                        # Record alias mapping for bare globals too
                        self.current_scope.func_aliases[target_name] = value.id
                        is_new_alias = True
                
                # If NOT creating a new alias, invalidate any existing alias with this name
                # (e.g., local f = table.insert; local f = other_func)
                if not is_new_alias:
                    if target_name in self.current_scope.func_aliases:
                        del self.current_scope.func_aliases[target_name]

                # record assignment info
                self._record_assignment(target_name, value, line, is_local=True)

        # visit values
        for value in node.values:
            self._visit(value)

    def _visit_Assign(self, node: Assign):
        """Handle assignment."""
        line = self._get_line(node)

        # Mark Name targets as assignment targets (not reads)
        for target in node.targets:
            if isinstance(target, Name):
                self.assignment_target_ids.add(id(target))
                
                # Invalidate alias if this variable was an alias
                # (reassigning breaks the alias relationship)
                target_name = target.id
                self._invalidate_alias(target_name)

        # check for global writes
        for target in node.targets:
            if isinstance(target, Name):
                target_name = target.id
                # it's a global write if not in any scope's locals
                if not self._is_in_locals(target_name):
                    self.global_writes.append((target_name, line))

                if len(node.values) == 1:
                    self._record_assignment(target_name, node.values[0], line, is_local=False)

        # visit targets (for calls inside index expressions like db.storage[npc:id()])
        for target in node.targets:
            self._visit(target)

        # visit values
        for value in node.values:
            self._visit(value)
    
    def _invalidate_alias(self, var_name: str):
        """Remove alias mapping when a variable is reassigned."""
        scope = self.current_scope
        while scope:
            if var_name in scope.func_aliases:
                del scope.func_aliases[var_name]
                return  # Only remove from innermost scope where it exists
            scope = scope.parent

    def _infer_type(self, node: Node) -> Optional[str]:
        """Infer the Lua type of an expression node."""
        if isinstance(node, (Number, AddOp, SubOp, MultOp, FloatDivOp, ModOp, ExpoOp, UMinusOp)):
            return 'number'
        if isinstance(node, (String, Concat)):
            return 'string'
        if isinstance(node, (TrueExpr, FalseExpr, AndLoOp, OrLoOp, ULNotOp, LessThanOp, GreaterThanOp, LessOrEqThanOp, GreaterOrEqThanOp, EqToOp, NotEqToOp)):
            return 'boolean'
        if isinstance(node, Table):
            return 'table'
        if isinstance(node, (Function, LocalFunction)):
            return 'function'
        if isinstance(node, Nil):
            return 'nil'
        if isinstance(node, ULengthOP):
            return 'number'
        if isinstance(node, Call):
            _, _, full_name = self._get_call_name(node)
            if full_name in (
                'math.floor', 'math.ceil', 'math.abs', 'math.sqrt', 'math.sin', 'math.cos',
                'math.tan', 'math.asin', 'math.acos', 'math.atan', 'math.atan2', 'math.exp',
                'math.log', 'math.log10', 'math.random', 'math.pow', 'math.min', 'math.max',
                'math.deg', 'math.rad', 'math.fmod', 'math.modf', 'math.sinh', 'math.cosh',
                'math.tanh', 'tonumber', 'string.len', 'string.byte', 'bit.band', 'bit.bor',
                'bit.bxor', 'bit.bnot', 'bit.lshift', 'bit.rshift', 'bit.arshift', 'bit.rol', 'bit.ror'
            ):
                return 'number'
            if full_name in ('tostring', 'string.format', 'string.sub', 'string.gsub', 'string.lower', 'string.upper', 'string.char'):
                return 'string'
        if isinstance(node, Name):
            key = self._find_local_key(node.id)
            if key and key in self.active_assignments:
                return self.active_assignments[key].inferred_type
        return None

    def _is_in_locals(self, name: str) -> bool:
        """Check if name is in any scope's locals."""
        scope = self.current_scope
        while scope:
            if name in scope.locals:
                return True
            scope = scope.parent
        return False

    def _record_assignment(self, target: str, value: Node, line: int, is_local: bool):
        """Record an assignment for analysis."""
        # Track dead assignments
        key = self._find_local_key(target)
        if key:
            # If there was a previous assignment that was never used
            if key in self.active_assignments:
                prev_assign = self.active_assignments[key]
                if not prev_assign.is_used and not self.loop_depth > 0:
                    # Ignore assignments in loops for now as they are complex
                    # (might be used in next iteration)
                    pass

            new_assign = Assignment(line=line, node=value, inferred_type=self._infer_type(value))
            self.local_vars[key].assignments.append(new_assign)
            self.active_assignments[key] = new_assign

        if isinstance(value, Call):
            value_type = 'call'
            value_repr = node_to_string(value)
        elif isinstance(value, Index):
            value_type = 'index'
            value_repr = node_to_string(value)
        elif isinstance(value, Concat):
            value_type = 'concat'
            value_repr = node_to_string(value)

            # record concat info
            left_var = None
            if isinstance(value.left, Name):
                left_var = value.left.id
            
            # get the right side expression
            right_expr = node_to_string(value.right)
            
            # find innermost loop scope
            loop_scope = None
            if self.loop_depth > 0:
                # walk up scopes to find loop
                s = self.current_scope
                while s:
                    if s.scope_type == 'loop':
                        loop_scope = s
                        break
                    s = s.parent

            self.concats.append(ConcatInfo(
                target=target,
                left_var=left_var,
                line=line,
                scope=self.current_scope,
                in_loop=self.loop_depth > 0,
                loop_depth=self.loop_depth,
                loop_scope=loop_scope,
                right_expr=right_expr,
            ))
        elif isinstance(value, (Number, String, TrueExpr, FalseExpr, Nil)):
            value_type = 'literal'
            value_repr = node_to_string(value)
        else:
            value_type = 'other'
            value_repr = node_to_string(value)

        self.assigns.append(AssignInfo(
            target=target,
            value_type=value_type,
            value_repr=value_repr,
            line=line,
            node=value,
            scope=self.current_scope,
            is_local=is_local,
            in_loop=self.loop_depth > 0,
        ))
        
        # Track nil-returning function assignments
        self._track_nil_source(target, value, value_repr, line, is_local)

    def _track_nil_source(self, target: str, value: Node, value_repr: str, line: int, is_local: bool):
        """Track if a variable is assigned from a nil-returning function."""
        source_func = None
        
        # Check if it's a call to a nil-returning function
        if isinstance(value, Call):
            # get the function name
            _, _, full_name = self._get_call_name(value)
            if full_name and full_name in NIL_RETURNING_FUNCTIONS:
                source_func = full_name
        
        # Check for method call (Invoke) - e.g., obj:parent()
        elif isinstance(value, Invoke):
            method_name = value.func.id if isinstance(value.func, Name) else ''
            method_pattern = f':{method_name}'
            if method_pattern in NIL_RETURNING_FUNCTIONS:
                source_func = method_pattern
        
        # Check for index access - e.g., db.actor, alife():object(id)
        elif isinstance(value, Index):
            full_name = node_to_string(value)
            # check direct matches like db.actor
            if full_name in NIL_RETURNING_FUNCTIONS:
                source_func = full_name
        
        if source_func:
            key = (id(self.current_scope), target)
            self.nil_sources[key] = NilSourceInfo(
                var_name=target,
                source_call=value_repr,
                source_func=source_func,
                assign_line=line,
                scope=self.current_scope,
                is_local=is_local,
                is_guarded=False,
            )
        else:
            # if variable is reassigned from non-nil source, remove from tracking
            key = (id(self.current_scope), target)
            if key in self.nil_sources:
                del self.nil_sources[key]

    def _check_nil_access(self, source_node: Node, source_str: str, full_call: str, line: int, access_type: str):
        """Check if we're accessing a potentially nil variable."""
        # only check simple variable names for now
        if not isinstance(source_node, Name):
            return
        
        var_name = source_node.id
        
        # check if this variable is from a nil-returning function in an enclosing scope
        nil_source = self._find_nil_source(var_name)
        if not nil_source:
            return
        
        # check if there's a nil guard before this access
        if self._has_nil_guard(var_name, nil_source.assign_line, line):
            nil_source.is_guarded = True
            return
        
        # determine if this is safe to auto-fix
        # Safe if: assignment is on previous line, this is the only usage before any branch
        is_safe = self._is_safe_nil_fix(nil_source, line)
        
        self.nil_accesses.append(NilAccessInfo(
            var_name=var_name,
            access_type=access_type,
            access_call=full_call,
            access_line=line,
            nil_source=nil_source,
            is_safe_to_fix=is_safe,
        ))

    def _find_nil_source(self, var_name: str) -> Optional[NilSourceInfo]:
        """Find nil source for a variable in current or enclosing scopes."""
        # key is (scope_id, var_name)
        scope = self.current_scope
        while scope:
            key = (id(scope), var_name)
            if key in self.nil_sources:
                return self.nil_sources[key]
            scope = scope.parent
        return None

    @staticmethod
    def _strip_line_comments_and_strings(text: str) -> str:
        """Strip comment and string content from a line so regex only matches real code."""
        result = []
        i = 0
        while i < len(text):
            c = text[i]
            # line comment
            if c == '-' and i + 1 < len(text) and text[i + 1] == '-':
                break
            # string literal
            elif c in ('"', "'"):
                quote = c
                result.append(c)
                i += 1
                while i < len(text):
                    sc = text[i]
                    if sc == '\\':
                        result.append(' ')
                        i += 1
                        if i < len(text):
                            result.append(' ')
                            i += 1
                        continue
                    if sc == quote:
                        result.append(sc)
                        i += 1
                        break
                    result.append(' ')
                    i += 1
                continue
            # long string [[...]]
            elif c == '[' and i + 1 < len(text) and text[i + 1] in ('[', '='):
                eq_count = 0
                j = i + 1
                while j < len(text) and text[j] == '=':
                    eq_count += 1
                    j += 1
                if j < len(text) and text[j] == '[':
                    close = ']' + '=' * eq_count + ']'
                    end = text.find(close, j + 1)
                    if end != -1:
                        result.append(' ' * (end + len(close) - i))
                        i = end + len(close)
                        continue
                result.append(c)
                i += 1
                continue
            else:
                result.append(c)
                i += 1
        return ''.join(result)

    def _has_nil_guard(self, var_name: str, assign_line: int, access_line: int) -> bool:
        """Check if there's a nil guard between assignment and access."""
        if assign_line >= access_line:
            return False
        
        var_escaped = re.escape(var_name)
        
        guard_patterns = [
            rf'\bif\s+{var_escaped}\s+then\b',
            rf'\bif\s+{var_escaped}\s+and\b',
            rf'\bif\s+not\s+{var_escaped}\s+then\b',
            rf'\bif\s+{var_escaped}\s*~=\s*nil\b',
            rf'\bif\s+{var_escaped}\s*==\s*nil\s+then\s+return\b',
            rf'\b{var_escaped}\s+and\s+{var_escaped}[:\.]',
            rf'\bif\s*\(\s*{var_escaped}\s*\)\s*then\b',
        ]
        
        combined_pattern = re.compile('|'.join(guard_patterns), re.IGNORECASE)
        
        for line_num in range(assign_line, access_line):
            if line_num <= 0 or line_num > len(self.source_lines):
                continue
            line_text = self.source_lines[line_num - 1]
            
            # only match against actual code, not comments or string contents
            cleaned = self._strip_line_comments_and_strings(line_text)
            if combined_pattern.search(cleaned):
                return True
        
        return False

    def _is_safe_nil_fix(self, nil_source: NilSourceInfo, access_line: int) -> bool:
        """
        Determine if a nil access is safe to auto-fix.
        
        Safe conditions:
        1. Access is on the line immediately after assignment
        2. It's a local variable (not global)
        3. Assignment and access are in the same scope
        4. No complex control flow between them
        5. Access line is NOT a local declaration (would break scope if wrapped)
        6. Access line is NOT a control flow statement (if/for/while - too complex)
        """
        # must be immediately after (next line)
        if access_line != nil_source.assign_line + 1:
            return False
        
        # must be local
        if not nil_source.is_local:
            return False
        
        # must be in same scope
        if self.current_scope != nil_source.scope:
            return False
        
        # check that the line between is not a control flow statement
        if nil_source.assign_line <= 0 or nil_source.assign_line > len(self.source_lines):
            return False
        
        # check access line content
        if access_line > 0 and access_line <= len(self.source_lines):
            access_text = self.source_lines[access_line - 1].strip()
            
            # CRITICAL: access line must NOT be a local declaration
            if access_text.startswith('local '):
                return False
            
            # CRITICAL: access line must NOT be control flow (too complex to wrap)
            control_keywords = ('if ', 'if(', 'for ', 'while ', 'repeat', 'function ', 'function(')
            if any(access_text.startswith(kw) for kw in control_keywords):
                return False
            
        return True

    def _visit_Call(self, node: Call):
        """Handle function call."""
        line = self._get_line(node)
        module, func, full_name = self._get_call_name(node)

        # Check if this is an aliased stdlib call
        # e.g., if 'local tinsert = table.insert' was declared,
        # then 'tinsert(t, v)' should be recognized as 'table.insert(t, v)'
        if full_name and module is None:
            # This is a bare function call - check if it's an alias
            canonical = self._resolve_alias(full_name)
            if canonical:
                full_name = canonical
                # Also update module/func if it's a module.func pattern
                if '.' in canonical:
                    module, func = canonical.split('.', 1)

        if full_name:
            self.calls.append(CallInfo(
                full_name=full_name,
                module=module,
                func=func,
                args=node.args,
                line=line,
                node=node,
                scope=self.current_scope,
                in_loop=self.loop_depth > 0,
                loop_depth=self.loop_depth,
                parent_if_node=self.current_if_chain,
                branch_index=self.current_branch_index,
            ))
            
            # Track RegisterScriptCallback for unused variable/function detection
            if full_name == 'RegisterScriptCallback' and len(node.args) >= 2:
                callback_func = node_to_string(node.args[1])
                if callback_func:
                    self.callback_registrations.add(callback_func)
            
            # Track vector() allocations in loops
            if full_name == 'vector' and self.loop_depth > 0:
                # check if we're inside a per-frame callback
                in_per_frame = False
                scope = self.current_scope
                while scope:
                    if scope.is_hot_callback:
                        in_per_frame = True
                        break
                    scope = scope.parent
                
                self.vector_allocations.append(VectorAllocationInfo(
                    line=line,
                    call_node=node,
                    loop_depth=self.loop_depth,
                    scope=self.current_scope,
                    in_per_frame_callback=in_per_frame,
                ))

        # visit children
        self._visit(node.func)
        for arg in node.args:
            self._visit(arg)

    def _visit_Invoke(self, node: Invoke):
        """Handle method call (obj:method())."""
        line = self._get_line(node)

        # record as call
        source = node_to_string(node.source)
        func = node.func.id if isinstance(node.func, Name) else node_to_string(node.func)
        full_name = f"{source}:{func}"

        self.calls.append(CallInfo(
            full_name=full_name,
            module=source,
            func=func,
            args=node.args,
            line=line,
            node=node,
            scope=self.current_scope,
            in_loop=self.loop_depth > 0,
            loop_depth=self.loop_depth,
            parent_if_node=self.current_if_chain,
            branch_index=self.current_branch_index,
        ))
        
        # Check for potential nil access
        self._check_nil_access(node.source, source, full_name, line, 'method')

        self._visit(node.source)
        for arg in node.args:
            self._visit(arg)

    def _visit_Concat(self, node: Concat):
        """Handle concatenation operator."""
        line = self._get_line(node)

        left_var = None
        if isinstance(node.left, Name):
            left_var = node.left.id

        # only interesting if we're in a loop
        if self.loop_depth > 0:
            self.concats.append(ConcatInfo(
                target=None,  # no assignment context here
                left_var=left_var,
                line=line,
                scope=self.current_scope,
                in_loop=True,
                loop_depth=self.loop_depth,
            ))

        self._visit(node.left)
        self._visit(node.right)

    # visitor pass-through for other nodes
    def _visit_Index(self, node: Index):
        # track index access for optimization
        line = self._get_line(node)
        is_write = False
        # check if this is a write
        parent = self.parent_map.get(id(node))
        if isinstance(parent, (Assign, LocalAssign)):
            if node in parent.targets:
                is_write = True

        self.index_accesses.append((node, line, self.current_scope, is_write))

        self._visit(node.value)
        self._visit(node.idx)

    def _visit_Table(self, node: Table):
        for field in node.fields:
            self._visit(field)

    def _visit_Field(self, node: Field):
        if node.key:
            self._visit(node.key)
        self._visit(node.value)

    def _visit_Return(self, node: Return):
        for val in node.values:
            self._visit(val)

    # binary ops
    def _visit_AddOp(self, node: AddOp): self._visit(node.left); self._visit(node.right)
    def _visit_SubOp(self, node: SubOp): self._visit(node.left); self._visit(node.right)
    def _visit_MultOp(self, node: MultOp): self._visit(node.left); self._visit(node.right)
    def _visit_FloatDivOp(self, node: FloatDivOp): self._visit(node.left); self._visit(node.right)
    def _visit_ModOp(self, node: ModOp): self._visit(node.left); self._visit(node.right)
    def _visit_ExpoOp(self, node: ExpoOp): self._visit(node.left); self._visit(node.right)
    def _visit_AndLoOp(self, node: AndLoOp): self._visit(node.left); self._visit(node.right)
    def _visit_OrLoOp(self, node: OrLoOp): self._visit(node.left); self._visit(node.right)
    def _visit_EqToOp(self, node: EqToOp): self._visit(node.left); self._visit(node.right)
    def _visit_NotEqToOp(self, node: NotEqToOp): self._visit(node.left); self._visit(node.right)
    
    def _visit_LessThanOp(self, node: LessThanOp):
        self._check_distance_comparison(node, '<')
        self._visit(node.left)
        self._visit(node.right)

    def _visit_GreaterThanOp(self, node: GreaterThanOp):
        self._check_distance_comparison(node, '>')
        self._visit(node.left)
        self._visit(node.right)

    def _visit_LessOrEqThanOp(self, node: LessOrEqThanOp):
        self._check_distance_comparison(node, '<=')
        self._visit(node.left)
        self._visit(node.right)

    def _visit_GreaterOrEqThanOp(self, node: GreaterOrEqThanOp):
        self._check_distance_comparison(node, '>=')
        self._visit(node.left)
        self._visit(node.right)
    
    def _check_distance_comparison(self, node, op: str):
        """Check if this comparison involves distance_to() that could use distance_to_sqr()."""
        # Pattern: obj:distance_to(target) < N  or  N > obj:distance_to(target)
        invoke_node = None
        threshold_node = None
        
        # check left side for distance_to invoke
        if isinstance(node.left, Invoke):
            func_name = node.left.func.id if isinstance(node.left.func, Name) else None
            if func_name == 'distance_to':
                invoke_node = node.left
                threshold_node = node.right
        
        # check right side for distance_to invoke (reversed comparison)
        if invoke_node is None and isinstance(node.right, Invoke):
            func_name = node.right.func.id if isinstance(node.right.func, Name) else None
            if func_name == 'distance_to':
                invoke_node = node.right
                threshold_node = node.left
                # Reverse the operator for analysis
                op = {'<': '>', '>': '<', '<=': '>=', '>=': '<='}[op]
        
        if invoke_node is None:
            return
        
        # check if threshold is a numeric literal
        if not isinstance(threshold_node, Number):
            return
        
        threshold_value = threshold_node.n
        
        # extract source and target
        source_obj = node_to_string(invoke_node.source)
        target_obj = node_to_string(invoke_node.args[0]) if invoke_node.args else ""
        
        self.distance_comparisons.append(DistanceComparisonInfo(
            line=self._get_line(node),
            source_obj=source_obj,
            target_obj=target_obj,
            comparison_op=op,
            threshold_value=threshold_value,
            threshold_node=threshold_node,
            full_node=node,
            invoke_node=invoke_node,
        ))

    # unary ops
    def _visit_UMinusOp(self, node: UMinusOp): self._visit(node.operand)
    def _visit_UBNotOp(self, node: UBNotOp): self._visit(node.operand)
    def _visit_ULNotOp(self, node: ULNotOp): self._visit(node.operand)
    def _visit_ULengthOP(self, node: ULengthOP): self._visit(node.operand)

    # terminal nodes - no children
    def _visit_Name(self, node: Name):
        """Handle Name node - track variable reads for unused detection."""
        # Only count as read if NOT an assignment target
        if id(node) not in self.assignment_target_ids:
            var_name = node.id
            # Mark active assignment as used
            key = self._find_local_key(var_name)
            if key and key in self.active_assignments:
                self.active_assignments[key].is_used = True

            # Find which scope this variable belongs to (walk up scope chain)
            scope = self.current_scope
            while scope:
                key = (id(scope), var_name)
                if key in self.local_vars:
                    self.local_vars[key].is_read = True
                    break
                if key in self.local_funcs:
                    self.local_funcs[key].is_read = True
                    break
                # Check if it's in this scope's locals (even if not tracked)
                if var_name in scope.locals:
                    break  # Found the scope, but might not be tracked (e.g., _ prefixed)
                scope = scope.parent
    
    def _visit_Number(self, node): pass
    def _visit_String(self, node): pass
    def _visit_Nil(self, node): pass
    def _visit_TrueExpr(self, node): pass
    def _visit_FalseExpr(self, node): pass
    def _visit_SemiColon(self, node): pass
    def _visit_Comment(self, node): pass
    def _visit_Break(self, node): pass


    # PATTERN ANALYSIS

    def _analyze_patterns(self):
        """Analyze collected data and generate findings."""
        self._analyze_table_insert()
        self._analyze_table_remove()
        self._analyze_deprecated_funcs()
        self._analyze_math_pow()
        self._analyze_math_atan2()
        self._analyze_math_mod()
        self._analyze_math_log()
        self._analyze_math_deg_rad()
        self._analyze_math_random_0_1()
        self._analyze_string_rep()
        self._analyze_uncached_globals()
        self._analyze_repeated_calls_in_scope()
        self._analyze_string_concat_in_loop()
        self._analyze_string_format_in_loop()
        self._analyze_debug_statements()
        self._analyze_global_writes()
        self._analyze_nil_access()
        self._analyze_dead_code()
        self._analyze_per_frame_callbacks()
        self._analyze_distance_to_comparisons()
        self._analyze_vector_allocations_in_loops()
        self._analyze_plain_string_find()
        self._analyze_pairs_on_array()
        self._analyze_ipairs_hot_loop()
        self._analyze_redundant_tostring()
        self._analyze_slow_loop_funcs()
        self._analyze_luajit_nyi()
        self._analyze_redundant_boolean_comp()
        self._analyze_math_random_1()
        self._analyze_redundant_return_bool()
        self._analyze_string_format_to_concat()
        self._analyze_table_concat_literal()
        self._analyze_string_sub_to_byte()
        self._analyze_string_sub_to_byte_simple()
        self._analyze_string_lower_case()
        self._analyze_math_abs_positive()
        self._analyze_constant_folding()
        self._analyze_math_min_max()
        self._analyze_expo_to_mult()
        self._analyze_table_new_in_loop()
        self._analyze_repeated_member_access_in_loop()
        self._analyze_string_match_existence()
        self._analyze_unpack_to_indexing()
        self._analyze_divide_by_constant()
        self._analyze_if_nil_assign()
        self._analyze_redundant_tonumber_tostring()
        self._analyze_string_byte_1()
        self._analyze_pairs_to_next()
        self._analyze_return_ternary()
        self._analyze_algebraic_simplification()
        self._analyze_string_starts_with()
        self._analyze_logical_identity()
        self._analyze_nested_redundant_calls()
        self._analyze_table_literal_indices()
        self._analyze_bit_identity()

    def _analyze_bit_identity(self):
        """Find redundant bitwise operations like bit.band(x, 0)."""
        for call in self.calls:
            if call.full_name in ('bit.band', 'bit.bor', 'bit.bxor', 'bit.lshift', 'bit.rshift', 'bit.arshift') and len(call.args) == 2:
                arg1 = call.args[0]
                arg2 = call.args[1]
                val1 = self._get_const_val(arg1)
                val2 = self._get_const_val(arg2)

                target = None
                replacement = None
                reason = None

                s1 = node_to_string(arg1)
                s2 = node_to_string(arg2)
                is_same = (s1 == s2) and self._is_simple_expr(arg1)

                if call.full_name == 'bit.band':
                    if is_same: target = arg1; reason = "bit.band(x, x)"
                    elif val2 == 0: replacement = "0"; reason = "bit.band(x, 0)"
                    elif val1 == 0: replacement = "0"; reason = "bit.band(0, x)"
                    elif val2 == 0xFFFFFFFF: target = arg1; reason = "bit.band(x, -1)"
                    elif val1 == 0xFFFFFFFF: target = arg2; reason = "bit.band(-1, x)"
                elif call.full_name == 'bit.bor':
                    if is_same: target = arg1; reason = "bit.bor(x, x)"
                    elif val2 == 0: target = arg1; reason = "bit.bor(x, 0)"
                    elif val1 == 0: target = arg2; reason = "bit.bor(0, x)"
                    elif val2 == 0xFFFFFFFF: replacement = "-1"; reason = "bit.bor(x, -1)"
                    elif val1 == 0xFFFFFFFF: replacement = "-1"; reason = "bit.bor(-1, x)"
                elif call.full_name == 'bit.bxor':
                    if is_same: replacement = "0"; reason = "bit.bxor(x, x)"
                    elif val2 == 0: target = arg1; reason = "bit.bxor(x, 0)"
                    elif val1 == 0: target = arg2; reason = "bit.bxor(0, x)"
                elif call.full_name in ('bit.lshift', 'bit.rshift', 'bit.arshift'):
                    if val2 == 0: target = arg1; reason = f"{call.full_name}(x, 0)"

                if target or replacement:
                    if target:
                        replacement = node_to_string(target)

                    self.findings.append(Finding(
                        pattern_name='bitwise_identity',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'Bitwise identity simplification: {reason} -> {replacement}',
                        details={
                            'node': call.node,
                            'replacement': replacement
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_table_literal_indices(self):
        """Find table literals with explicit numeric indices { [1] = a } -> { a }."""
        if not self._ast_tree: return

        for node in ast.walk(self._ast_tree):
            if isinstance(node, Table) and node.fields:
                is_sequential = True
                curr_idx = 1
                fixable_fields = []

                for field in node.fields:
                    if field.key and isinstance(field.key, Number) and field.key.n == curr_idx:
                        fixable_fields.append(field)
                        curr_idx += 1
                    elif not field.key:
                        # Sequential part
                        curr_idx += 1
                    else:
                        is_sequential = False
                        break

                if is_sequential and fixable_fields:
                    # build replacement
                    field_strs = []
                    for field in node.fields:
                        field_strs.append(node_to_string(field.value))

                    replacement = "{ " + ", ".join(field_strs) + " }"

                    self.findings.append(Finding(
                        pattern_name='table_literal_indices',
                        severity='GREEN',
                        line_num=self._get_line(node),
                        message=f'Simplify table literal: remove explicit numeric indices',
                        details={
                            'node': node,
                            'replacement': replacement
                        },
                        source_line=self._get_source_line(self._get_line(node)),
                    ))

    def _analyze_nested_redundant_calls(self):
        """Find nested redundant or inverse calls like math.abs(math.abs(x)) or math.deg(math.rad(x))."""
        # (outer_func, inner_func) -> type ('redundant' or 'inverse')
        nested_patterns = {
            ('math.abs', 'math.abs'): 'redundant',
            ('math.floor', 'math.floor'): 'redundant',
            ('math.ceil', 'math.ceil'): 'redundant',
            ('tostring', 'tostring'): 'redundant',
            ('tonumber', 'tonumber'): 'redundant',
            ('string.lower', 'string.lower'): 'redundant',
            ('string.upper', 'string.upper'): 'redundant',
            ('math.deg', 'math.rad'): 'inverse',
            ('math.rad', 'math.deg'): 'inverse',
            ('math.log', 'math.exp'): 'inverse',
        }

        for call in self.calls:
            if len(call.args) != 1:
                continue

            inner = call.args[0]
            if not isinstance(inner, Call):
                continue

            _, _, inner_full_name = self._get_call_name(inner)
            pattern_key = (call.full_name, inner_full_name)

            if pattern_key in nested_patterns:
                pattern_type = nested_patterns[pattern_key]
                inner_arg_str = node_to_string(inner.args[0]) if inner.args else ""

                if pattern_type == 'redundant':
                    replacement = f'{inner_full_name}({inner_arg_str})'
                    message = f'Nested redundant call: {call.full_name}({inner_full_name}(x)) -> {inner_full_name}(x)'
                else: # inverse
                    replacement = inner_arg_str
                    message = f'Inverse operations: {call.full_name}({inner_full_name}(x)) -> x'

                if not replacement: # should not happen for these patterns
                    continue

                self.findings.append(Finding(
                    pattern_name='nested_redundant_call',
                    severity='GREEN',
                    line_num=call.line,
                    message=message,
                    details={
                        'node': call.node,
                        'replacement': replacement
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_logical_identity(self):
        """Find redundant logical operations like x and true, x or false."""
        if not self._ast_tree: return

        for node in ast.walk(self._ast_tree):
            # Check for binary comparison inversion: not (a == b) -> a ~= b
            if isinstance(node, ULNotOp) and isinstance(node.operand, (EqToOp, NotEqToOp, LessThanOp, GreaterThanOp, LessOrEqThanOp, GreaterOrEqThanOp)):
                op = node.operand
                left = node_to_string(op.left)
                right = node_to_string(op.right)

                inversion_map = {
                    EqToOp: '~=',
                    NotEqToOp: '==',
                    LessThanOp: '>=',
                    GreaterThanOp: '<=',
                    LessOrEqThanOp: '>',
                    GreaterOrEqThanOp: '<'
                }

                new_op = inversion_map[type(op)]
                replacement = f"{left} {new_op} {right}"

                self.findings.append(Finding(
                    pattern_name='comparison_inversion',
                    severity='GREEN',
                    line_num=self._get_line(node),
                    message=f'Comparison inversion: not ({left} {node_to_string(op).split()[1]} {right}) -> {replacement}',
                    details={
                        'node': node,
                        'replacement': replacement
                    },
                    source_line=self._get_source_line(self._get_line(node)),
                ))

            # Check for x and true, x and false, x or true, x or false
            if isinstance(node, (AndLoOp, OrLoOp)):
                left = node.left
                right = node.right

                target = None
                replacement = None
                reason = None

                l_bool = None
                if isinstance(left, TrueExpr): l_bool = True
                elif isinstance(left, FalseExpr): l_bool = False

                r_bool = None
                if isinstance(right, TrueExpr): r_bool = True
                elif isinstance(right, FalseExpr): r_bool = False

                if isinstance(node, AndLoOp):
                    if r_bool is True: target = left; reason = "x and true"
                    elif r_bool is False: replacement = "false"; reason = "x and false"
                    elif l_bool is True: target = right; reason = "true and x"
                    elif l_bool is False: replacement = "false"; reason = "false and x"
                else: # OrLoOp
                    if r_bool is False: target = left; reason = "x or false"
                    elif r_bool is True: replacement = "true"; reason = "x or true"
                    elif l_bool is False: target = right; reason = "false or x"
                    elif l_bool is True: replacement = "true"; reason = "true or x"

                if target or replacement:
                    # In Lua, (x and true) is NOT exactly x if x is not boolean (it keeps x)
                    # BUT if it's in a boolean context, it's safe to simplify.
                    is_bool_context = False
                    parent = self.parent_map.get(id(node))
                    if isinstance(parent, (If, While, Repeat)) and getattr(parent, 'test', None) == node:
                        is_bool_context = True
                    elif isinstance(parent, (AndLoOp, OrLoOp, ULNotOp)):
                        is_bool_context = True

                    if not is_bool_context and target:
                        # If not in boolean context, we can only simplify if target is already boolean
                        if self._infer_type(target) != 'boolean':
                            target = None

                    if target:
                        replacement = node_to_string(target)

                    if replacement:
                        self.findings.append(Finding(
                            pattern_name='logical_identity',
                            severity='GREEN',
                            line_num=self._get_line(node),
                            message=f'Logical identity simplification: {reason} -> {replacement}',
                            details={
                                'node': node,
                                'replacement': replacement
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))

            # Check for not not x where x is already boolean
            elif isinstance(node, ULNotOp) and isinstance(node.operand, ULNotOp):
                inner_operand = node.operand.operand
                if self._infer_type(inner_operand) == 'boolean':
                    replacement = node_to_string(inner_operand)
                    self.findings.append(Finding(
                        pattern_name='logical_identity',
                        severity='GREEN',
                        line_num=self._get_line(node),
                        message=f'Redundant double negation: not not {replacement} -> {replacement}',
                        details={
                            'node': node,
                            'replacement': replacement
                        },
                        source_line=self._get_source_line(self._get_line(node)),
                    ))

    def _analyze_string_starts_with(self):
        """Find string.find(s, prefix) == 1 and suggest string.sub(s, 1, #prefix) == prefix."""
        lua_regex_chars = set(b'^$()%.[]*+-?')
        for node in ast.walk(self._ast_tree):
            if isinstance(node, EqToOp):
                call = None

                if isinstance(node.left, Call) and isinstance(node.right, Number) and node.right.n == 1:
                    call = node.left
                elif isinstance(node.right, Call) and isinstance(node.left, Number) and node.left.n == 1:
                    call = node.right

                if call:
                    _, _, full_name = self._get_call_name(call)
                    if full_name == 'string.find' and len(call.args) >= 2:
                        s_node = call.args[0]
                        prefix_node = call.args[1]

                        if isinstance(prefix_node, String):
                            prefix_val = prefix_node.s
                            prefix_bytes = prefix_val if isinstance(prefix_val, bytes) else prefix_val.encode('utf-8', errors='replace')

                            # if it contains regex chars, we can't safely use this optimization
                            # unless it's a plain find, but we're suggesting sub which is plain anyway
                            if not any(b in lua_regex_chars for b in prefix_bytes):
                                s_str = node_to_string(s_node)
                                prefix_len = len(prefix_bytes)
                                prefix_text = prefix_bytes.decode('utf-8', errors='replace')

                                # suggestions
                                if prefix_len == 1:
                                    # string.byte(s) == code
                                    replacement = f'string.byte({s_str}) == {prefix_bytes[0]}'
                                    pattern = 'string_starts_with_byte'
                                else:
                                    replacement = f'string.sub({s_str}, 1, {prefix_len}) == "{prefix_text}"'
                                    pattern = 'string_starts_with_sub'

                                self.findings.append(Finding(
                                    pattern_name=pattern,
                                    severity='GREEN',
                                    line_num=self._get_line(node),
                                    message=f'string.find(s, "{prefix_text}") == 1 -> {replacement} (faster in LuaJIT)',
                                    details={
                                        's_str': s_str,
                                        'prefix': prefix_text,
                                        'replacement': replacement,
                                        'node': node,
                                    },
                                    source_line=self._get_source_line(self._get_line(node)),
                                ))

    def _analyze_algebraic_simplification(self):
        """Find redundant algebraic or string operations like x + 0, x * 1, s .. ""."""
        if not self._ast_tree: return

        ops = (AddOp, SubOp, MultOp, FloatDivOp, ExpoOp, Concat)

        for node in ast.walk(self._ast_tree):
            if isinstance(node, ops):
                left_val = self._get_const_val(node.left)
                right_val = self._get_const_val(node.right)

                target = None
                reason = None

                if isinstance(node, AddOp):
                    if right_val == 0: target = node.left; reason = "x + 0"
                    elif left_val == 0: target = node.right; reason = "0 + x"
                elif isinstance(node, SubOp):
                    if right_val == 0: target = node.left; reason = "x - 0"
                elif isinstance(node, Concat):
                    if right_val == "": target = node.left; reason = 's .. ""'
                    elif left_val == "": target = node.right; reason = '"" .. s'
                elif isinstance(node, MultOp):
                    if right_val == 1: target = node.left; reason = "x * 1"
                    elif left_val == 1: target = node.right; reason = "1 * x"
                elif isinstance(node, FloatDivOp):
                    if right_val == 1: target = node.left; reason = "x / 1"
                elif isinstance(node, ExpoOp):
                    if right_val == 1: target = node.left; reason = "x ^ 1"
                    elif right_val == 0:
                        # x ^ 0 is always 1 in Lua, even for 0 ^ 0
                        self.findings.append(Finding(
                            pattern_name='algebraic_simplification',
                            severity='GREEN',
                            line_num=self._get_line(node),
                            message=f'Algebraic simplification: x ^ 0 -> 1',
                            details={
                                'target_node': None,
                                'node': node,
                                'replacement': '1'
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))
                        continue

                if target:
                    # Type safety checks
                    is_safe = False
                    if isinstance(node, (AddOp, SubOp, MultOp, FloatDivOp, ExpoOp)):
                        if self._infer_type(target) == 'number':
                            is_safe = True
                    elif isinstance(node, Concat):
                        if self._infer_type(target) == 'string':
                            is_safe = True

                    if is_safe:
                        target_str = node_to_string(target)
                        self.findings.append(Finding(
                            pattern_name='algebraic_simplification',
                            severity='GREEN',
                            line_num=self._get_line(node),
                            message=f'Algebraic simplification: {reason} -> {target_str}',
                            details={
                                'target_node': target,
                                'node': node,
                                'replacement': target_str
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))

        # Check for table.concat(t, "") -> table.concat(t)
        for call in self.calls:
            if call.full_name == 'table.concat' and len(call.args) == 2:
                sep = self._get_const_val(call.args[1])
                if sep == "":
                    self.findings.append(Finding(
                        pattern_name='table_concat_default_sep',
                        severity='GREEN',
                        line_num=call.line,
                        message='table.concat(t, "") -> table.concat(t)',
                        details={
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

            # string.sub(s, 1, -1) -> s
            elif call.full_name == 'string.sub' and len(call.args) == 3:
                s_node = call.args[0]
                arg1 = self._get_const_val(call.args[1])
                arg2 = self._get_const_val(call.args[2])
                if arg1 == 1 and arg2 == -1:
                    if self._infer_type(s_node) == 'string':
                        s_str = node_to_string(s_node)
                        self.findings.append(Finding(
                            pattern_name='string_sub_identity',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'string.sub({s_str}, 1, -1) -> {s_str}',
                            details={
                                'node': call.node,
                                'replacement': s_str
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _analyze_redundant_tonumber_tostring(self):
        """Find redundant tonumber() or tostring() calls."""
        for call in self.calls:
            if call.full_name in ('tonumber', 'tostring') and len(call.args) == 1:
                arg = call.args[0]
                inferred = self._infer_type(arg)

                if (call.full_name == 'tonumber' and inferred == 'number') or \
                   (call.full_name == 'tostring' and inferred == 'string'):
                    arg_str = node_to_string(arg)
                    self.findings.append(Finding(
                        pattern_name='redundant_type_conversion',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'Redundant {call.full_name}({arg_str})',
                        details={
                            'func': call.full_name,
                            'arg_str': arg_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_string_byte_1(self):
        """Find string.byte(s, 1) and suggest string.byte(s)."""
        for call in self.calls:
            if call.full_name == 'string.byte' and len(call.args) == 2:
                if isinstance(call.args[1], Number) and call.args[1].n == 1:
                    s_str = node_to_string(call.args[0])
                    self.findings.append(Finding(
                        pattern_name='string_byte_1',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'string.byte({s_str}, 1) -> string.byte({s_str})',
                        details={
                            's_str': s_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_pairs_to_next(self):
        """Suggest next, t instead of pairs(t) in hot loops for LuaJIT."""
        for call in self.calls:
            if call.full_name == 'pairs' and (call.scope.is_hot_callback or call.in_loop):
                table_name = node_to_string(call.args[0]) if call.args else "t"
                self.findings.append(Finding(
                    pattern_name='pairs_to_next',
                    severity='YELLOW',
                    line_num=call.line,
                    message=f'pairs({table_name}) in hot loop -> next, {table_name}, nil is faster in LuaJIT',
                    details={
                        'table': table_name,
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_return_ternary(self):
        """Find if cond then return a else return b end and suggest ternary return."""
        if not self._ast_tree: return
        for node in ast.walk(self._ast_tree):
            if isinstance(node, If) and node.orelse and isinstance(node.orelse, Block):
                if len(node.body.body) == 1 and len(node.orelse.body) == 1:
                    stmt1 = node.body.body[0]
                    stmt2 = node.orelse.body[0]

                    if isinstance(stmt1, Return) and isinstance(stmt2, Return):
                        if len(stmt1.values) == 1 and len(stmt2.values) == 1:
                            v1 = stmt1.values[0]
                            v2 = stmt2.values[0]

                            # check if they are simple enough for ternary
                            # AND v1 must be truthy for Lua ternary pattern to work
                            if self._is_simple_expr(v1) and self._is_simple_expr(v2) and self._is_guaranteed_truthy(v1):
                                cond_str = node_to_string(node.test)
                                v1_str = node_to_string(v1)
                                v2_str = node_to_string(v2)

                                self.findings.append(Finding(
                                    pattern_name='return_ternary_simplification',
                                    severity='YELLOW',
                                    line_num=self._get_line(node),
                                    message=f'if {cond_str} then return {v1_str} else return {v2_str} end -> return {cond_str} and {v1_str} or {v2_str}',
                                    details={
                                        'cond': cond_str,
                                        'v1': v1_str,
                                        'v2': v2_str,
                                        'node': node,
                                    },
                                    source_line=self._get_source_line(self._get_line(node)),
                                ))

    def _analyze_string_match_existence(self):
        """Find string.match() used in boolean context with plain patterns."""
        lua_regex_chars = set('^$()%.[]*+-?')
        for call in self.calls:
            if call.full_name == 'string.match' and len(call.args) == 2:
                # check if it's in a boolean context
                is_bool_context = False
                parent = self.parent_map.get(id(call.node))
                if isinstance(parent, (If, While, Repeat)) and getattr(parent, 'test', None) == call.node:
                    is_bool_context = True
                elif isinstance(parent, (AndLoOp, OrLoOp, ULNotOp)):
                    is_bool_context = True

                if is_bool_context:
                    pattern_node = call.args[1]
                    if isinstance(pattern_node, String):
                        pattern_text = pattern_node.s
                        if isinstance(pattern_text, bytes):
                            pattern_text = pattern_text.decode('utf-8', errors='replace')

                        if not any(c in lua_regex_chars for c in pattern_text):
                            self.findings.append(Finding(
                                pattern_name='string_match_existence',
                                severity='GREEN',
                                line_num=call.line,
                                message=f'string.match(s, "{pattern_text}") in boolean context -> use string.find(s, "{pattern_text}", 1, true)',
                                details={
                                    'pattern': pattern_text,
                                    's_str': node_to_string(call.args[0]),
                                    'node': call.node,
                                },
                                source_line=self._get_source_line(call.line),
                            ))

    def _analyze_unpack_to_indexing(self):
        """Find local a, b = unpack(t) and suggest direct indexing."""
        if not self._ast_tree: return
        for node in ast.walk(self._ast_tree):
            if isinstance(node, LocalAssign) and len(node.targets) >= 2 and len(node.values) == 1:
                val = node.values[0]
                if isinstance(val, Call):
                    _, _, full_name = self._get_call_name(val)
                    if full_name == 'unpack' and len(val.args) == 1:
                        if len(node.targets) <= 4:
                            table_name = node_to_string(val.args[0])
                            self.findings.append(Finding(
                                pattern_name='unpack_to_indexing',
                                severity='GREEN',
                                line_num=self._get_line(node),
                                message=f'unpack({table_name}) to multiple locals -> use direct indexing',
                                details={
                                    'table': table_name,
                                    'targets': [node_to_string(t) for t in node.targets],
                                    'node': node,
                                },
                                source_line=self._get_source_line(self._get_line(node)),
                            ))

    def _analyze_divide_by_constant(self):
        """Find division by constant and suggest multiplication."""
        if not self._ast_tree: return
        for node in ast.walk(self._ast_tree):
            if isinstance(node, FloatDivOp):
                if isinstance(node.right, Number) and node.right.n != 0:
                    divisor = node.right.n
                    # suggest for common ones
                    if divisor in (2, 4, 5, 8, 10, 16, 20, 25, 32, 40, 50, 64, 100, 128, 256, 512, 1024):
                        reciprocal = 1.0 / divisor
                        # format nicely
                        if reciprocal == int(reciprocal):
                            rec_str = str(int(reciprocal))
                        else:
                            rec_str = f"{reciprocal:.6g}"

                        self.findings.append(Finding(
                            pattern_name='divide_by_constant',
                            severity='GREEN',
                            line_num=self._get_line(node),
                            message=f'Division by {divisor} -> multiply by {rec_str}',
                            details={
                                'divisor': divisor,
                                'reciprocal': rec_str,
                                'node': node,
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))

    def _analyze_if_nil_assign(self):
        """Find if x == nil then x = val end patterns."""
        if not self._ast_tree: return
        for node in ast.walk(self._ast_tree):
            if isinstance(node, If) and not node.orelse:
                # check condition: x == nil or not x
                is_nil_check = False
                var_name = None

                test = node.test
                if isinstance(test, EqToOp):
                    if isinstance(test.left, Name) and isinstance(test.right, Nil):
                        var_name = test.left.id
                        is_nil_check = True
                    elif isinstance(test.right, Name) and isinstance(test.left, Nil):
                        var_name = test.right.id
                        is_nil_check = True
                elif isinstance(test, ULNotOp) and isinstance(test.operand, Name):
                    var_name = test.operand.id
                    is_nil_check = True

                if is_nil_check and var_name:
                    # check body: x = val
                    if isinstance(node.body, Block) and len(node.body.body) == 1:
                        stmt = node.body.body[0]
                        if isinstance(stmt, Assign) and len(stmt.targets) == 1 and len(stmt.values) == 1:
                            target = stmt.targets[0]
                            if isinstance(target, Name) and target.id == var_name:
                                val_str = node_to_string(stmt.values[0])
                                self.findings.append(Finding(
                                    pattern_name='if_nil_assign',
                                    severity='YELLOW',
                                    line_num=self._get_line(node),
                                    message=f'if {var_name} == nil then {var_name} = {val_str} end -> {var_name} = {var_name} or {val_str}',
                                    details={
                                        'var': var_name,
                                        'val': val_str,
                                        'node': node,
                                    },
                                    source_line=self._get_source_line(self._get_line(node)),
                                ))

    def _analyze_plain_string_find(self):
        """Find string.find() with plain strings that can use plain=true flag."""
        lua_regex_chars = set('^$()%.[]*+-?')
        for call in self.calls:
            if call.full_name == 'string.find' and len(call.args) == 2:
                pattern_node = call.args[1]
                if isinstance(pattern_node, String):
                    pattern_text = pattern_node.s
                    if isinstance(pattern_text, bytes):
                        pattern_text = pattern_text.decode('utf-8', errors='replace')

                    if not any(c in lua_regex_chars for c in pattern_text):
                        self.findings.append(Finding(
                            pattern_name='string_find_plain',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'string.find("{pattern_text}") -> use plain=true for performance',
                            details={
                                'pattern': pattern_text,
                                'full_match': f'string.find({node_to_string(call.args[0])}, "{pattern_text}")',
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _analyze_pairs_on_array(self):
        """Find pairs() used in hot callbacks where ipairs() or numeric loops might be faster."""
        for call in self.calls:
            if call.full_name == 'pairs' and call.scope.is_hot_callback:
                table_name = node_to_string(call.args[0]) if call.args else "table"
                self.findings.append(Finding(
                    pattern_name='pairs_on_array',
                    severity='YELLOW',
                    line_num=call.line,
                    message=f'pairs({table_name}) in hot callback -> check if ipairs() or numeric loop is possible',
                    details={
                        'table': table_name,
                        'function': call.scope.name,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_ipairs_hot_loop(self):
        """Find ipairs() used in hot callbacks where numeric loops are faster."""
        for call in self.calls:
            if call.full_name == 'ipairs' and call.scope.is_hot_callback:
                table_name = node_to_string(call.args[0]) if call.args else "table"

                # check if it's used in a Forin loop
                parent_loop = None
                curr = call.node
                while True:
                    curr = self.parent_map.get(id(curr))
                    if not curr: break
                    if isinstance(curr, Forin):
                        parent_loop = curr
                        break

                if parent_loop and len(parent_loop.targets) == 2:
                    k_var = node_to_string(parent_loop.targets[0])
                    v_var = node_to_string(parent_loop.targets[1])

                    # only auto-fix if table is a simple variable (to avoid double evaluation)
                    is_simple_table = self._is_simple_expr(call.args[0]) if call.args else False
                    severity = 'GREEN' if is_simple_table else 'YELLOW'

                    self.findings.append(Finding(
                        pattern_name='ipairs_hot_loop',
                        severity=severity,
                        line_num=call.line,
                        message=f'ipairs({table_name}) in hot callback -> for i=1, #{table_name} do is faster in LuaJIT',
                        details={
                            'table': table_name,
                            'k_var': k_var,
                            'v_var': v_var,
                            'node': call.node,
                            'loop_node': parent_loop,
                        },
                        source_line=self._get_source_line(call.line),
                    ))
                else:
                    self.findings.append(Finding(
                        pattern_name='ipairs_hot_loop',
                        severity='YELLOW',
                        line_num=call.line,
                        message=f'ipairs({table_name}) in hot callback -> for i=1, #{table_name} do is faster in LuaJIT',
                        details={
                            'table': table_name,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_math_min_max(self):
        """Find math.min(a, b) and math.max(a, b) with simple arguments."""
        for call in self.calls:
            if call.full_name in ('math.min', 'math.max') and len(call.args) == 2:
                arg1 = call.args[0]
                arg2 = call.args[1]

                s1 = node_to_string(arg1)
                s2 = node_to_string(arg2)

                # Identity: math.min(x, x) -> x
                if s1 == s2:
                    self.findings.append(Finding(
                        pattern_name='math_identity',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'Math identity: {call.full_name}({s1}, {s1}) -> {s1}',
                        details={
                            'node': call.node,
                            'replacement': s1
                        },
                        source_line=self._get_source_line(call.line),
                    ))
                    continue

                # Identity with math.huge
                v1 = self._get_const_val(arg1)
                v2 = self._get_const_val(arg2)

                target = None
                if call.full_name == 'math.min':
                    if v2 == float('inf') or v2 == py_math.inf: target = arg1
                    elif v1 == float('inf') or v1 == py_math.inf: target = arg2
                else: # math.max
                    if v2 == float('-inf') or v2 == -py_math.inf: target = arg1
                    elif v1 == float('-inf') or v1 == -py_math.inf: target = arg2

                if target:
                    res_str = node_to_string(target)
                    self.findings.append(Finding(
                        pattern_name='math_identity',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'Math identity with infinity: {call.full_name}(...) -> {res_str}',
                        details={
                            'node': call.node,
                            'replacement': res_str
                        },
                        source_line=self._get_source_line(call.line),
                    ))
                    continue

                if self._is_simple_expr(arg1) and self._is_simple_expr(arg2):
                    a_str = node_to_string(arg1)
                    b_str = node_to_string(arg2)
                    op = '<' if call.full_name == 'math.min' else '>'

                    self.findings.append(Finding(
                        pattern_name='math_min_max_inline',
                        severity='YELLOW',
                        line_num=call.line,
                        message=f'{call.full_name}({a_str}, {b_str}) -> {a_str} {op} {b_str} and {a_str} or {b_str}',
                        details={
                            'func': call.full_name,
                            'arg1': a_str,
                            'arg2': b_str,
                            'op': op,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_slow_loop_funcs(self):
        """Find potentially slow operations in hot loops."""
        for call in self.calls:
            if call.in_loop and call.full_name in ('table.insert', 'table.remove'):
                is_slow = False
                if call.full_name == 'table.insert' and len(call.args) == 3:
                    is_slow = True
                elif call.full_name == 'table.remove' and len(call.args) >= 2:
                    is_slow = True

                if is_slow:
                    severity = 'RED' if call.scope.is_hot_callback else 'YELLOW'
                    self.findings.append(Finding(
                        pattern_name='slow_loop_operation',
                        severity=severity,
                        line_num=call.line,
                        message=f'Potentially slow operation {call.full_name}() in loop (O(N))',
                        details={'func': call.full_name},
                        source_line=self._get_source_line(call.line)
                    ))

    def _analyze_string_sub_to_byte_simple(self):
        """Find string.sub(s, i, i) calls that can be optimized to string.char(string.byte(s, i))."""
        for call in self.calls:
            if call.full_name == 'string.sub' and len(call.args) >= 2:
                arg1 = call.args[0]
                arg2 = call.args[1]
                arg3 = call.args[2] if len(call.args) > 2 else None

                # Identity: string.sub(s, 1) -> s
                if len(call.args) == 2 and isinstance(arg2, Number) and arg2.n == 1:
                    if self._infer_type(arg1) == 'string':
                        s_str = node_to_string(arg1)
                        self.findings.append(Finding(
                            pattern_name='string_identity',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'String identity: string.sub({s_str}, 1) -> {s_str}',
                            details={
                                'node': call.node,
                                'replacement': s_str
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue

                # string.sub(s, i, i)
                is_single_char = False
                if arg3 is not None:
                    if node_to_string(arg2) == node_to_string(arg3):
                        is_single_char = True
                elif isinstance(arg2, Number) and arg2.n == 1:
                    # string.sub(s, 1)
                    is_single_char = True

                if is_single_char:
                    s_str = node_to_string(arg1)
                    i_str = node_to_string(arg2)

                    # if i is 1, we can just use string.byte(s)
                    if i_str == "1":
                        msg = f'string.sub({s_str}, 1, 1) -> string.char(string.byte({s_str}))'
                    else:
                        msg = f'string.sub({s_str}, {i_str}, {i_str}) -> string.char(string.byte({s_str}, {i_str}))'

                    self.findings.append(Finding(
                        pattern_name='string_sub_to_byte_simple',
                        severity='GREEN',
                        line_num=call.line,
                        message=msg,
                        details={
                            's_str': s_str,
                            'i_str': i_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_string_sub_to_byte(self):
        """Find string.sub(s, n, n) == "c" that can be string.byte(s, n) == code."""
        for node in ast.walk(self._ast_tree):
            if isinstance(node, (EqToOp, NotEqToOp)):
                call = None
                const_str = None

                if isinstance(node.left, Call) and isinstance(node.right, String):
                    call = node.left
                    const_str = node.right.s
                elif isinstance(node.right, Call) and isinstance(node.left, String):
                    call = node.right
                    const_str = node.left.s

                if call and const_str and len(const_str) == 1:
                    _, _, full_name = self._get_call_name(call)
                    if full_name == 'string.sub' and len(call.args) >= 2:
                        # check if it's string.sub(s, n, n)
                        s_str = node_to_string(call.args[0])
                        n_str = node_to_string(call.args[1])
                        m_str = node_to_string(call.args[2]) if len(call.args) > 2 else None

                        if n_str == m_str or (m_str is None and n_str == "1"):
                            byte_val = ord(const_str)
                            op = '==' if isinstance(node, EqToOp) else '~='

                            self.findings.append(Finding(
                                pattern_name='string_sub_to_byte',
                                severity='YELLOW',
                                line_num=self._get_line(node),
                                message=f'string.sub({s_str}, {n_str}) {op} "{const_str}" -> string.byte({s_str}, {n_str}) {op} {byte_val}',
                                details={
                                    's_str': s_str,
                                    'n_str': n_str,
                                    'op': op,
                                    'byte_val': byte_val,
                                    'node': node,
                                },
                                source_line=self._get_source_line(self._get_line(node)),
                            ))

    def _analyze_table_new_in_loop(self):
        """Find table re-initialization in loops."""
        for assign in self.assigns:
            if assign.in_loop and assign.value_type == 'other' and isinstance(assign.node, Table):
                # check if the table is empty
                if not assign.node.fields:
                    self.findings.append(Finding(
                        pattern_name='table_new_in_loop',
                        severity='YELLOW',
                        line_num=assign.line,
                        message=f'Table re-initialization {assign.target} = {{}} in loop -> use table.clear()',
                        details={
                            'target': assign.target,
                            'node': assign.node,
                        },
                        source_line=self._get_source_line(assign.line),
                    ))

    def _analyze_expo_to_mult(self):
        """Find x^2, x^3 etc. that can be simplified to multiplication."""
        for node in ast.walk(self._ast_tree):
            if isinstance(node, ExpoOp):
                if isinstance(node.right, Number) and node.right.n in (2, 3):
                    base_str = node_to_string(node.left)
                    exponent = int(node.right.n)

                    if self._is_simple_expr(node.left):
                        replacement = '*'.join([base_str] * exponent)
                        self.findings.append(Finding(
                            pattern_name='expo_to_mult',
                            severity='GREEN',
                            line_num=self._get_line(node),
                            message=f'Exponent to multiplication: {base_str}^{exponent} -> {replacement}',
                            details={
                                'replacement': replacement,
                                'node': node,
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))
                    else:
                        # Complex expression like (a+b)^2
                        self.findings.append(Finding(
                            pattern_name='expo_to_mult_complex',
                            severity='YELLOW',
                            line_num=self._get_line(node),
                            message=f'Exponent to multiplication: ({base_str})^{exponent} -> use local variable for base to avoid redundant calculation',
                            details={
                                'base_str': base_str,
                                'exponent': exponent,
                                'node': node,
                            },
                            source_line=self._get_source_line(self._get_line(node)),
                        ))

    def _analyze_constant_folding(self):
        """Find arithmetic and string operations on constant values that can be folded."""
        if not self._ast_tree: return

        for node in ast.walk(self._ast_tree):
            # Skip terminal nodes
            if isinstance(node, (Number, String, TrueExpr, FalseExpr, Nil, Name)):
                continue

            # Skip Index nodes like math.huge, math.pi - they are more readable as is
            if isinstance(node, Index) and node_to_string(node) in ('math.huge', 'math.pi'):
                continue

            val = self._get_const_val(node)
            if val is not None:
                # If parent is also constant, skip this one to fold at highest level
                parent = self.parent_map.get(id(node))
                if parent:
                    p_val = self._get_const_val(parent)
                    if p_val is not None:
                        continue

                # Format result
                if isinstance(val, (int, float)):
                    # Don't fold to infinity or NaN as it breaks Lua readability/correctness
                    if not py_math.isfinite(val):
                        continue
                    if val == int(val) and not isinstance(val, bool):
                        res_str = str(int(val))
                    else:
                        res_str = f"{val:.10g}"
                elif isinstance(val, bool):
                    res_str = str(val).lower()
                elif isinstance(val, str):
                    res_str = f'"{val}"'
                else:
                    continue

                self.findings.append(Finding(
                    pattern_name='constant_folding',
                    severity='GREEN',
                    line_num=self._get_line(node),
                    message=f'Constant folding: -> {res_str}',
                    details={'result': res_str, 'node': node},
                    source_line=self._get_source_line(self._get_line(node)),
                ))

    def _analyze_repeated_member_access_in_loop(self):
        """Find repeated member access (e.g., self.object) in loops."""
        # track access counts per loop scope
        loop_accesses = defaultdict(lambda: defaultdict(list))

        for node, line, scope, is_write in self.index_accesses:
            if not is_write:
                # check if it's a member access like self.xxx
                if isinstance(node, Index) and isinstance(node.value, Name) and node.value.id == 'self':
                    # find enclosing loop
                    curr = scope
                    while curr and curr.scope_type != 'loop':
                        curr = curr.parent

                    if curr:
                        member_name = node_to_string(node)
                        loop_accesses[id(curr)][member_name].append((node, line))

        for loop_id, members in loop_accesses.items():
            for member, accesses in members.items():
                if len(accesses) >= 3:
                    first_node, first_line = accesses[0]
                    self.findings.append(Finding(
                        pattern_name='repeated_member_access_in_loop',
                        severity='YELLOW',
                        line_num=first_line,
                        message=f'Repeated access to {member} in loop ({len(accesses)}x) -> localize it',
                        details={
                            'member': member,
                            'count': len(accesses),
                            'lines': [a[1] for a in accesses],
                        },
                        source_line=self._get_source_line(first_line),
                    ))

    def _analyze_luajit_nyi(self):
        """Find LuaJIT NYI functions in hot callbacks or loops."""
        for call in self.calls:
            is_nyi = False
            if call.full_name in LUAJIT_NYI_FUNCS:
                # Some functions are only NYI with specific arguments
                if call.full_name == 'table.insert' and len(call.args) < 3:
                    is_nyi = False # table.insert(t, v) is NOT NYI
                elif call.full_name == 'table.remove' and len(call.args) < 2:
                    is_nyi = False # table.remove(t) is NOT NYI
                else:
                    is_nyi = True

            if is_nyi:
                is_hot = call.scope.is_hot_callback
                in_loop = call.in_loop

                if is_hot or in_loop:
                    severity = 'RED' if is_hot and in_loop else 'YELLOW'
                    context = "hot callback" if is_hot else "loop"
                    if is_hot and in_loop: context = "hot loop"

                    self.findings.append(Finding(
                        pattern_name='luajit_nyi_warning',
                        severity=severity,
                        line_num=call.line,
                        message=f'LuaJIT NYI: {call.full_name}() in {context} aborts JIT compilation',
                        details={'func': call.full_name},
                        source_line=self._get_source_line(call.line)
                    ))

    def _analyze_string_lower_case(self):
        """Find string.lower(s) == "UPPER" which is always false."""
        for node in ast.walk(self._ast_tree):
            if isinstance(node, (EqToOp, NotEqToOp)):
                call = None
                const_str = None

                left = node.left
                right = node.right

                if isinstance(left, Call) and isinstance(right, String):
                    call = left
                    const_str = right.s
                elif isinstance(right, Call) and isinstance(left, String):
                    call = right
                    const_str = left.s

                if call and const_str:
                    if isinstance(const_str, bytes):
                        const_str = const_str.decode('utf-8', errors='replace')

                    _, _, full_name = self._get_call_name(call)
                    if full_name == 'string.lower':
                        if any(c.isupper() for c in const_str):
                            is_eq = isinstance(node, EqToOp)
                            severity = 'RED' if is_eq else 'YELLOW'
                            msg = f'string.lower() comparison with uppercase string "{const_str}" is always {"false" if is_eq else "true"}'
                            self.findings.append(Finding(
                                pattern_name='always_false_comparison',
                                severity=severity,
                                line_num=self._get_line(node),
                                message=msg,
                                details={'const': const_str},
                                source_line=self._get_source_line(self._get_line(node))
                            ))

    def _analyze_string_format_to_concat(self):
        """Find string.format("%s...", ...) that can be concatenation."""
        for call in self.calls:
            if call.full_name == 'string.format' and len(call.args) >= 2:
                fmt_node = call.args[0]
                if isinstance(fmt_node, String):
                    fmt = fmt_node.s
                    if isinstance(fmt, bytes):
                        fmt = fmt.decode('utf-8', errors='replace')

                    # only handle simple %s and constant text
                    placeholders = re.findall(r'%[0-9.]*[a-zA-Z%]', fmt)
                    if placeholders and all(p == '%s' for p in placeholders) and len(placeholders) == len(call.args) - 1:
                        # build concatenation string
                        parts = fmt.split('%s')
                        concat_parts = []
                        arg_idx = 1
                        for i, part in enumerate(parts):
                            if part:
                                # escape quotes if needed, but for now just use it
                                concat_parts.append(f'"{part}"')
                            if i < len(parts) - 1:
                                arg_node = call.args[arg_idx]
                                arg_str = node_to_string(arg_node)
                                # numbers are safe to concat directly in Lua
                                if not isinstance(arg_node, (Number, String, Name, Index)):
                                    arg_str = f'({arg_str})'

                                # to match string.format, we should technically use tostring()
                                # but usually %s is used on things that are already strings or numbers
                                concat_parts.append(arg_str)
                                arg_idx += 1

                        replacement = ' .. '.join(concat_parts)
                        self.findings.append(Finding(
                            pattern_name='string_format_to_concat',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'string.format("{fmt}", ...) -> concatenation: {replacement}',
                            details={
                                'replacement': replacement,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _analyze_redundant_return_bool(self):
        """Find if cond then return true else return false end."""
        for node in ast.walk(self._ast_tree):
            if isinstance(node, If) and node.orelse and isinstance(node.orelse, Block):
                # check if body is 'return true' and orelse is 'return false'
                if len(node.body.body) == 1 and len(node.orelse.body) == 1:
                    stmt1 = node.body.body[0]
                    stmt2 = node.orelse.body[0]

                    if isinstance(stmt1, Return) and isinstance(stmt2, Return):
                        if len(stmt1.values) == 1 and len(stmt2.values) == 1:
                            val1 = stmt1.values[0]
                            val2 = stmt2.values[0]

                            if (isinstance(val1, TrueExpr) and isinstance(val2, FalseExpr)) or \
                               (isinstance(val1, FalseExpr) and isinstance(val2, TrueExpr)):
                                cond_str = node_to_string(node.test)
                                inverted = isinstance(val1, FalseExpr)
                                msg = f'if {cond_str} then return {"false" if inverted else "true"} else return {"true" if inverted else "false"} end -> return {"not " if inverted else ""}{cond_str}'
                                self.findings.append(Finding(
                                    pattern_name='redundant_return_bool',
                                    severity='YELLOW',
                                    line_num=self._get_line(node),
                                    message=msg,
                                    details={
                                        'cond_str': cond_str,
                                        'inverted': inverted,
                                        'node': node,
                                    },
                                    source_line=self._get_source_line(self._get_line(node)),
                                ))

    def _analyze_redundant_boolean_comp(self):
        """Find redundant boolean comparisons like x == true or x == false."""
        for node in ast.walk(self._ast_tree):
            if isinstance(node, (EqToOp, NotEqToOp)):
                left = node.left
                right = node.right

                target_node = None
                bool_val = None

                if isinstance(right, (TrueExpr, FalseExpr)):
                    target_node = left
                    bool_val = isinstance(right, TrueExpr)
                elif isinstance(left, (TrueExpr, FalseExpr)):
                    target_node = right
                    bool_val = isinstance(left, TrueExpr)

                if target_node:
                    op = '==' if isinstance(node, EqToOp) else '~='
                    target_str = node_to_string(target_node)

                    # Result is true if (val == true) or (val ~= false)
                    # Result is false if (val == false) or (val ~= true)
                    # We want to simplify this.

                    self.findings.append(Finding(
                        pattern_name='redundant_boolean_comp',
                        severity='GREEN',
                        line_num=self._get_line(node),
                        message=f'Redundant boolean comparison: {target_str} {op} {str(bool_val).lower()}',
                        details={
                            'target_node': target_node,
                            'bool_val': bool_val,
                            'op': op,
                            'full_node': node
                        },
                        source_line=self._get_source_line(self._get_line(node))
                    ))

    def _analyze_redundant_tostring(self):
        """Find tostring(s) where s is already a string literal."""
        for call in self.calls:
            if call.full_name == 'tostring' and len(call.args) == 1:
                arg = call.args[0]
                if isinstance(arg, String):
                    s_val = node_to_string(arg)
                    self.findings.append(Finding(
                        pattern_name='redundant_tostring',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'redundant tostring({s_val})',
                        details={
                            'value': s_val,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_string_format_in_loop(self):
        """Find string.format() used in loops (can be expensive)."""
        for call in self.calls:
            if call.full_name == 'string.format' and call.in_loop:
                self.findings.append(Finding(
                    pattern_name='string_format_in_loop',
                    severity='YELLOW',
                    line_num=call.line,
                    message='string.format() in loop -> consider simple concatenation or pre-calculating',
                    details={
                        'loop_depth': call.loop_depth,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_table_insert(self):
        """Find table.insert(t, v) and table.insert(t, #t+1, v) patterns."""
        for call in self.calls:
            if call.full_name == 'table.insert':
                if len(call.args) == 3:
                    # check for table.insert(t, #t+1, v)
                    pos_node = call.args[1]
                    table_node = call.args[0]
                    table_name = node_to_string(table_node)

                    if isinstance(pos_node, AddOp):
                        # check if it's #t + 1 or table.getn(t) + 1
                        left = pos_node.left
                        right = pos_node.right

                        is_len_plus_1 = False
                        if isinstance(right, Number) and right.n == 1:
                            if isinstance(left, ULengthOP):
                                if node_to_string(left.operand) == table_name:
                                    is_len_plus_1 = True
                            elif isinstance(left, Call):
                                _, _, l_full = self._get_call_name(left)
                                if l_full == 'table.getn' and len(left.args) == 1:
                                    if node_to_string(left.args[0]) == table_name:
                                        is_len_plus_1 = True

                        if is_len_plus_1:
                                self.findings.append(Finding(
                                    pattern_name='table_insert_append_len',
                                    severity='GREEN',
                                    line_num=call.line,
                                    message=f'table.insert({table_name}, #{table_name}+1, v) -> {table_name}[#{table_name}+1] = v',
                                    details={
                                        'table': table_name,
                                        'value': node_to_string(call.args[2]),
                                        'node': call.node,
                                    },
                                    source_line=self._get_source_line(call.line),
                                ))
                                continue

                if len(call.args) == 2:
                    # 2-arg form: table.insert(t, v)
                    table_name = node_to_string(call.args[0])
                    value = node_to_string(call.args[1])

                    self.findings.append(Finding(
                        pattern_name='table_insert_append',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'table.insert({table_name}, v) -> {table_name}[#{table_name}+1] = v',
                        details={
                            'table': table_name,
                            'value': value,
                            'full_match': f'table.insert({table_name}, {value})',
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))
                elif len(call.args) == 3:
                    # 3-arg form: table.insert(t, pos, v)
                    pos_node = call.args[1]
                    if isinstance(pos_node, Number) and pos_node.n == 1:
                        table_name = node_to_string(call.args[0])
                        self.findings.append(Finding(
                            pattern_name='table_insert_front',
                            severity='YELLOW',
                            line_num=call.line,
                            message=f'table.insert({table_name}, 1, v) is O(n) -> avoid inserting at front of large tables',
                            details={
                                'table': table_name,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _analyze_table_remove(self):
        """Find table.remove(t) or table.remove(t, #t) patterns that can be t[#t] = nil."""
        for call in self.calls:
            if call.full_name == 'table.remove':
                is_last = False
                if len(call.args) == 1:
                    is_last = True
                elif len(call.args) == 2:
                    # check if 2nd arg is #t
                    table_name = node_to_string(call.args[0])
                    pos_node = call.args[1]
                    if isinstance(pos_node, ULengthOP) and node_to_string(pos_node.operand) == table_name:
                        is_last = True

                if not is_last:
                    continue

                # SAFETY: only safe to auto-fix if return value is NOT used
                # We check this by seeing if the call is part of an assignment or other expression
                # This is a bit complex without full data-flow, so we'll just check if it's
                # a standalone statement in a block.

                is_standalone = False
                parent = self.parent_map.get(id(call.node))
                if isinstance(parent, Block):
                    is_standalone = True

                # Severity YELLOW if potentially unsafe, GREEN if standalone
                severity = 'GREEN' if is_standalone else 'YELLOW'

                table_name = node_to_string(call.args[0])
                self.findings.append(Finding(
                    pattern_name='table_remove_last',
                    severity=severity,
                    line_num=call.line,
                    message=f'table.remove({table_name}) -> {table_name}[#{table_name}] = nil',
                    details={
                        'table': table_name,
                        'full_match': f'table.remove({table_name})',
                        'node': call.node,
                        'is_standalone': is_standalone,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_deprecated_funcs(self):
        """Find deprecated functions: table.getn, string.len."""
        for call in self.calls:
            if call.full_name == 'table.getn' and len(call.args) == 1:
                arg = node_to_string(call.args[0])
                self.findings.append(Finding(
                    pattern_name='table_getn',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'table.getn({arg}) -> #{arg}',
                    details={
                        'table': arg,
                        'full_match': f'table.getn({arg})',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

            elif call.full_name == 'string.len' and len(call.args) == 1:
                arg = node_to_string(call.args[0])
                self.findings.append(Finding(
                    pattern_name='string_len',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'string.len({arg}) -> #{arg}',
                    details={
                        'string': arg,
                        'full_match': f'string.len({arg})',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_math_random_1(self):
        """Find math.random(1, N) that can be math.random(N)."""
        for call in self.calls:
            if call.full_name == 'math.random' and len(call.args) == 2:
                arg1 = call.args[0]
                if isinstance(arg1, Number) and arg1.n == 1:
                    n_str = node_to_string(call.args[1])
                    self.findings.append(Finding(
                        pattern_name='math_random_1',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'math.random(1, {n_str}) -> math.random({n_str})',
                        details={
                            'n_str': n_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_math_abs_positive(self):
        """Find math.abs() on values that are always positive."""
        # Functions that always return positive or zero results
        positive_funcs = {'time_global', 'device().time_global', 'level.get_time_days',
                         'math.random', 'math.sqrt', 'math.exp'}
        # (outer_func, inner_func) pairs where outer(inner(x)) is redundant
        redundant_wrappers = {
            ('math.abs', 'math.sqrt'),
            ('math.abs', 'math.exp'),
            ('math.abs', 'math.abs'), # also handled in nested_redundant_call, but fine here too
        }

        for call in self.calls:
            if call.full_name == 'math.abs' and len(call.args) == 1:
                arg = call.args[0]
                is_positive = False
                if isinstance(arg, Number) and arg.n >= 0:
                    is_positive = True
                elif isinstance(arg, ULengthOP):
                    is_positive = True
                elif isinstance(arg, Call):
                    _, _, inner_name = self._get_call_name(arg)
                    if inner_name in positive_funcs:
                        is_positive = True
                    elif (call.full_name, inner_name) in redundant_wrappers:
                        is_positive = True

                if is_positive:
                    arg_str = node_to_string(arg)
                    self.findings.append(Finding(
                        pattern_name='math_abs_positive',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'math.abs({arg_str}) is redundant',
                        details={
                            'arg_str': arg_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_table_concat_literal(self):
        """Find table.concat({a, b}) that can be a .. b."""
        for call in self.calls:
            if call.full_name == 'table.concat' and len(call.args) >= 1:
                arg1 = call.args[0]
                if isinstance(arg1, Table) and len(arg1.fields) >= 2 and len(arg1.fields) <= 4:
                    # check if all fields are simple values (no complex expressions)
                    all_simple = True
                    parts = []
                    for field in arg1.fields:
                        if field.key: # named key {k=v} - not handled by concat on array
                            all_simple = False
                            break
                        if not self._is_simple_expr(field.value) and not isinstance(field.value, (Call, Invoke)):
                            # expressions like a+b might need parens, skip for now
                            all_simple = False
                            break
                        parts.append(node_to_string(field.value))

                    if all_simple:
                        sep = '""'
                        if len(call.args) >= 2:
                            if isinstance(call.args[1], String):
                                sep = node_to_string(call.args[1])
                            else:
                                all_simple = False # complex separator

                        if all_simple:
                            # if separator is "", just join with ..
                            # otherwise join with .. sep ..
                            if sep in ('""', "''"):
                                replacement = ' .. '.join(parts)
                            else:
                                replacement = f' .. {sep} .. '.join(parts)

                            self.findings.append(Finding(
                                pattern_name='table_concat_literal',
                                severity='GREEN',
                                line_num=call.line,
                                message=f'table.concat({{...}}) -> concatenation: {replacement}',
                                details={
                                    'replacement': replacement,
                                    'node': call.node,
                                },
                                source_line=self._get_source_line(call.line),
                            ))

    def _analyze_math_random_0_1(self):
        """Find math.random(0, 1) and suggest math.random()."""
        for call in self.calls:
            if call.full_name == 'math.random' and len(call.args) == 2:
                arg1 = call.args[0]
                arg2 = call.args[1]
                if isinstance(arg1, Number) and arg1.n == 0 and \
                   isinstance(arg2, Number) and arg2.n == 1:
                    self.findings.append(Finding(
                        pattern_name='math_random_0_1',
                        severity='GREEN',
                        line_num=call.line,
                        message='math.random(0, 1) -> math.random()',
                        details={'node': call.node},
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_math_deg_rad(self):
        """Find math.deg(x) or math.rad(x) and suggest multiplication by constant."""
        import math as py_math
        for call in self.calls:
            if call.full_name in ('math.deg', 'math.rad') and len(call.args) == 1:
                arg = call.args[0]
                if isinstance(arg, Number):
                    continue # handled by constant folding

                x_str = node_to_string(arg)

                if call.full_name == 'math.deg':
                    # x * (180 / pi)
                    const = 180 / py_math.pi
                    msg = f'math.deg({x_str}) -> {x_str} * {const:.10g}'
                    pattern = 'math_deg_to_mult'
                else:
                    # x * (pi / 180)
                    const = py_math.pi / 180
                    msg = f'math.rad({x_str}) -> {x_str} * {const:.10g}'
                    pattern = 'math_rad_to_mult'

                self.findings.append(Finding(
                    pattern_name=pattern,
                    severity='GREEN',
                    line_num=call.line,
                    message=msg,
                    details={
                        'x_str': x_str,
                        'const': f'{const:.10g}',
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_string_rep(self):
        """Find string.rep(s, 2) and suggest s .. s."""
        for call in self.calls:
            if call.full_name == 'string.rep' and len(call.args) == 2:
                count_node = call.args[1]
                if isinstance(count_node, Number) and count_node.n == 2:
                    s_str = node_to_string(call.args[0])
                    if self._is_simple_expr(call.args[0]):
                        replacement = f'{s_str} .. {s_str}'
                        self.findings.append(Finding(
                            pattern_name='string_rep_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'string.rep({s_str}, 2) -> {replacement}',
                            details={
                                'replacement': replacement,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))

    def _analyze_math_log(self):
        """Find math.log(x, base) and suggest math.log(x) if base is e."""
        for call in self.calls:
            if call.full_name == 'math.log' and len(call.args) == 2:
                base_node = call.args[1]
                is_e = False
                is_10 = False

                # check for math.exp(1)
                if isinstance(base_node, Call):
                    _, _, full_name = self._get_call_name(base_node)
                    if full_name == 'math.exp' and len(base_node.args) == 1:
                        arg = base_node.args[0]
                        if isinstance(arg, Number) and arg.n == 1:
                            is_e = True

                # check for numeric value of e (approx)
                elif isinstance(base_node, Number):
                    if 2.71828 <= base_node.n <= 2.71829:
                        is_e = True
                    elif base_node.n == 10:
                        is_10 = True

                if is_e:
                    x_str = node_to_string(call.args[0])
                    self.findings.append(Finding(
                        pattern_name='math_log_base_e',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'math.log({x_str}, e) -> math.log({x_str})',
                        details={
                            'x_str': x_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))
                elif is_10:
                    x_str = node_to_string(call.args[0])
                    self.findings.append(Finding(
                        pattern_name='math_log_base_10',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'math.log({x_str}, 10) -> math.log10({x_str})',
                        details={
                            'x_str': x_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_math_mod(self):
        """Find math.mod(x, y) or math.fmod(x, y) and suggest x % y."""
        for call in self.calls:
            if call.full_name in ('math.mod', 'math.fmod') and len(call.args) == 2:
                x_str = node_to_string(call.args[0])
                y_str = node_to_string(call.args[1])

                pattern = 'math_mod_to_percent' if call.full_name == 'math.mod' else 'math_fmod_to_percent'
                # math.fmod has different behavior for negative numbers in standard Lua,
                # but in many STALKER mods it's used interchangeably or with positive numbers.
                # We mark fmod as YELLOW to be safe.
                severity = 'GREEN' if call.full_name == 'math.mod' else 'YELLOW'

                self.findings.append(Finding(
                    pattern_name=pattern,
                    severity=severity,
                    line_num=call.line,
                    message=f'{call.full_name}({x_str}, {y_str}) -> {x_str} % {y_str}',
                    details={
                        'x_str': x_str,
                        'y_str': y_str,
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_math_atan2(self):
        """Find math.atan2(y, 1) and suggest math.atan(y)."""
        for call in self.calls:
            if call.full_name == 'math.atan2' and len(call.args) == 2:
                arg2 = call.args[1]
                if isinstance(arg2, Number) and arg2.n == 1:
                    y_str = node_to_string(call.args[0])
                    self.findings.append(Finding(
                        pattern_name='math_atan2_to_atan',
                        severity='GREEN',
                        line_num=call.line,
                        message=f'math.atan2({y_str}, 1) -> math.atan({y_str})',
                        details={
                            'y_str': y_str,
                            'node': call.node,
                        },
                        source_line=self._get_source_line(call.line),
                    ))

    def _analyze_math_pow(self):
        """Find math.pow that can be simplified."""
        for call in self.calls:
            if call.full_name == 'math.pow' and len(call.args) == 2:
                base_node = call.args[0]
                base = node_to_string(base_node)
                exp_node = call.args[1]
                exp = self._get_const_val(exp_node)

                # check for simple cases
                if exp is not None:
                    full_match = f'math.pow({base}, {exp})'

                    if exp == 1:
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> {base}',
                            details={
                                'base': base,
                                'exponent': 1,
                                'type': 'power_1',
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue
                    elif exp == 0:
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> 1',
                            details={
                                'base': base,
                                'exponent': 0,
                                'type': 'power_0',
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue
                    elif exp == 0.5:
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> math.sqrt({base})',
                            details={
                                'base': base,
                                'exponent': exp,
                                'type': 'sqrt',
                                'is_simple': True,
                                'full_match': full_match,
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue
                    elif exp in (2, 3) and self._is_simple_expr(base_node):
                        # Suggest multiplication
                        replacement = '*'.join([base] * int(exp))
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> {replacement}',
                            details={
                                'base': base,
                                'exponent': exp,
                                'type': 'power',
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue
                    elif exp in (-1, -2, -0.5):
                        self.findings.append(Finding(
                            pattern_name='math_pow_simple',
                            severity='GREEN',
                            line_num=call.line,
                            message=f'{full_match} -> negative power simplification',
                            details={
                                'base': base,
                                'exponent': exp,
                                'type': 'power_neg',
                                'node': call.node,
                            },
                            source_line=self._get_source_line(call.line),
                        ))
                        continue

                # Fallback to base ^ exp for general cases
                self.findings.append(Finding(
                    pattern_name='math_pow_to_expo',
                    severity='GREEN',
                    line_num=call.line,
                    message=f'math.pow({base}, ...) -> {base} ^ ...',
                    details={
                        'base': base,
                        'exp_str': node_to_string(exp_node),
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _is_simple_expr(self, node: Node) -> bool:
        """Check if node is a simple expression (safe to repeat)."""
        return isinstance(node, (Name, Number, String, TrueExpr, FalseExpr))

    def _is_guaranteed_truthy(self, node: Node) -> bool:
        """Check if node is guaranteed to be truthy in Lua."""
        if isinstance(node, (Number, String, TrueExpr, Table, Function, LocalFunction)):
            return True
        # also check inferred type
        inferred = self._infer_type(node)
        if inferred in ('number', 'string', 'table', 'function'):
            return True
        return False
    
    def _count_calls_branch_aware(self, calls: List[CallInfo]) -> int:
        """
        Count function calls with branch awareness.
        
        In mutually exclusive if/elseif/else chains, only count the max calls
        in any single branch (not the sum across all branches).

        Standard count: 3 uses
        Branch-aware count: max(1, 2) = 2 uses (only one branch executes)
        """
        # group calls by if-chain (use id(node) as key since nodes aren't hashable)
        if_chains: Dict[Optional[int], Dict[int, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))
        calls_outside_if = []
        
        for call in calls:
            if call.parent_if_node is not None:
                # in an if-chain - group by (if_node_id, branch_index)
                if_chains[id(call.parent_if_node)][call.branch_index].append(call)
            else:
                # not in any if statement
                calls_outside_if.append(call)
        
        # count: calls outside if + max per if-chain
        total = len(calls_outside_if)
        
        for if_node_id, branches in if_chains.items():
            # take max calls across all branches in this if-chain
            max_in_chain = max(len(branch_calls) for branch_calls in branches.values())
            total += max_in_chain
        
        return total

    def _analyze_uncached_globals(self):
        """Find frequently used globals that should be cached."""
        # count calls by full_name, grouped by function scope
        scope_calls: Dict[Scope, Dict[str, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))

        for call in self.calls:
            # find enclosing function scope
            func_scope = self._find_function_scope(call.scope)
            if func_scope:
                scope_calls[func_scope][call.full_name].append(call)

        # check each function
        for func_scope, calls_by_name in scope_calls.items():
            globals_to_cache = {}

            for name, calls in calls_by_name.items():
                # skip if already cached or has direct replacement
                if name in DIRECT_REPLACEMENT_FUNCS:
                    continue
                if name in func_scope.cached_globals:
                    continue

                # check if it's a cacheable global
                is_bare = name in CACHEABLE_BARE_GLOBALS
                is_module_func = False

                if '.' in name:
                    module, func = name.split('.', 1)
                    if module in CACHEABLE_MODULE_FUNCS and func in CACHEABLE_MODULE_FUNCS[module]:
                        is_module_func = True

                if not is_bare and not is_module_func:
                    continue

                # threshold: configurable (default 4), hot callbacks use threshold-1
                # with --experimental, use branch-aware counting
                threshold = self.cache_threshold - 1 if func_scope.is_hot_callback else self.cache_threshold
                call_count = self._count_calls_branch_aware(calls)
                if call_count >= threshold:
                    globals_to_cache[name] = calls

            if globals_to_cache:
                # skip global scope - only cache inside actual functions
                if func_scope.name == '<global>' or func_scope.scope_type == 'global':
                    continue

                # create summary finding for this function
                example_lines = []
                for name, calls in list(globals_to_cache.items())[:5]:
                    for c in calls[:2]:
                        example_lines.append(f"L{c.line}: {name}")

                self.findings.append(Finding(
                    pattern_name='uncached_globals_summary',
                    severity='GREEN',
                    line_num=func_scope.start_line,
                    message=f'Cache {len(globals_to_cache)} globals in {func_scope.name}',
                    details={
                        'globals': {n: len(c) for n, c in globals_to_cache.items()},
                        'globals_info': globals_to_cache,  # name -> list of CallInfo with nodes
                        'function': func_scope.name,
                        'is_hot': func_scope.is_hot_callback,
                        'scope': func_scope,
                    },
                    source_line='\n'.join(example_lines),
                ))

    def _find_function_scope(self, scope: Scope) -> Optional[Scope]:
        """Find the enclosing function scope."""
        while scope:
            if scope.scope_type == 'function':
                return scope
            scope = scope.parent
        return self.global_scope

    def _analyze_repeated_calls_in_scope(self):
        """Find repeated expensive calls within function scope."""
        # expensive calls to track
        # NOTE: time_global() is NOT included because it returns different values
        # each call (current time) - caching it breaks elapsed time calculations
        # NOTE: level.object_by_id() is NOT auto-fixed because different IDs give
        # different objects, and even same IDs can change if object is destroyed
        expensive_calls = {'db.actor', 'alife', 'system_ini', 'game_ini', 'getFS',
                           'device', 'get_console', 'get_hud', 'level.name'}

        # method calls that are safe to cache (immutable object properties)
        # based on X-Ray engine source analysis:
        # - :section() returns stored NameSection member (xr_object.h:155)
        # - :id() returns stored Props.net_ID member (xr_object.h:98)
        # - :clsid() returns stored m_script_clsid member (GameObject.h:257)
        # - :story_id() returns m_story_id set once from config (xrServer_Objects_ALife.cpp:375)
        cacheable_methods = {'section', 'id', 'clsid', 'story_id'}

        # group by function scope
        scope_calls: Dict[Scope, Dict[str, List[CallInfo]]] = defaultdict(lambda: defaultdict(list))

        for call in self.calls:
            if call.full_name in expensive_calls:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    scope_calls[func_scope][call.full_name].append(call)

            # track cacheable method calls on objects (:section(), :id(), :clsid())
            if call.func in cacheable_methods and ':' in call.full_name:
                func_scope = self._find_function_scope(call.scope)
                if func_scope:
                    key = f"{call.full_name}()"
                    scope_calls[func_scope][key].append(call)

        for func_scope, calls_by_name in scope_calls.items():
            for name, calls in calls_by_name.items():
                threshold = self.cache_threshold - 1 if func_scope.is_hot_callback else self.cache_threshold
                call_count = self._count_calls_branch_aware(calls)

                if call_count >= threshold:
                    # suggest caching
                    severity = 'GREEN'

                    if name == 'db.actor':
                        suggestion = 'local actor = db.actor'
                    elif name == 'alife':
                        suggestion = 'local sim = alife()'
                    elif name == 'system_ini':
                        suggestion = 'local ini = system_ini()'
                    elif name == 'device':
                        suggestion = 'local dev = device()'
                    elif name == 'get_console':
                        suggestion = 'local console = get_console()'
                    elif name == 'get_hud':
                        suggestion = 'local hud = get_hud()'
                    elif name == 'level.name':
                        suggestion = 'local level_name = level.name()'
                    else:
                        suggestion = f'Cache {name} result'

                    self.findings.append(Finding(
                        pattern_name=f'repeated_{name.replace(".", "_").replace(":", "_")}',
                        severity=severity,
                        line_num=calls[0].line,
                        message=f'{name} called {len(calls)}x in {func_scope.name}',
                        details={
                            'count': len(calls),
                            'function': func_scope.name,
                            'is_hot': func_scope.is_hot_callback,
                            'suggestion': suggestion,
                            'lines': [c.line for c in calls],
                            'calls': calls,  # list of CallInfo with nodes
                            'scope': func_scope,
                            'original_call': name,  # preserve original like "self.object:id()"
                        },
                        source_line=suggestion,
                    ))

    def _analyze_string_concat_in_loop(self):
        """Find string concatenation patterns in loops."""
        # find self-concatenation: s = s .. x
        loop_concats: Dict[Tuple[Scope, str], List[ConcatInfo]] = defaultdict(list)

        for concat in self.concats:
            if concat.in_loop and concat.target and concat.left_var:
                if concat.target == concat.left_var:
                    # self concat: s = s .. x
                    key = (concat.scope, concat.target)
                    loop_concats[key].append(concat)

        for (scope, var), concats in loop_concats.items():
            if len(concats) >= 1:
                concat_info = concats[0]
                loop_scope = concat_info.loop_scope
                
                # check if variable is initialized to empty string before loop
                init_line = None
                is_safe = False
                
                # SAFETY: don't auto-fix nested loops (loop_depth > 1) because we can't
                # reliably determine which loop's end to place table.concat after
                if loop_scope and concat_info.loop_depth == 1:
                    # look for var = "" or var = '' IMMEDIATELY before the loop
                    # must be: within 3 lines, NOT inside any loop, and must be local declaration
                    empty_strings = ('""', "''", '[[]]')
                    for assign in self.assigns:
                        if (assign.target == var and 
                            assign.value_type == 'literal' and
                            assign.value_repr in empty_strings and
                            assign.line < loop_scope.start_line and
                            assign.line >= loop_scope.start_line - 3 and
                            not assign.in_loop and
                            assign.is_local):
                            init_line = assign.line
                            is_safe = True
                            break
                
                self.findings.append(Finding(
                    pattern_name='string_concat_in_loop',
                    severity='YELLOW',
                    line_num=concat_info.line,
                    message=f'String concat in loop: {var} = {var} .. x',
                    details={
                        'variable': var,
                        'count': len(concats),
                        'loop_depth': concat_info.loop_depth,
                        'suggestion': 'Use table.insert() + table.concat()',
                        'right_expr': concat_info.right_expr,
                        'loop_start': loop_scope.start_line if loop_scope else None,
                        'loop_end': loop_scope.end_line if loop_scope else None,
                        'init_line': init_line,
                        'is_safe': is_safe,
                        'concat_lines': [c.line for c in concats],
                    },
                    source_line=self._get_source_line(concat_info.line),
                ))

    def _analyze_debug_statements(self):
        """Find debug/logging statements."""
        for call in self.calls:
            func_name = call.func
            # exclude math.log - it's mathematical logarithm, not logging
            if call.full_name and call.full_name.startswith('math.'):
                continue
            if func_name in DEBUG_FUNCTIONS:
                self.findings.append(Finding(
                    pattern_name='debug_statement',
                    severity='DEBUG',
                    line_num=call.line,
                    message=f'Debug call: {func_name}()',
                    details={
                        'function': func_name,
                        'node': call.node,
                    },
                    source_line=self._get_source_line(call.line),
                ))

    def _analyze_global_writes(self):
        """Track global variable writes."""
        for name, line in self.global_writes:
            # skip common patterns that are intentional
            if name.startswith('_') or name.isupper():
                continue

            self.findings.append(Finding(
                pattern_name='global_write',
                severity='RED',
                line_num=line,
                message=f'Global write: {name}',
                details={
                    'variable': name,
                },
                source_line=self._get_source_line(line),
            ))

    def _analyze_nil_access(self):
        """Generate findings for potential nil access patterns."""
        for access in self.nil_accesses:
            nil_source = access.nil_source
            reason = NIL_RETURNING_FUNCTIONS.get(nil_source.source_func, 'may return nil')
            
            # determine severity based on whether it's safe to fix
            if access.is_safe_to_fix:
                severity = 'YELLOW'  # can be auto-fixed with --fix-nil
                message = (f"Potential nil access: '{access.var_name}' from {nil_source.source_func}() "
                          f"used without nil check (auto-fixable)")
            else:
                severity = 'YELLOW'  # warning only, needs manual review
                message = (f"Potential nil access: '{access.var_name}' from {nil_source.source_func}() "
                          f"used without nil check")
            
            self.findings.append(Finding(
                pattern_name='potential_nil_access',
                severity=severity,
                line_num=access.access_line,
                message=message,
                details={
                    'var_name': access.var_name,
                    'source_func': nil_source.source_func,
                    'source_call': nil_source.source_call,
                    'assign_line': nil_source.assign_line,
                    'access_call': access.access_call,
                    'access_type': access.access_type,
                    'is_safe_to_fix': access.is_safe_to_fix,
                    'is_local': nil_source.is_local,
                    'reason': reason,
                },
                source_line=self._get_source_line(access.access_line),
            ))

    def _analyze_dead_code(self):
        """Analyze for dead/unreachable code patterns."""
        if not hasattr(self, '_ast_tree') or self._ast_tree is None:
            return
        
        # Phase 1: 100% safe patterns (auto-fixable)
        self._detect_code_after_return()
        self._detect_code_after_break()
        self._detect_if_false_blocks()
        self._detect_while_false_loops()
        self._detect_unnecessary_else()
        self._detect_constant_conditions()
        
        # Phase 2: Warning patterns (not auto-fixable)
        self._detect_unused_local_vars()
        self._detect_unused_local_funcs()
        self._detect_dead_assignments()

    def _detect_dead_assignments(self):
        """Detect variables assigned a value that is never read."""
        for (scope_id, name), info in self.local_vars.items():
            if info.is_loop_var:
                continue

            for assign in info.assignments:
                if not assign.is_used and assign.node is not None:
                    # node is None for initial declaration "local x"
                    self.findings.append(Finding(
                        pattern_name='dead_assignment',
                        severity='YELLOW',
                        line_num=assign.line,
                        message=f'Value assigned to "{name}" is never read',
                        details={
                            'name': name,
                            'line': assign.line,
                        },
                        source_line=self._get_source_line(assign.line),
                    ))

    def _detect_code_after_return(self):
        """Detect unreachable code after unconditional return statements."""
        self._walk_for_dead_after_terminator(Return, 'return')

    def _detect_code_after_break(self):
        """Detect unreachable code after break statements in loops."""
        self._walk_for_dead_after_terminator(Break, 'break')

    def _walk_for_dead_after_terminator(self, terminator_type, terminator_name: str):
        """Walk AST to find dead code after terminators (return/break)."""
        
        def check_block(block_body: List[Node], scope_name: str, in_loop: bool = False):
            """Check a block for dead code after terminators."""
            if not block_body:
                return
            
            for i, stmt in enumerate(block_body):
                # check if this is a terminator
                is_terminator = isinstance(stmt, terminator_type)
                
                # for break, only count as terminator if we're in a loop
                if isinstance(stmt, Break) and not in_loop:
                    continue
                
                if is_terminator and i < len(block_body) - 1:
                    # there are statements after the terminator
                    dead_start = i + 1
                    dead_stmts = block_body[dead_start:]
                    
                    # filter out comments and semicolons
                    real_dead = [s for s in dead_stmts 
                                if not isinstance(s, (Comment, SemiColon))]
                    
                    if real_dead:
                        first_dead = real_dead[0]
                        last_dead = real_dead[-1]
                        start_line = self._get_line(first_dead)
                        end_line = self._get_end_line(last_dead) or start_line
                        
                        # get code preview
                        preview_lines = []
                        for ln in range(start_line, min(start_line + 3, end_line + 1)):
                            if 0 < ln <= len(self.source_lines):
                                preview_lines.append(self.source_lines[ln - 1].rstrip())
                        code_preview = '\n'.join(preview_lines)
                        if end_line > start_line + 2:
                            code_preview += '\n...'
                        
                        self.dead_code.append(DeadCodeInfo(
                            dead_type=f'after_{terminator_name}',
                            start_line=start_line,
                            end_line=end_line,
                            scope_name=scope_name,
                            description=f'Unreachable code after {terminator_name}',
                            is_safe_to_remove=True,
                            code_preview=code_preview,
                            node=first_dead,
                        ))
                        
                        self.findings.append(Finding(
                            pattern_name=f'dead_code_after_{terminator_name}',
                            severity='GREEN',  # safe to auto-fix
                            line_num=start_line,
                            message=f'Unreachable code after {terminator_name} statement (lines {start_line}-{end_line})',
                            details={
                                'dead_type': f'after_{terminator_name}',
                                'start_line': start_line,
                                'end_line': end_line,
                                'scope_name': scope_name,
                                'is_safe_to_remove': True,
                                'dead_stmt_count': len(real_dead),
                            },
                            source_line=self._get_source_line(start_line),
                        ))
                
                # recurse into nested structures
                if isinstance(stmt, (Function, LocalFunction, Method)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        func_name = self._get_func_name(stmt)
                        check_block(body, func_name, False)
                
                elif isinstance(stmt, If):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, in_loop)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        if isinstance(stmt.orelse, Block):
                            check_block(stmt.orelse.body, scope_name, in_loop)
                        elif isinstance(stmt.orelse, (If, ElseIf)):
                            check_block([stmt.orelse], scope_name, in_loop)
                
                elif isinstance(stmt, ElseIf):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, in_loop)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        if isinstance(stmt.orelse, Block):
                            check_block(stmt.orelse.body, scope_name, in_loop)
                        elif isinstance(stmt.orelse, (If, ElseIf)):
                            check_block([stmt.orelse], scope_name, in_loop)
                
                elif isinstance(stmt, (While, Repeat)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, True)  # now in a loop
                
                elif isinstance(stmt, (Fornum, Forin)):
                    if hasattr(stmt, 'body') and stmt.body:
                        body = stmt.body.body if isinstance(stmt.body, Block) else [stmt.body]
                        check_block(body, scope_name, True)  # now in a loop
        
        # start from the root
        if hasattr(self._ast_tree, 'body') and self._ast_tree.body:
            body = self._ast_tree.body.body if isinstance(self._ast_tree.body, Block) else [self._ast_tree.body]
            check_block(body, '<global>', False)

    def _get_func_name(self, node: Node) -> str:
        """Get function name from function node."""
        if isinstance(node, Function):
            return node_to_string(node.name) if node.name else '<anon>'
        elif isinstance(node, LocalFunction):
            return node.name.id if isinstance(node.name, Name) else '<anon>'
        elif isinstance(node, Method):
            source = node_to_string(node.source)
            method = node.name.id if isinstance(node.name, Name) else ""
            return f"{source}:{method}"
        return '<unknown>'

    def _detect_if_false_blocks(self):
        """Detect 'if false then ... end' blocks."""
        self._walk_for_false_conditions(If, 'if_false')

    def _detect_while_false_loops(self):
        """Detect 'while false do ... end' loops."""
        self._walk_for_false_conditions(While, 'while_false')

    def _walk_for_false_conditions(self, node_type, dead_type: str):
        """Walk AST to find if/while with literal false conditions."""
        
        def is_literal_false(node: Node) -> bool:
            """Check if node is literal false or nil."""
            return isinstance(node, (FalseExpr, Nil))
        
        # single flat walk - O(n) instead of O(n²)
        for node in ast.walk(self._ast_tree):
            if isinstance(node, node_type):
                if hasattr(node, 'test') and is_literal_false(node.test):
                    start_line = self._get_line(node)
                    end_line = self._get_end_line(node) or start_line
                    
                    # get code preview
                    preview_lines = []
                    for ln in range(start_line, min(start_line + 3, end_line + 1)):
                        if 0 < ln <= len(self.source_lines):
                            preview_lines.append(self.source_lines[ln - 1].rstrip())
                    code_preview = '\n'.join(preview_lines)
                    if end_line > start_line + 2:
                        code_preview += '\n...'
                    
                    type_name = 'if' if node_type == If else 'while'
                    
                    self.dead_code.append(DeadCodeInfo(
                        dead_type=dead_type,
                        start_line=start_line,
                        end_line=end_line,
                        scope_name='<unknown>',
                        description=f'{type_name} false block (never executes)',
                        is_safe_to_remove=True,
                        code_preview=code_preview,
                        node=node,
                    ))
                    
                    self.findings.append(Finding(
                        pattern_name=f'dead_code_{dead_type}',
                        severity='GREEN',  # safe to auto-fix
                        line_num=start_line,
                        message=f'Dead code: {type_name} false (lines {start_line}-{end_line})',
                        details={
                            'dead_type': dead_type,
                            'start_line': start_line,
                            'end_line': end_line,
                            'scope_name': '<unknown>',
                            'is_safe_to_remove': True,
                        },
                        source_line=self._get_source_line(start_line),
                    ))

    def _detect_unnecessary_else(self):
        """Detect unnecessary else blocks after if blocks that always return/break.
        
        Pattern:
            if condition then
                return x
            else            -- This else is unnecessary
                return y
            end
        
        Can be simplified to:
            if condition then
                return x
            end
            return y
        """
        from luaparser.astnodes import If, ElseIf, Block, Return, Break
        
        def block_always_terminates(body) -> bool:
            """Check if a block always ends with return or break."""
            if not body:
                return False
            
            # Get the body list
            if isinstance(body, Block):
                stmts = body.body if hasattr(body, 'body') else []
            elif isinstance(body, list):
                stmts = body
            else:
                return False
            
            if not stmts:
                return False
            
            # Check if last statement is return or break
            last_stmt = stmts[-1] if stmts else None
            return isinstance(last_stmt, (Return, Break))
        
        def check_if_node(node: If):
            """Check an if node for unnecessary else."""
            if not node.orelse:
                return  # No else clause
            
            # Check if the if body always terminates
            if not block_always_terminates(node.body):
                return  # If body doesn't always terminate, else is needed
            
            # Get else clause info
            orelse = node.orelse
            
            # Handle elseif chain - check if ALL branches terminate
            if isinstance(orelse, ElseIf):
                # For elseif, we need more complex analysis
                # Skip for now - just handle simple if/else
                return
            
            # Simple else block
            if isinstance(orelse, Block):
                else_start = self._get_line(orelse)
                if_start = self._get_line(node)
                
                # Find the 'else' keyword line (should be just before the else block content)
                # The else block starts at the 'else' keyword
                else_keyword_line = else_start
                
                # Look backwards from else block to find 'else' keyword
                for search_line in range(else_start, if_start, -1):
                    line_text = self._get_source_line(search_line)
                    if line_text and line_text.strip().lower() == 'else':
                        else_keyword_line = search_line
                        break
                
                self.findings.append(Finding(
                    pattern_name='unnecessary_else',
                    severity='YELLOW',  # style suggestion, not auto-fix for now
                    line_num=else_keyword_line,
                    message=f'Unnecessary else after return/break - code can be simplified',
                    details={
                        'if_line': if_start,
                        'else_line': else_keyword_line,
                        'suggestion': 'Remove else and dedent the else body',
                    },
                    source_line=self._get_source_line(else_keyword_line),
                ))
        
        def walk(node):
            """Walk AST looking for if statements."""
            if isinstance(node, If):
                check_if_node(node)
            
            # Recurse into children
            for child in iter_children(node):
                walk(child)
        
        walk(self._ast_tree)

    def _detect_constant_conditions(self):
        """Detect if/while statements with constant conditions.
        
        Patterns detected:
        - if 1 then (always true)
        - if 0 then (always false in Lua? No - 0 is truthy!)
        - if nil then (always false)
        - if "string" then (always true)
        - while true do (intentional infinite loop - skip)
        - while 1 do (always true)
        
        Note: In Lua, only nil and false are falsy. 0, "", etc are truthy.
        """
        from luaparser.astnodes import If, While, TrueExpr, FalseExpr, Nil, Number, String
        
        def is_constant_truthy(node) -> tuple:
            """Check if a node is a constant truthy/falsy value.
            Returns (is_constant, is_truthy, description)
            """
            # nil is always falsy
            if isinstance(node, Nil):
                return (True, False, 'nil')
            
            # false is always falsy
            if isinstance(node, FalseExpr):
                return (True, False, 'false')
            
            # true is always truthy (but usually intentional)
            if isinstance(node, TrueExpr):
                return (True, True, 'true')
            
            # Numbers are always truthy (including 0!)
            if isinstance(node, Number):
                val = node.n if hasattr(node, 'n') else '?'
                return (True, True, f'number {val}')
            
            # Strings are always truthy (including "")
            if isinstance(node, String):
                return (True, True, 'string literal')
            
            return (False, None, None)
        
        def check_condition(node, node_type: str, line: int):
            """Check a condition node."""
            is_const, is_truthy, desc = is_constant_truthy(node)
            
            if not is_const:
                return
            
            # Skip "while true do" - this is intentional infinite loop pattern
            if node_type == 'while' and isinstance(node, TrueExpr):
                return
            
            # Skip "if true then" in some contexts (feature flags, etc.)
            # Actually, let's report it as it's often a leftover from debugging
            
            if is_truthy:
                if node_type == 'if':
                    msg = f'Constant condition: if {desc} (always true, else branch is dead code)'
                else:
                    msg = f'Constant condition: while {desc} (infinite loop)'
                severity = 'YELLOW'
            else:
                # Always false - this is dead code
                msg = f'Constant condition: {node_type} {desc} (always false, body is dead code)'
                severity = 'GREEN'  # Can auto-remove
            
            self.findings.append(Finding(
                pattern_name='constant_condition',
                severity=severity,
                line_num=line,
                message=msg,
                details={
                    'node_type': node_type,
                    'condition_desc': desc,
                    'is_truthy': is_truthy,
                },
                source_line=self._get_source_line(line),
            ))
        
        def walk(node):
            """Walk AST looking for if/while statements."""
            if isinstance(node, If):
                line = self._get_line(node)
                check_condition(node.test, 'if', line)
            elif isinstance(node, While):
                line = self._get_line(node)
                check_condition(node.test, 'while', line)
            
            # Recurse into children
            for child in iter_children(node):
                walk(child)
        
        walk(self._ast_tree)

    def _detect_unused_local_vars(self):
        """Detect local variables that are assigned but never read (Phase 2 - warning only)."""
        for (scope_id, name), info in self.local_vars.items():
            if info.is_function:
                continue
                
            if not info.is_read:
                if info.is_loop_var:
                    # Unused ipairs value optimization
                    parent_scope = info.scope
                    if parent_scope and parent_scope.name == '<forin>':
                        loop_node = None
                        for node in ast.walk(self._ast_tree):
                            if isinstance(node, Forin) and self._get_line(node) == info.assign_line:
                                loop_node = node
                                break

                        if loop_node and len(loop_node.targets) == 2 and len(loop_node.iter) == 1:
                            v_target = loop_node.targets[1]
                            if isinstance(v_target, Name) and v_target.id == name:
                                iter_call = loop_node.iter[0]
                                if isinstance(iter_call, Call):
                                    _, _, full_name = self._get_call_name(iter_call)
                                    if full_name == 'ipairs' and len(iter_call.args) == 1:
                                        table_name = node_to_string(iter_call.args[0])
                                        k_var = node_to_string(loop_node.targets[0])
                                        is_simple_table = self._is_simple_expr(iter_call.args[0])
                                        severity = 'GREEN' if is_simple_table else 'YELLOW'

                                        self.findings.append(Finding(
                                            pattern_name='unused_ipairs_value',
                                            severity=severity,
                                            line_num=info.assign_line,
                                            message=f"Unused value '{name}' in ipairs loop -> use numeric loop for performance",
                                            details={
                                                'table': table_name,
                                                'k_var': k_var,
                                                'v_var': name,
                                                'node': iter_call,
                                                'loop_node': loop_node,
                                            },
                                            source_line=self._get_source_line(info.assign_line),
                                        ))
                                        continue

                    scope_name = info.scope.name if info.scope else '<unknown>'
                    self.findings.append(Finding(
                        pattern_name='unused_loop_variable',
                        severity='YELLOW',
                        line_num=info.assign_line,
                        message=f"Loop variable '{name}' is never used in {scope_name}",
                        details={
                            'var_name': name,
                            'assign_line': info.assign_line,
                            'scope_name': scope_name,
                        },
                        source_line=self._get_source_line(info.assign_line),
                    ))
                    continue

                if name in self.callback_registrations: continue
                scope_name = info.scope.name if info.scope else '<unknown>'
                pattern = 'unused_parameter' if info.is_param else 'unused_local_variable'
                msg_prefix = "Parameter" if info.is_param else "Local variable"

                self.findings.append(Finding(
                    pattern_name=pattern,
                    severity='YELLOW',
                    line_num=info.assign_line,
                    message=f"{msg_prefix} '{name}' is assigned but never used in {scope_name}",
                    details={
                        'var_name': name,
                        'assign_line': info.assign_line,
                        'scope_name': scope_name,
                        'is_safe_to_remove': False,
                        'is_param': info.is_param,
                    },
                    source_line=self._get_source_line(info.assign_line),
                ))

    def _detect_unused_local_funcs(self):
        """Detect local functions that are never called (Phase 2 - warning only)."""
        for (scope_id, name), info in self.local_funcs.items():
            if not info.is_read:
                if name in self.callback_registrations: continue
                if name in HOT_CALLBACKS or name in SAFE_CALLBACK_PARAMS: continue
                scope_name = info.scope.name if info.scope else '<unknown>'
                self.findings.append(Finding(
                    pattern_name='unused_local_function',
                    severity='YELLOW',
                    line_num=info.assign_line,
                    message=f"Local function '{name}' appears to be unused in {scope_name}",
                    details={
                        'func_name': name,
                        'assign_line': info.assign_line,
                        'scope_name': scope_name,
                        'is_safe_to_remove': False,
                    },
                    source_line=self._get_source_line(info.assign_line),
                ))

    def _analyze_per_frame_callbacks(self):
        """Analyze per-frame callbacks and flag them with performance info.
        
        These callbacks run every frame and deserve extra attention:
        - actor_on_update
        - npc_on_update
        - monster_on_update
        - physic_object_on_update
        - etc
        """
        # Expensive calls that should be avoided or cached in per-frame callbacks
        EXPENSIVE_CALLS = frozenset({
            'pairs', 'ipairs', 'string.find', 'string.match', 'string.gmatch',
            'string.gsub', 'string.format', 'table.sort', 'table.concat',
            'io.open', 'io.read', 'io.write', 'os.execute',
            'alife', 'alife_object', 'level.object_by_id', 'simulation_objects',
            'get_story_object', 'get_object_by_name',
        })
        
        for callback_info in self.per_frame_callbacks:
            scope = callback_info.scope
            
            # gather statistics about the callback
            calls_in_scope = [c for c in self.calls 
                            if c.scope is scope or 
                            (c.scope and c.scope.parent is scope)]
            
            # count loops
            loop_count = sum(1 for s in self.scopes 
                           if s.scope_type == 'loop' and s.parent is scope)
            
            # find expensive calls
            expensive_calls = []
            for call in calls_in_scope:
                if call.full_name in EXPENSIVE_CALLS or call.func in EXPENSIVE_CALLS:
                    expensive_calls.append(f"{call.full_name} (line {call.line})")
            
            # find uncached globals used multiple times
            global_usage = defaultdict(list)
            for call in calls_in_scope:
                if call.full_name in CACHEABLE_BARE_GLOBALS or \
                   (call.module and call.module in CACHEABLE_MODULE_FUNCS):
                    global_usage[call.full_name].append(call.line)
            
            uncached_globals = []
            for name, lines in global_usage.items():
                if len(lines) >= 2 and name not in scope.cached_globals:
                    uncached_globals.append(f"{name} ({len(lines)}x)")
            
            # build message
            issues = []
            if loop_count > 0:
                issues.append(f"{loop_count} loop(s)")
            if expensive_calls:
                issues.append(f"{len(expensive_calls)} expensive call(s)")
            if uncached_globals:
                issues.append(f"{len(uncached_globals)} uncached global(s)")
            
            # determine severity based on issues found
            if expensive_calls or loop_count > 0:
                severity = 'RED'
            elif uncached_globals:
                severity = 'YELLOW'
            else:
                severity = 'DEBUG'
            
            # always report per-frame callbacks
            message = f"Per-frame callback: {callback_info.name} (lines {callback_info.start_line}-{callback_info.end_line})"
            if issues:
                message += f" - {', '.join(issues)}"
            
            self.findings.append(Finding(
                pattern_name='per_frame_callback',
                severity=severity,
                line_num=callback_info.start_line,
                message=message,
                details={
                    'callback_name': callback_info.name,
                    'start_line': callback_info.start_line,
                    'end_line': callback_info.end_line,
                    'loop_count': loop_count,
                    'expensive_calls': expensive_calls,
                    'uncached_globals': uncached_globals,
                    'total_calls': len(calls_in_scope),
                },
                source_line=self._get_source_line(callback_info.start_line),
            ))

    # @TODO: Check distance_to() implementation in xray source code more thoroughly
    def _analyze_distance_to_comparisons(self):
        """
        Find distance_to() calls in comparisons that can use distance_to_sqr() instead.
        Should replace compared value with its square too.
        
        Pattern: pos:distance_to(target) < 10
        Optimized: pos:distance_to_sqr(target) < 100  -- (10^2, avoids sqrt)
        
        This is auto-fixable and provides performance improvement
        since distance_to() requires a square root operation.
        """
        for comp in self.distance_comparisons:
            squared_threshold = comp.threshold_value ** 2
            
            # format the squared value nicely
            if squared_threshold == int(squared_threshold):
                squared_str = str(int(squared_threshold))
            else:
                squared_str = f"{squared_threshold:.6g}"
            
            original = f"{comp.source_obj}:distance_to({comp.target_obj}) {comp.comparison_op} {comp.threshold_value}"
            optimized = f"{comp.source_obj}:distance_to_sqr({comp.target_obj}) {comp.comparison_op} {squared_str}"
            
            self.findings.append(Finding(
                pattern_name='distance_to_comparison',
                severity='GREEN',  # Auto-fixable
                line_num=comp.line,
                message=f'Use distance_to_sqr() to avoid sqrt: {original} -> {optimized}',
                details={
                    'source_obj': comp.source_obj,
                    'target_obj': comp.target_obj,
                    'comparison_op': comp.comparison_op,
                    'original_threshold': comp.threshold_value,
                    'squared_threshold': squared_threshold,
                    'squared_threshold_str': squared_str,
                    'invoke_node': comp.invoke_node,
                    'threshold_node': comp.threshold_node,
                    'full_node': comp.full_node,
                },
                source_line=self._get_source_line(comp.line),
            ))

    def _analyze_vector_allocations_in_loops(self):
        """
        Find vector() allocations inside loops.
        
        Each vector() call allocates memory that must be garbage collected.
        In loops, especially in per-frame callbacks, this can cause significant
        GC pressure and frame drops.
        
        Solution: Pre-allocate vectors at module level and reuse with :set()
        
        Example:
            -- Bad: allocates every iteration
            for i = 1, 100 do
                local pos = vector():set(x, y, z)
            end
            
            -- Good: reuse pre-allocated vector
            local temp_vec = vector()  -- at module level
            for i = 1, 100 do
                temp_vec:set(x, y, z)
            end
        
        NOT auto-fixable because it requires:
        1. Moving allocation to module/function level
        2. Understanding which vectors can be safely reused
        3. Ensuring no aliasing issues
        """
        for alloc in self.vector_allocations:
            severity = 'RED'
            
            context = ""
            if alloc.in_per_frame_callback:
                context = " in per-frame callback"
            if alloc.loop_depth > 1:
                context += f" (nested {alloc.loop_depth} loops deep)"
            
            message = f"vector() allocation in loop{context} - pre-allocate and reuse with :set()"
            
            self.findings.append(Finding(
                pattern_name='vector_alloc_in_loop',
                severity=severity,
                line_num=alloc.line,
                message=message,
                details={
                    'loop_depth': alloc.loop_depth,
                    'in_per_frame_callback': alloc.in_per_frame_callback,
                    'node': alloc.call_node,
                },
                source_line=self._get_source_line(alloc.line),
            ))


    def _get_source_line(self, line_num: int) -> str:
        """Get source line by number."""
        if 0 < line_num <= len(self.source_lines):
            return self.source_lines[line_num - 1].rstrip()
        return ""


def analyze_file(file_path: Path, cache_threshold: int = 4, experimental: bool = False) -> List[Finding]:
    """Convenience function to analyze a file."""
    analyzer = ASTAnalyzer(cache_threshold=cache_threshold, experimental=experimental)
    return analyzer.analyze_file(file_path)
