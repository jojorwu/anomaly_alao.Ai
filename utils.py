from typing import Optional, List, Tuple, Any, Dict
from luaparser.astnodes import (
    Node, Name, Number, String, TrueExpr, FalseExpr, Nil, Index,
    Call, Invoke, ULengthOP, UMinusOp, ULNotOp, UBNotOp,
    Concat, OrLoOp, AndLoOp, AddOp, SubOp, MultOp, FloatDivOp,
    ModOp, ExpoOp, EqToOp, NotEqToOp, LessThanOp, GreaterThanOp,
    LessOrEqThanOp, GreaterOrEqThanOp, Table
)

def node_to_string(node: Node) -> str:
    """Convert an AST node to its string representation."""
    if isinstance(node, Name):
        return node.id
    if isinstance(node, Number):
        return str(node.n)
    if isinstance(node, String):
        return format_string(node)
    if isinstance(node, (TrueExpr,)):
        return "true"
    if isinstance(node, (FalseExpr,)):
        return "false"
    if isinstance(node, (Nil,)):
        return "nil"
    if isinstance(node, Index):
        return format_index(node)
    if isinstance(node, (Call, Invoke)):
        return format_call(node)
    if isinstance(node, (ULengthOP, UMinusOp, ULNotOp, UBNotOp)):
        return format_unary_op(node)
    if isinstance(node, (Concat, OrLoOp, AndLoOp, AddOp, SubOp, MultOp, FloatDivOp, ModOp, ExpoOp,
                        EqToOp, NotEqToOp, LessThanOp, GreaterThanOp, LessOrEqThanOp, GreaterOrEqThanOp)):
        return format_binary_op(node)
    if isinstance(node, Table):
        return "{...}"
    return f"<{type(node).__name__}>"

def format_string(node: String) -> str:
    """Helper to format String nodes with proper escaping."""
    s = node.s
    if isinstance(s, bytes):
        s = s.decode('utf-8', errors='replace')
    # Escape special characters for Lua string literal
    escaped = s.replace('\\', '\\\\')
    escaped = escaped.replace('\a', '\\a').replace('\b', '\\b').replace('\f', '\\f')
    escaped = escaped.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    escaped = escaped.replace('\v', '\\v').replace('\0', '\\0')
    if '"' in escaped and "'" not in escaped:
        escaped = escaped.replace("'", "\\'")
        return f"'{escaped}'"
    escaped = escaped.replace('"', '\\"')
    return f'"{escaped}"'

def format_index(node: Index) -> str:
    """Helper to format Index nodes."""
    value = node_to_string(node.value)
    idx = node_to_string(node.idx)
    idx_token = getattr(node.idx, 'first_token', None)
    if idx_token is not None and str(idx_token) != 'None':
        return f"{value}[{idx}]"
    return f"{value}.{idx}"

def format_call(node: Any) -> str:
    """Helper to format Call and Invoke nodes."""
    if isinstance(node, Call):
        func = node_to_string(node.func)
        args = ", ".join(node_to_string(a) for a in node.args)
        return f"{func}({args})"
    if isinstance(node, Invoke):
        source = node_to_string(node.source)
        func = node_to_string(node.func)
        args = ", ".join(node_to_string(a) for a in node.args)
        return f"{source}:{func}({args})"
    return ""

def format_unary_op(node: Any) -> str:
    """Helper to format unary operator nodes."""
    operand = node_to_string(node.operand)
    if isinstance(node, ULengthOP): return f"#{operand}"
    if isinstance(node, UMinusOp): return f"-{operand}"
    if isinstance(node, ULNotOp): return f"not {operand}"
    if isinstance(node, UBNotOp): return f"~{operand}"
    return ""

def format_binary_op(node: Any) -> str:
    """Helper to format binary operator nodes."""
    left = node_to_string(node.left)
    right = node_to_string(node.right)
    ops = {
        Concat: "..", OrLoOp: "or", AndLoOp: "and", AddOp: "+", SubOp: "-",
        MultOp: "*", FloatDivOp: "/", ModOp: "%", ExpoOp: "^", EqToOp: "==",
        NotEqToOp: "~=", LessThanOp: "<", GreaterThanOp: ">", LessOrEqThanOp: "<=",
        GreaterOrEqThanOp: ">="
    }
    op_str = ops.get(type(node), "?")
    if isinstance(node, (OrLoOp, AndLoOp)):
        return f"({left} {op_str} {right})"
    return f"{left} {op_str} {right}"

def iter_children(node: Node):
    """Iterate over all child nodes of an AST node."""
    if node is None:
        return

    # For AST nodes, iterate over known child attributes
    # EXCLUDING 'parent' to avoid infinite loops
    child_attrs = [
        'body', 'test', 'orelse', 'targets', 'values', 'iter',
        'func', 'args', 'value', 'idx', 'key', 'left', 'right',
        'operand', 'fields', 'keys', 'source', 'step', 'start', 'stop',
        'name' # Important for functions!
    ]

    for attr in child_attrs:
        child = getattr(node, attr, None)
        if child is not None:
            if isinstance(child, list):
                for item in child:
                    if isinstance(item, Node):
                        yield item
            elif isinstance(child, Node):
                yield child

def get_parent_map(root_node: Node) -> Dict[int, Node]:
    """Iteratively build a map of node id -> parent node."""
    parent_map = {}
    if root_node is None: return parent_map
    stack = [root_node]
    visited = set()
    while stack:
        node = stack.pop()
        if id(node) in visited:
            continue
        visited.add(id(node))
        for child in iter_children(node):
            if isinstance(child, Node):
                parent_map[id(child)] = node
                stack.append(child)
    return parent_map
