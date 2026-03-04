"""
As the name of this file suggests, this is a whole-program analyzer meant for cross-file dead code detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Set, List, Tuple, Optional, Any
from collections import defaultdict
import sys
import io
from models import Finding, detect_file_encoding

from luaparser import ast
from luaparser.astnodes import (
    Node, Chunk, Block,
    Function, LocalFunction, Method,
    Assign, LocalAssign,
    Call, Invoke,
    Index, Name, String,
    Return, Break,
)
from utils import node_to_string, iter_children, set_parents


@dataclass
class SymbolDefinition:
    """Tracks where a symbol is defined."""
    name: str
    file_path: Path
    line: int
    symbol_type: str  # 'global_function', 'module_function', 'local_function', 'global_var', 'local_var'
    scope: str  # 'global', 'module', 'local'
    is_callback: bool = False
    is_class_method: bool = False


@dataclass
class SymbolUsage:
    """Tracks where a symbol is used."""
    name: str
    file_path: Path
    line: int
    usage_type: str  # 'call', 'read', 'callback_register'


@dataclass
class CrossFileAnalysis:
    """Results of whole-program analysis."""
    definitions: Dict[str, List[SymbolDefinition]] = field(default_factory=lambda: defaultdict(list))
    usages: Dict[str, List[SymbolUsage]] = field(default_factory=lambda: defaultdict(list))
    registered_callbacks: Set[str] = field(default_factory=set)
    exported_symbols: Set[str] = field(default_factory=set)  # symbols that might be used externally
    
    def is_symbol_used(self, name: str) -> bool:
        """Check if a symbol is used anywhere."""
        return name in self.usages or name in self.registered_callbacks or name in self.exported_symbols
    
    def get_unused_globals(self) -> List[SymbolDefinition]:
        """Get global symbols that appear unused."""
        unused = []
        for name, defs in self.definitions.items():
            for d in defs:
                # only check global scope - module and local are different beasts
                if d.scope == 'global' and not self.is_symbol_used(name):
                    unused.append(d)
        return unused



# Known callback names that the engine or scripts can call
KNOWN_CALLBACKS = frozenset({
    # Engine callbacks (registered via RegisterScriptCallback)
    'actor_on_update', 'actor_on_first_update', 'actor_on_before_death',
    'actor_on_item_take', 'actor_on_item_drop', 'actor_on_item_use',
    'actor_on_weapon_fired', 'actor_on_weapon_jammed', 'actor_on_weapon_reload',
    'actor_on_hud_animation_end', 'actor_on_hud_animation_play',
    'actor_on_feel_touch', 'actor_on_footstep',
    'actor_on_trade', 'actor_on_info_callback',
    'npc_on_update', 'npc_on_death_callback', 'npc_on_before_hit', 'npc_on_hit_callback',
    'npc_on_net_spawn', 'npc_on_net_destroy',
    'monster_on_update', 'monster_on_death_callback', 'monster_on_before_hit', 'monster_on_hit_callback',
    'monster_on_net_spawn', 'monster_on_net_destroy',
    'on_key_press', 'on_key_release', 'on_key_hold',
    'on_before_hit', 'on_hit',
    'physic_object_on_hit_callback',
    'save_state', 'load_state',
    'on_game_start', 'on_game_load',
    'server_entity_on_register', 'server_entity_on_unregister',
    'squad_on_npc_creation', 'squad_on_first_update', 'squad_on_update',
    'smart_terrain_on_update',
    'on_before_level_changing', 'on_level_changing',
    
    # Class methods that engine calls
    'net_spawn', 'net_destroy', 'reinit', 'reload', 
    'update', 'save', 'load', 'finalize',
    'death_callback', 'hit_callback', 'use_callback',
    'activate_scheme', 'deactivate_scheme', 'reset_scheme',
    'evaluate', 'execute',
    
    # UI callbacks
    'InitControls', 'InitCallBacks', 'OnMsgYes', 'OnMsgNo', 'OnMsgOk', 'OnMsgCancel',
    'OnKeyboard', 'OnButton_clicked', 'OnListItemClicked', 'OnListItemDbClicked',
    
    # MCM (Mod Configuration Menu) callbacks
    'on_mcm_load', 'on_option_change',
})

# Patterns that indicate a function is exported/public
EXPORT_PATTERNS = {
    '_G',           # _G.func = ...
    'rawset',       # rawset(_G, "name", func)
    'module',       # module pattern
}


class WholeProgramAnalyzer:
    """Performs whole-program analysis across multiple script files."""
    

    def get_findings(self) -> List[Tuple[Path, Finding]]:
        """Identify unused global symbols and return them as Findings."""
        unused_findings = []
        unused_symbols = self.analysis.get_unused_globals()

        for d in unused_symbols:
            try:
                encoding = detect_file_encoding(d.file_path)
                lines = d.file_path.read_text(encoding=encoding, errors='ignore').splitlines()
                source_line = lines[d.line - 1].strip() if 0 < d.line <= len(lines) else ""
            except Exception:
                source_line = ""

            finding = Finding(
                pattern_name='unused_global_symbol',
                severity='YELLOW',
                line_num=d.line,
                message=f"Global {d.symbol_type.replace('_', ' ')} '{d.name}' appears to be unused across all scripts",
                details={
                    'name': d.name,
                    'symbol_type': d.symbol_type,
                },
                source_line=source_line
            )
            unused_findings.append((d.file_path, finding))

        return unused_findings

    def __init__(self):
        self.analysis = CrossFileAnalysis()
        self.files_analyzed: Set[Path] = set()
        self.parse_errors: List[Tuple[Path, str]] = []
        self._ast_cache: Dict[Path, Tuple[Optional[Chunk], str]] = {}  # path -> (tree, source)
    
    def analyze_directory(self, directory: Path, recursive: bool = True) -> CrossFileAnalysis:
        """Analyze all .script files in a directory."""
        pattern = '**/*.script' if recursive else '*.script'
        script_files = list(directory.glob(pattern))
        return self._analyze_files_impl(script_files)
    
    def analyze_files(self, files: List[Path]) -> CrossFileAnalysis:
        """Analyze a specific list of files."""
        return self._analyze_files_impl(files)
    
    def _analyze_files_impl(self, files: List[Path]) -> CrossFileAnalysis:
        """Internal implementation that parses once and runs both passes."""
        # parse all files once and cache
        for script_path in files:
            self._ensure_parsed(script_path)
            # set parents for robust logic analysis
            cached = self._ast_cache.get(script_path)
            if cached and cached[0]:
                set_parents(cached[0])
        
        # pass 1: collect all definitions
        for script_path in files:
            cached = self._ast_cache.get(script_path)
            if cached and cached[0]:
                self.files_analyzed.add(script_path)
                self._visit_for_definitions(cached[0], script_path)
        
        # pass 2: collect all usages
        for script_path in files:
            cached = self._ast_cache.get(script_path)
            if cached and cached[0]:
                self._visit_for_usages(cached[0], script_path)
        
        # clear cache to free memory
        self._ast_cache.clear()
        
        return self.analysis
    
    def _ensure_parsed(self, file_path: Path):
        """Parse file and cache result."""
        if file_path in self._ast_cache:
            return
        
        try:
            source = file_path.read_text(encoding='utf-8', errors='ignore')
            old_stderr = sys.stderr
            sys.stderr = io.StringIO()
            try:
                tree = ast.parse(source)
            finally:
                sys.stderr = old_stderr
            self._ast_cache[file_path] = (tree, source)
        except Exception as e:
            self.parse_errors.append((file_path, str(e)))
            self._ast_cache[file_path] = (None, "")
    
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
    
    def _node_to_string(self, node: Node) -> str:
        """Convert AST node to string representation."""
        if isinstance(node, Name):
            return node.id
        if isinstance(node, Index):
            return self._format_index(node)
        if isinstance(node, String):
            return self._format_string(node)
        return ""

    def _format_string(self, node: String) -> str:
        """Helper to format String nodes."""
        s = node.s
        if isinstance(s, bytes):
            s = s.decode('utf-8', errors='replace')
        return s

    def _format_index(self, node: Index) -> str:
        """Helper to format Index nodes."""
        value = self._node_to_string(node.value)
        idx = self._node_to_string(node.idx)
        idx_token = getattr(node.idx, 'first_token', None)
        if idx_token is not None and str(idx_token) != 'None':
            return f"{value}[{idx}]"
        return f"{value}.{idx}"

    def _visit_for_definitions(self, node: Node, file_path: Path):
        """Visit AST to collect definitions."""
        if node is None:
            return

        # Optimization: use ast.walk instead of recursive visits
        for current_node in ast.walk(node):
            if isinstance(current_node, Function):
                self._define_function(current_node, file_path)
            elif isinstance(current_node, LocalFunction):
                self._define_local_function(current_node, file_path)
            elif isinstance(current_node, Method):
                self._define_method(current_node, file_path)
            elif isinstance(current_node, Assign):
                self._define_assign(current_node, file_path)



    def _define_function(self, node: Function, file_path: Path):
        """Handle Function definition."""
        if isinstance(node.name, Name):
            name = node.name.id
            line = self._get_line(node)
            is_callback = name in KNOWN_CALLBACKS

            self.analysis.definitions[name].append(SymbolDefinition(
                name=name,
                file_path=file_path,
                line=line,
                symbol_type='global_function',
                scope='global',
                is_callback=is_callback,
            ))

            if is_callback:
                self.analysis.registered_callbacks.add(name)

        elif isinstance(node.name, Index):
            full_name = self._node_to_string(node.name)
            line = self._get_line(node)

            self.analysis.definitions[full_name].append(SymbolDefinition(
                name=full_name,
                file_path=file_path,
                line=line,
                symbol_type='module_function',
                scope='module',
            ))
            self.analysis.exported_symbols.add(full_name)

    def _define_local_function(self, node: LocalFunction, file_path: Path):
        """Handle LocalFunction definition."""
        if isinstance(node.name, Name):
            name = node.name.id
            line = self._get_line(node)
            is_callback = name in KNOWN_CALLBACKS

            self.analysis.definitions[f"local:{file_path.stem}:{name}"].append(SymbolDefinition(
                name=name,
                file_path=file_path,
                line=line,
                symbol_type='local_function',
                scope='local',
                is_callback=is_callback,
            ))

    def _define_method(self, node: Method, file_path: Path):
        """Handle Method definition."""
        source = self._node_to_string(node.source)
        method = node.name.id if isinstance(node.name, Name) else ""
        full_name = f"{source}:{method}"
        line = self._get_line(node)
        is_class_method = method in KNOWN_CALLBACKS

        self.analysis.definitions[full_name].append(SymbolDefinition(
            name=full_name,
            file_path=file_path,
            line=line,
            symbol_type='method',
            scope='module',
            is_class_method=is_class_method,
        ))

        if is_class_method:
            self.analysis.exported_symbols.add(full_name)

    def _define_assign(self, node: Assign, file_path: Path):
        """Handle Assignment for global/module definitions."""
        for target in node.targets:
            if isinstance(target, Name):
                name = target.id
                line = self._get_line(node)

                if node.values and len(node.values) == 1:
                    val = node.values[0]
                    if isinstance(val, Function):
                        is_callback = name in KNOWN_CALLBACKS
                        self.analysis.definitions[name].append(SymbolDefinition(
                            name=name,
                            file_path=file_path,
                            line=line,
                            symbol_type='global_function',
                            scope='global',
                            is_callback=is_callback,
                        ))
                        if is_callback:
                            self.analysis.registered_callbacks.add(name)
                    else:
                        self.analysis.definitions[name].append(SymbolDefinition(
                            name=name,
                            file_path=file_path,
                            line=line,
                            symbol_type='global_var',
                            scope='global',
                        ))
            elif isinstance(target, Index):
                full_name = self._node_to_string(target)
                line = self._get_line(node)

                self.analysis.definitions[full_name].append(SymbolDefinition(
                    name=full_name,
                    file_path=file_path,
                    line=line,
                    symbol_type='module_var',
                    scope='module',
                ))
                self.analysis.exported_symbols.add(full_name)
    
    def _visit_for_usages(self, node: Optional[Node], file_path: Path):
        """Visit AST to collect usages."""
        if node is None:
            return

        # Optimization: use ast.walk instead of recursive visits
        # to avoid O(N^2) complexity with child visits
        for current_node in ast.walk(node):
            if isinstance(current_node, (Function, LocalFunction, Method, Assign, LocalAssign)):
                continue

            if isinstance(current_node, Call):
                self._usage_call(current_node, file_path)
            elif isinstance(current_node, Invoke):
                self._usage_invoke(current_node, file_path)
            elif isinstance(current_node, Name):
                self._usage_name(current_node, file_path)
            elif isinstance(current_node, Index):
                self._usage_index(current_node, file_path)

    def _usage_call(self, node: Call, file_path: Path):
        """Handle Call usage."""
        func_name = self._node_to_string(node.func)
        line = self._get_line(node)

        if func_name:
            self.analysis.usages[func_name].append(SymbolUsage(
                name=func_name,
                file_path=file_path,
                line=line,
                usage_type='call',
            ))

        if func_name == 'RegisterScriptCallback' and len(node.args) >= 2:
            callback_name = self._node_to_string(node.args[0])
            callback_func = self._node_to_string(node.args[1])
            if callback_name:
                self.analysis.registered_callbacks.add(callback_name)
            if callback_func:
                self.analysis.registered_callbacks.add(callback_func)
                self.analysis.usages[callback_func].append(SymbolUsage(
                    name=callback_func,
                    file_path=file_path,
                    line=line,
                    usage_type='callback_register',
                ))

    def _usage_invoke(self, node: Invoke, file_path: Path):
        """Handle Invoke usage."""
        obj_name = self._node_to_string(node.source)
        line = self._get_line(node)
        if obj_name:
            self.analysis.usages[obj_name].append(SymbolUsage(
                name=obj_name,
                file_path=file_path,
                line=line,
                usage_type='read',
            ))

    def _usage_name(self, node: Name, file_path: Path):
        """Handle Name usage."""
        # Only count as usage if it's not the name in a function/method definition
        # or assignment target, which are already handled by _define_*.
        # Also skip if it's the function being called (handled by _usage_call)
        parent = getattr(node, 'parent', None)
        if isinstance(parent, (Function, LocalFunction, Method)):
            if node is getattr(parent, 'name', None):
                return
            args = getattr(parent, 'args', [])
            if isinstance(args, list) and node in args:
                return
        elif isinstance(parent, (Assign, LocalAssign)):
            targets = getattr(parent, 'targets', [])
            if isinstance(targets, list) and node in targets:
                return
        elif isinstance(parent, Call):
            if node is getattr(parent, 'func', None):
                return
        
        name = node.id
        line = self._get_line(node)
        self.analysis.usages[name].append(SymbolUsage(
            name=name,
            file_path=file_path,
            line=line,
            usage_type='read',
        ))

    def _usage_index(self, node: Index, file_path: Path):
        """Handle Index usage."""
        full_name = self._node_to_string(node)
        line = self._get_line(node)
        if full_name:
            self.analysis.usages[full_name].append(SymbolUsage(
                name=full_name,
                file_path=file_path,
                line=line,
                usage_type='read',
            ))


def analyze_mods_directory(mods_path: Path) -> CrossFileAnalysis:
    """Convenience function to analyze entire mods directory."""
    analyzer = WholeProgramAnalyzer()
    
    # Find all gamedata/scripts directories
    script_dirs = list(mods_path.glob('*/gamedata/scripts'))
    
    all_scripts = []
    for script_dir in script_dirs:
        all_scripts.extend(script_dir.glob('*.script'))
    
    return analyzer.analyze_files(all_scripts)
