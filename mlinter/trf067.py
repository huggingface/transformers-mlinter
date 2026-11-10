# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRF067: a sub-model call must not capture every hidden state when only some of them are read."""

import ast
from collections.abc import Iterator
from pathlib import Path

from ._helpers import Violation, _decorator_names, _has_rule_suppression, call_leaf_name


RULE_ID = ""  # Set by discovery

FLAG = "output_hidden_states"
# Backbones under this decorator return a filtered subset of the stages they compute, so capturing every
# hidden state is how they work, not a leak.
FILTERING_DECORATOR = "filter_output_hidden_states"


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
Position = tuple[int, int]
Comprehension = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
# Putting the tuple in one of these hands it on whole: to an output, a caller, another module.
# Statements after which the rest of their block never runs.
EXITS = (ast.Return, ast.Raise, ast.Break, ast.Continue)
CONTAINERS = (ast.Tuple, ast.List, ast.Set, ast.Dict, ast.keyword, ast.Return, ast.Yield, ast.Starred)


SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _own_nodes(function_node: FunctionNode) -> Iterator[ast.AST]:
    """Every node in the function body, without descending into nested functions, lambdas or classes."""
    stack: list[ast.AST] = [node for node in function_node.body if not isinstance(node, SCOPES)]
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, SCOPES):
                stack.append(child)


def _position(node: ast.AST) -> Position:
    return node.lineno, node.col_offset


def _is_true(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and node.value is True


def _is_flag(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and node.value == FLAG


def _dict_forces(node: ast.AST | None) -> bool:
    """Whether a dict literal or `dict(...)` call sets the flag to `True`."""
    if isinstance(node, ast.Dict):
        return any(key is not None and _is_flag(key) and _is_true(value) for key, value in zip(node.keys, node.values))
    if isinstance(node, ast.Call) and call_leaf_name(node) == "dict":
        return any(keyword.arg == FLAG and _is_true(keyword.value) for keyword in node.keywords)
    return False


class _Function:
    """Name bindings and statement structure of one function body."""

    def __init__(self, function_node: FunctionNode):
        self.nodes = list(_own_nodes(function_node))
        self.parents: dict[int, ast.AST] = {id(statement): function_node for statement in function_node.body}
        for node in self.nodes:
            for child in ast.iter_child_nodes(node):
                self.parents[id(child)] = node
        # name -> [(position the store takes effect, store node)]
        self.stores: dict[str, list[tuple[Position, ast.Name]]] = {}
        for node in self.nodes:
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and not self._comprehension_of(node):
                self.stores.setdefault(node.id, []).append((self._takes_effect(node), node))

    def parent(self, node: ast.AST) -> ast.AST | None:
        return self.parents.get(id(node))

    def statement_of(self, node: ast.AST) -> ast.stmt | None:
        current: ast.AST | None = node
        while current is not None and not isinstance(current, ast.stmt):
            current = self.parent(current)
        return current

    def _comprehension_of(self, target: ast.Name) -> Comprehension | None:
        """The comprehension whose `for` binds `target`, if any."""
        node: ast.AST | None = target
        while node is not None:
            parent = self.parent(node)
            if isinstance(parent, ast.comprehension) and parent.target is node:
                return self.parent(parent)  # ty: ignore[invalid-return-type]
            if not isinstance(parent, (ast.Tuple, ast.List, ast.Starred)):
                return None
            node = parent
        return None

    def _takes_effect(self, store: ast.Name) -> Position:
        # In `hidden_states = hidden_states[-2]` the right-hand side is read before the name is rebound.
        statement = self.statement_of(store)
        if isinstance(statement, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            return statement.end_lineno or statement.lineno, statement.end_col_offset or 0
        return _position(store)

    def binding_of(self, load: ast.Name) -> ast.Name | None:
        """The store a load reads: its comprehension's target, else the latest store before it."""
        node: ast.AST | None = load
        while node is not None:
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
                for generator in node.generators:
                    for target in ast.walk(generator.target):
                        if isinstance(target, ast.Name) and target.id == load.id:
                            return target
            node = self.parent(node)
        position = _position(load)
        earlier = [(effect, store) for effect, store in self.stores.get(load.id, []) if effect < position]
        return max(earlier, key=lambda item: item[0])[1] if earlier else None

    def loads_of(self, stores: list[ast.Name]) -> list[ast.Name]:
        store_ids = {id(store) for store in stores}
        return [
            node
            for node in self.nodes
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in {store.id for store in stores}
            and id(self.binding_of(node)) in store_ids
        ]

    def reaches(self, statement: ast.stmt, node: ast.AST) -> bool:
        """Whether `node` can run after `statement` on some path that doesn't leave the function first.

        Walks out from `statement` one block at a time. A `return`, `raise`, `break` or `continue` later in a block
        cuts the path; a `node` in a sibling branch (the `else` of the `if` holding `statement`) is never reached.
        """
        current: ast.AST = statement
        while True:
            parent = self.parent(current)
            if parent is None:
                return False
            block = next(
                (
                    value
                    for _, value in ast.iter_fields(parent)
                    if isinstance(value, list) and any(item is current for item in value)
                ),
                None,
            )
            if block is None:
                return False
            index = next(i for i, item in enumerate(block) if item is current)
            for later in block[index + 1 :]:
                if self._contains(later, node):
                    return True
                if isinstance(later, EXITS):
                    return False
            # Leave this block: continue after the statement that holds it.
            while parent is not None and not isinstance(parent, ast.stmt):
                parent = self.parent(parent)
            if parent is None or isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return False
            current = parent

    def _contains(self, statement: ast.stmt, node: ast.AST) -> bool:
        ancestor: ast.AST | None = node
        while ancestor is not None:
            if ancestor is statement:
                return True
            ancestor = self.parent(ancestor)
        return False


def _reads_name(value: ast.AST | None, name: str) -> bool:
    """Whether a rebind is built from the dict it replaces, e.g. `kwargs = {k: v for k, v in kwargs.items() if ...}`,
    which keeps whatever the flag already was."""
    return value is not None and any(
        isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, ast.Load) for node in ast.walk(value)
    )


def _flag_writes(function: _Function) -> list[tuple[str, bool, ast.stmt]]:
    """Every statement that sets or clears the flag in a kwargs dict, as (dict name, forces it on, statement)."""
    writes: list[tuple[str, bool, ast.stmt]] = []
    for node in function.nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and _is_flag(target.slice):
                    writes.append((target.value.id, _is_true(node.value), node))
                elif isinstance(target, ast.Name) and not _reads_name(node.value, target.id):
                    writes.append((target.id, _dict_forces(node.value), node))
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and not _reads_name(node.value, node.target.id)
        ):
            writes.append((node.target.id, _dict_forces(node.value), node))
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and _is_flag(target.slice):
                    writes.append((target.value.id, False, node))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and isinstance(node.value.func.value, ast.Name)
        ):
            call, name = node.value, node.value.func.value.id
            method = node.value.func.attr
            if method == "update":
                for keyword in call.keywords:
                    if keyword.arg == FLAG:
                        writes.append((name, _is_true(keyword.value), node))
                for argument in call.args:
                    if isinstance(argument, ast.Dict) and any(
                        key is not None and _is_flag(key) for key in argument.keys
                    ):
                        writes.append((name, _dict_forces(argument), node))
            elif method in ("setdefault", "pop") and call.args and _is_flag(call.args[0]):
                forces = method == "setdefault" and len(call.args) > 1 and _is_true(call.args[1])
                writes.append((name, forces, node))
    return writes


def _forcing_site(call: ast.Call, function: _Function, writes: list[tuple[str, bool, ast.stmt]]) -> ast.AST | None:
    """What switches every hidden state on for `call`: its own keyword, or the kwargs write it unpacks."""
    for keyword in call.keywords:
        if keyword.arg == FLAG:
            return keyword if _is_true(keyword.value) else None
    for keyword in call.keywords:
        if keyword.arg is None and isinstance(keyword.value, ast.Name):
            # The latest write that can reach the call decides. A conditional force still leaks on its path; one in a
            # branch that returns first, or in a sibling branch, does not reach the call.
            reaching = [
                (statement, forces)
                for name, forces, statement in writes
                if name == keyword.value.id and function.reaches(statement, call)
            ]
            if reaching:
                statement, forces = max(reaching, key=lambda item: _position(item[0]))
                if forces:
                    return statement
    return None


class _Uses:
    def __init__(self) -> None:
        self.indexed = False
        self.whole = False


def _is_loop_counter(index: ast.AST, function: _Function) -> bool:
    """Whether an index walks every position: the variable of `range(len(...))` or `enumerate`'s counter."""
    if not isinstance(index, ast.Name):
        return False
    target = function.binding_of(index)
    if target is None:
        return False
    parent = function.parent(target)
    counter_of_tuple = isinstance(parent, ast.Tuple) and bool(parent.elts) and parent.elts[0] is target
    loop = function.parent(parent) if counter_of_tuple else parent
    if not isinstance(loop, (ast.For, ast.AsyncFor, ast.comprehension)):
        return False
    iterator = loop.iter
    if not isinstance(iterator, ast.Call):
        return False
    if call_leaf_name(iterator) == "enumerate":
        return counter_of_tuple
    if call_leaf_name(iterator) == "range" and not counter_of_tuple:
        return any(isinstance(argument, ast.Call) and call_leaf_name(argument) == "len" for argument in iterator.args)
    return False


def _classify(node: ast.AST, function: _Function, uses: _Uses) -> None:
    """Record how one reference to the hidden-states tuple is used."""
    parent = function.parent(node)
    if isinstance(parent, (ast.IfExp, ast.BoolOp, ast.NamedExpr)) and getattr(parent, "test", None) is not node:
        # `outputs.hidden_states if output_hidden_states else None` is used the way the whole expression is.
        _classify(parent, function, uses)
    elif isinstance(parent, ast.Subscript) and parent.value is node:
        if isinstance(parent.slice, ast.Slice) or _is_loop_counter(parent.slice, function):
            uses.whole = True
        else:
            uses.indexed = True
    elif isinstance(parent, (ast.For, ast.AsyncFor, ast.comprehension)) and parent.iter is node:
        uses.whole = True
    elif isinstance(parent, CONTAINERS):
        uses.whole = True
    elif isinstance(parent, ast.Call) and node in parent.args and call_leaf_name(parent) != "len":
        # torch.stack(...), torch.cat(...), zip(...), a weighted-sum helper: each consumes the whole tuple.
        uses.whole = True
    elif isinstance(parent, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and parent.value is node:
        # Stored on an attribute or under another name: it may be read anywhere from there.
        uses.whole = True
    # Anything else -- `len(...)`, an `is None` check -- neither reads a layer nor needs all of them.


def _assigned_names(node: ast.AST, function: _Function) -> list[ast.Name]:
    parent = function.parent(node)
    if isinstance(parent, ast.Assign) and parent.value is node:
        return [target for target in parent.targets if isinstance(target, ast.Name)]
    if isinstance(parent, ast.AnnAssign) and parent.value is node and isinstance(parent.target, ast.Name):
        return [parent.target]
    return []


def _hidden_states_uses(call: ast.Call, function: _Function, uses: _Uses) -> None:
    results = _assigned_names(call, function)
    result_loads = {id(load) for load in function.loads_of(results)} if results else set()
    for node in function.nodes:
        if not (isinstance(node, ast.Attribute) and node.attr == "hidden_states"):
            continue
        if node.value is not call and id(node.value) not in result_loads:
            continue
        aliases = _assigned_names(node, function)
        if aliases:
            # `hidden_states = outputs.hidden_states`: follow the tuple until the name is rebound.
            for load in function.loads_of(aliases):
                _classify(load, function, uses)
        else:
            _classify(node, function, uses)


def _check_function(function_node: FunctionNode, file_path: Path, source_lines: list[str]) -> list[Violation]:
    function = _Function(function_node)
    writes = _flag_writes(function)
    # Calls grouped by what forces their capture: the suggested fix applies to all of them at once.
    groups: dict[int, tuple[ast.AST, _Uses]] = {}
    for call in function.nodes:
        if not isinstance(call, ast.Call):
            continue
        site = _forcing_site(call, function, writes)
        if site is None:
            continue
        _, uses = groups.setdefault(id(site), (site, _Uses()))
        _hidden_states_uses(call, function, uses)

    violations: list[Violation] = []
    for site, uses in groups.values():
        if not uses.indexed or uses.whole:
            continue
        line = site.value.lineno if isinstance(site, ast.keyword) else site.lineno
        statement = function.statement_of(site)
        if _has_rule_suppression(source_lines, RULE_ID, line) or (
            statement is not None and _has_rule_suppression(source_lines, RULE_ID, statement.lineno)
        ):
            continue
        violations.append(
            Violation(
                file_path=file_path,
                line_number=line,
                message=(
                    f"{RULE_ID}: `{FLAG}=True` keeps every layer's hidden states alive, but "
                    f"`{function_node.name}` only reads some of them by index. Capture just those with "
                    "`@select_hidden_states(...)`, or pass the index as `output_hidden_states=<index>`."
                ),
            )
        )
    return violations


def check(tree: ast.Module, file_path: Path, source_lines: list[str]) -> list[Violation]:
    if not file_path.name.startswith(("modeling_", "modular_")):
        return []
    if not any(FLAG in line for line in source_lines):
        return []

    violations: list[Violation] = []
    for function_node in ast.walk(tree):
        if not isinstance(function_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body_lines = source_lines[function_node.lineno - 1 : function_node.end_lineno]
        if not any(FLAG in line for line in body_lines):
            continue
        if FILTERING_DECORATOR in _decorator_names(function_node):
            continue
        violations.extend(_check_function(function_node, file_path, source_lines))
    return violations
