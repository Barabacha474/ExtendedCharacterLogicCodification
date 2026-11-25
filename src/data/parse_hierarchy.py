"""Parse Fandom_Codified_Profiles `code` strings into flat condition tables.

Each `code` is a Python source string defining parse_by_scene(scene).
We extract every check_scene(...) call with its parent chain.
"""
from __future__ import annotations
import ast
import json
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class Cond:
    block_id: int
    cond_id: str
    question: str
    var: Optional[str] = None
    parents: list[dict] = field(default_factory=list)  # [{cond_id, required(bool)}]
    depth: int = 0
    lineno: int = 0


def _call_question(node: ast.AST) -> Optional[str]:
    """Return the question string if node is check_scene(scene, "<q>")."""
    if not isinstance(node, ast.Call):
        return None
    f = node.func
    name = None
    if isinstance(f, ast.Name):
        name = f.id
    elif isinstance(f, ast.Attribute):
        name = f.attr
    if name != "check_scene":
        return None
    if len(node.args) < 2:
        return None
    a = node.args[1]
    if isinstance(a, ast.Constant) and isinstance(a.value, str):
        return a.value
    return None


def _literal_value(node: ast.AST):
    if isinstance(node, ast.Constant):
        return node.value
    return object()  # sentinel: not a literal


def _resolve_test(test: ast.AST, var_map: dict[str, Cond]):
    """
    Returns one of:
      ("new", question, required)
      ("ref", Cond, required)
      (None, None, None)
    """
    q = _call_question(test)
    if q is not None:
        return ("new", q, True)

    if isinstance(test, ast.Name) and test.id in var_map:
        return ("ref", var_map[test.id], True)

    if isinstance(test, ast.Compare) and isinstance(test.left, ast.Name):
        v = test.left.id
        if v in var_map and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Is, ast.Eq)):
            val = _literal_value(test.comparators[0])
            if isinstance(val, bool) or val is None:
                return ("ref", var_map[v], val)
    return (None, None, None)


def parse_block(block_id: int, code: str) -> list[Cond]:
    tree = ast.parse(code)
    conds: list[Cond] = []
    var_map: dict[str, Cond] = {}

    def new_cond(question: str, parents, depth: int, lineno: int, var=None) -> Cond:
        c = Cond(
            block_id=block_id,
            cond_id=f"b{block_id}_c{len(conds)}",
            question=question,
            var=var,
            parents=[dict(p) for p in parents],
            depth=depth,
            lineno=lineno,
        )
        conds.append(c)
        return c

    def visit(stmts, parents, depth):
        for st in stmts:
            if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                q = _call_question(st.value)
                if q is not None:
                    c = new_cond(q, parents, depth, st.lineno, var=st.targets[0].id)
                    var_map[st.targets[0].id] = c
                    continue
                visit([ast.Expr(value=st.value)], parents, depth) if False else None
                continue

            if isinstance(st, ast.If):
                kind, payload, required = _resolve_test(st.test, var_map)
                if kind == "new":
                    c = new_cond(payload, parents, depth, st.lineno)
                    visit(st.body, parents + [{"cond_id": c.cond_id, "required": True}], depth + 1)
                    # orelse: elif or else
                    if st.orelse:
                        if len(st.orelse) == 1 and isinstance(st.orelse[0], ast.If):
                            # elif: treat as chained If with parent required False
                            visit([st.orelse[0]], parents + [{"cond_id": c.cond_id, "required": False}], depth + 1)
                        else:
                            visit(st.orelse, parents + [{"cond_id": c.cond_id, "required": False}], depth + 1)
                elif kind == "ref":
                    parent_entry = {"cond_id": payload.cond_id, "required": bool(required) if required is not None else True}
                    visit(st.body, parents + [parent_entry], depth + 1)
                    if st.orelse:
                        if len(st.orelse) == 1 and isinstance(st.orelse[0], ast.If):
                            visit([st.orelse[0]], parents + [{"cond_id": payload.cond_id, "required": False}], depth + 1)
                        else:
                            visit(st.orelse, parents + [{"cond_id": payload.cond_id, "required": False}], depth + 1)
                else:
                    # not a check_scene test; recurse without adding parent
                    visit(st.body, parents, depth)
                    visit(st.orelse, parents, depth)
                continue

            # recurse into other compound statements
            for child in ast.iter_child_nodes(st):
                if isinstance(child, ast.stmt):
                    visit([child], parents, depth)

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "parse_by_scene":
            visit(node.body, [], 0)
            break
    return conds


def parse_profiles(df) -> list[Cond]:
    out: list[Cond] = []
    for i, code in enumerate(df["code"].tolist()):
        try:
            out.extend(parse_block(i, code))
        except SyntaxError as e:
            print(f"[WARN] block {i} syntax error: {e}")
    return out


if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("data/raw/Fandom_Codified_Profiles/data/train-00000-of-00001.parquet")
    k = df[df["character"] == "Edward"].reset_index(drop=True)
    conds = parse_profiles(k)
    print(f"blocks: {len(k)}  conditions: {len(conds)}")
    from collections import Counter
    print("depth distribution:", dict(Counter(c.depth for c in conds)))
    print("conditions with parents:", sum(1 for c in conds if c.parents))
    # dump sample
    for c in conds[:5]:
        print(asdict(c))