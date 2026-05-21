from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Literal


ProofNodeStatus = Literal["open", "expanded", "pruned", "complete", "failed"]


@dataclass
class ProofTreeNode:
    problem: str
    strategy: str
    claim: str
    partial_solution: str = ""
    parent_id: str | None = None
    depth: int = 0
    node_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: ProofNodeStatus = "open"
    score: float = 0.0
    verifier_score: float | None = None
    heuristic_score: float = 0.0
    critique: str = ""
    children: list["ProofTreeNode"] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_child(self, child: "ProofTreeNode") -> None:
        child.parent_id = self.node_id
        child.depth = self.depth + 1
        self.children.append(child)

    def is_terminal(self) -> bool:
        return self.status in {"pruned", "complete", "failed"}

    def path(self, index: dict[str, "ProofTreeNode"]) -> list["ProofTreeNode"]:
        nodes = [self]
        current = self
        while current.parent_id:
            current = index[current.parent_id]
            nodes.append(current)
        return list(reversed(nodes))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["children"] = [child.to_dict() for child in self.children]
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def iter_nodes(root: ProofTreeNode):
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def index_nodes(root: ProofTreeNode) -> dict[str, ProofTreeNode]:
    return {node.node_id: node for node in iter_nodes(root)}


def open_nodes(root: ProofTreeNode) -> list[ProofTreeNode]:
    return [node for node in iter_nodes(root) if node.status == "open"]


def best_nodes(root: ProofTreeNode, limit: int = 5) -> list[ProofTreeNode]:
    candidates = [node for node in iter_nodes(root) if node.depth > 0 and node.status != "pruned"]
    return sorted(candidates, key=lambda node: node.score, reverse=True)[:limit]
