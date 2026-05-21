from __future__ import annotations

import json

from src.proof_tree import ProofTreeNode, iter_nodes
from src.retrieval_context import truncate


def tree_to_text(root: ProofTreeNode, max_chars: int = 220) -> str:
    lines = []
    for node in iter_nodes(root):
        indent = "  " * node.depth
        label = f"{node.strategy} | score={node.score:.2f} | {node.status}"
        lines.append(f"{indent}- {node.node_id}: {label}")
        if node.claim:
            lines.append(f"{indent}  claim: {truncate(node.claim, max_chars)}")
    return "\n".join(lines)


def tree_to_json(root: ProofTreeNode) -> str:
    return json.dumps(root.to_dict(), indent=2, ensure_ascii=False)


def tree_to_mermaid(root: ProofTreeNode, max_label_chars: int = 80) -> str:
    lines = ["graph TD"]
    for node in iter_nodes(root):
        label = truncate(f"{node.strategy}: {node.claim}", max_label_chars).replace('"', "'").replace("\n", " ")
        lines.append(f'  {node.node_id}["{label}<br/>score={node.score:.2f}, {node.status}"]')
        for child in node.children:
            lines.append(f"  {node.node_id} --> {child.node_id}")
    return "\n".join(lines)
