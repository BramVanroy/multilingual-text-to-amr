import copy
import re
from typing import List

import penman
from penman import Graph, Triple


def remove_wiki_from_graph(graph: Graph) -> Graph:
    triples = []
    for t in graph.triples:
        v1, rel, v2 = t
        if rel == ":wiki":
            t = Triple(v1, rel, "+")
        triples.append(t)

    return Graph(triples, metadata=graph.metadata)


def tokenize_encoded_graph(linearized: str) -> List[str]:
    linearized = re.sub(r"(\".+?\")", r" \1 ", linearized)
    pieces = []
    for piece in linearized.split():
        if not (piece.startswith('"') and piece.endswith('"')):
            piece = piece.replace("(", " ( ")
            piece = piece.replace(")", " ) ")
            piece = piece.replace(":", " :")
            piece = piece.replace("/", " / ")
            piece = piece.strip()

        pieces.append(piece)
    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()
    return linearized.split(" ")


def dfs_linearize(graph: Graph, use_pointer_tokens: bool = True):
    graph_ = copy.deepcopy(graph)
    graph_.metadata = {}
    linearized = penman.encode(graph_).replace("–", "-")  # NLLB does not have an en-hyphen
    linearized_nodes = tokenize_encoded_graph(linearized)

    if use_pointer_tokens:
        remap = {}
        for i in range(1, len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes[i - 1]
            # Only add pointers if we are not inside a literal
            if nxt == "/" and re.match(r"[a-z]\d*", lst) is not None:
                remap[lst] = f"<pointer:{len(remap)}>"
        i = 1
        linearized_nodes_ = [linearized_nodes[0]]
        while i < (len(linearized_nodes)):
            nxt = linearized_nodes[i]
            lst = linearized_nodes_[-1]
            if nxt in remap:
                if lst == "(" and linearized_nodes[i + 1] == "/":
                    nxt = remap[nxt]
                    i += 1
                elif lst.startswith(":"):
                    nxt = remap[nxt]
            linearized_nodes_.append(nxt)
            i += 1
        linearized_nodes = linearized_nodes_

    linearized_nodes = [tstrip for t in linearized_nodes if (tstrip := t.strip())]

    return linearized_nodes
