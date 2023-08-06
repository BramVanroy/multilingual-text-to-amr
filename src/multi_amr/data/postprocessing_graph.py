import enum
import re
import sys
from typing import List, Tuple

import networkx as nx
import penman
from penman import Graph


BACKOFF = penman.Graph(
    [
        penman.Triple("d2", ":instance", "dog"),
        penman.Triple("b1", ":instance", "bark-01"),
        penman.Triple("b1", ":ARG0", "d2"),
    ]
)


class ParsedStatus(enum.Enum):
    OK = 0
    FIXED = 1
    BACKOFF = 2


def connect_graph_if_not_connected(graph):
    try:
        encoded = penman.encode(graph)
        return graph, ParsedStatus.OK
    except:
        pass

    nxgraph = nx.MultiGraph()
    variables = graph.variables()
    for v1, _, v2 in graph.triples:
        if v1 in variables and v2 in variables:
            nxgraph.add_edge(v1, v2)
        elif v1 in variables:
            nxgraph.add_edge(v1, v1)

    triples = graph.triples.copy()
    new_triples = []
    addition = f"a{len(variables) + 1}"
    triples.append(penman.Triple(addition, ":instance", "and"))
    for i, conn_set in enumerate(nx.connected_components(nxgraph), start=1):
        edge = f":op{i}"
        conn_set = sorted(conn_set, key=lambda x: int(x[1:]))
        conn_set = [c for c in conn_set if c in variables]
        node = conn_set[0]
        new_triples.append(penman.Triple(addition, edge, node))
    triples = new_triples + triples
    metadata = graph.metadata
    graph = penman.Graph(triples)
    graph.metadata.update(metadata)
    penman.encode(graph)
    return graph, ParsedStatus.FIXED


def fix_and_make_graph(nodes):
    nodes_ = []
    for n in nodes:
        if isinstance(n, str):
            if n.startswith("<") and n.endswith(">") and (not n.startswith("<pointer:")):
                pass
            else:
                nodes_.append(n)
        else:
            nodes_.append(n)
    nodes = nodes_
    print("first loop", nodes)
    i = 0
    nodes_ = []
    while i < len(nodes):
        nxt = nodes[i]
        pst = None
        if isinstance(nxt, str) and nxt.startswith("<pointer:"):
            e = nxt.find(">")
            if e != len(nxt) - 1:
                pst = nxt[e + 1 :]
                nxt = nxt[: e + 1]
            nodes_.append(nxt)
            if pst is not None:
                nodes_.append(pst)
        else:
            nodes_.append(nxt)
        i += 1
    nodes = nodes_
    print("second loop", nodes)

    i = 1
    nodes_ = [nodes[0]]
    while i < len(nodes):
        nxt = nodes[i]
        if isinstance(nxt, str) and nxt.startswith("<pointer:"):
            nxt = "z" + nxt[9:-1]
            fol = nodes[i + 1]
            # is not expansion
            if isinstance(fol, str) and (fol.startswith(":") or (fol == ")")):
                nodes_.append(nxt)
            else:
                if nodes_[-1] != "(":
                    nodes_.append("(")
                nodes_.append(nxt)
                nodes_.append("/")
        else:
            nodes_.append(nxt)
        i += 1
    nodes = nodes_
    print("third loop", nodes)

    i = 0
    nodes_ = []
    last = True
    while i < (len(nodes) - 1):
        if nodes[i] == ":":
            nodes_.append(nodes[i] + nodes[i + 1])
            i += 2
            last = False
        else:
            nodes_.append(nodes[i])
            i += 1
            last = True
    if last:
        nodes_.append(nodes[-1])
    nodes = nodes_
    print("fourth loop", nodes)
    i = 0
    nodes_ = []
    while i < (len(nodes)):
        if i < 2:
            nodes_.append(nodes[i])
            i += 1
        elif nodes_[-2] == "/" and nodes[i] == "/":
            i += 2
        else:
            nodes_.append(nodes[i])
            i += 1
    nodes = nodes_
    print("fifth loop", nodes)
    i = 0
    newvars = 0
    variables = set()
    remap = {}
    nodes_ = []
    while i < (len(nodes)):
        next = nodes[i]

        if next == "/":
            last = nodes_[-1]
            if last in variables:
                last_remap = f"z{newvars+1000}"
                newvars += 1
                nodes_[-1] = last_remap
                remap[last] = last_remap
            variables.add(last)
            nodes_.append(next)

        elif _classify(next) == "VAR" and next in remap and (i < len(nodes) - 1) and nodes[i + 1] != "/":
            next = remap[next]
            nodes_.append(next)

        else:
            nodes_.append(next)

        i += 1
    print("sixth loop", nodes)
    nodes = nodes_
    pieces_ = []
    open_cnt = 0
    closed_cnt = 0
    if nodes[0] != "(":
        pieces_.append("(")
        open_cnt += 1
    for p in nodes:
        if p == "(":
            open_cnt += 1
        elif p == ")":
            closed_cnt += 1
        pieces_.append(p)
        if open_cnt == closed_cnt:
            break
    nodes = pieces_ + [")"] * (open_cnt - closed_cnt)

    pieces = []
    for piece in nodes:
        if not pieces:
            pieces.append("(")
        else:
            piece = str(piece)
            if piece.startswith('"') or piece.startswith('"') or '"' in piece.strip('"'):
                piece = '"' + piece.replace('"', "") + '"'

            prev = _classify(pieces[-1])
            next = _classify(piece)

            if next == "CONST":
                quote = False
                for char in (",", ":", "/", "(", ")", ".", "!", "?", "\\", "_", "="):
                    if char in piece:
                        quote = True
                        break
                if quote:
                    piece = '"' + piece.strip('"') + '"'

            if prev == "(":
                if next in ("VAR", "I"):
                    pieces.append(piece)
            elif prev == ")":
                if next in (")", "EDGE", "MODE"):
                    pieces.append(piece)
            elif prev == "VAR":
                if next in ("/", "EDGE", "MODE", ")"):
                    pieces.append(piece)
            elif prev == "/":
                if next in ("INST", "I"):
                    pieces.append(piece)
            elif prev == "INST":
                if next in (")", "EDGE", "MODE"):
                    pieces.append(piece)
            elif prev == "I":
                if next in ("/", ")", "EDGE", "MODE"):
                    pieces.append(piece)
            elif prev == "EDGE":
                if next in ("(", "VAR", "CONST", "I"):
                    pieces.append(piece)
                elif next == ")":
                    pieces[-1] = piece
                elif next in ("EDGE", "MODE"):
                    pieces[-1] = piece
            elif prev == "MODE":
                if next == "INST":
                    pieces.append(piece)
            elif prev == "CONST":
                if next in (")", "EDGE", "MODE"):
                    pieces.append(piece)

    pieces_ = []
    open_cnt = 0
    closed_cnt = 0
    if pieces[0] != "(":
        pieces_.append("(")
        open_cnt += 1
    for p in pieces:
        if p == "(":
            open_cnt += 1
        elif p == ")":
            closed_cnt += 1
        pieces_.append(p)
        if open_cnt == closed_cnt:
            break
    pieces = pieces_ + [")"] * (open_cnt - closed_cnt)

    linearized = re.sub(r"\s+", " ", " ".join(pieces)).strip()

    graph = penman.decode(linearized + " ")
    triples = []
    newvars = 2000
    for triple in graph.triples:
        x, rel, y = triple
        if x is None:
            pass
        elif rel == ":instance" and y is None:
            triples.append(penman.Triple(x, rel, "thing"))
        elif y is None:
            var = f"z{newvars}"
            newvars += 1
            triples.append(penman.Triple(x, rel, var))
            triples.append(penman.Triple(var, ":instance", "thing"))
        else:
            triples.append(triple)
    graph = penman.Graph(triples)
    linearized = penman.encode(graph)

    def fix_text(_linearized=linearized):
        n = 0

        def _repl1(match):
            nonlocal n
            out = match.group(1) + match.group(2) + str(3000 + n) + " / " + match.group(2) + match.group(3)
            n += 1
            return out

        _linearized = re.sub(r"(\(\s?)([a-z])([^\/:)]+[:\)])", _repl1, _linearized, flags=re.IGNORECASE | re.MULTILINE)

        def _repl2(match):
            return match.group(1)

        _linearized = re.sub(
            r"(\(\s*[a-z][\d+]\s*\/\s*[^\s\)\(:\/]+\s*)((?:/\s*[^\s\)\(:\/]+\s*)+)",
            _repl2,
            _linearized,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # adds a ':' to args w/o it
        _linearized = re.sub(r"([^:])(ARG)", r"\1 :\2", _linearized)

        # removes edges with no node
        # linearized = re.sub(r':[^\s\)\(:\/]+?\s*\)', ')', linearized, flags=re.MULTILINE)

        return _linearized

    linearized = fix_text(linearized)
    g = penman.decode(linearized)
    return g


def _classify(node):
    if not isinstance(node, str):
        return "CONST"
    elif node == "i":
        return "I"
    elif re.match(r"^[a-z]\d*$", node) is not None:
        return "VAR"
    elif node[0].isdigit():
        return "CONST"
    elif node.startswith('"') and node.endswith('"'):
        return "CONST"
    elif node in ("+", "-"):
        return "CONST"
    elif node == ":mode":
        return "MODE"
    elif node.startswith(":"):
        return "EDGE"
    elif node in ["/", "(", ")"]:
        return node
    elif node[0].isalpha():
        for char in (",", ":", "/", "(", ")", ".", "!", "?", "\\"):
            if char in node:
                return "CONST"
        return "INST"
    else:
        return "CONST"


def tokens2graph(tokens: List[str], verbose: bool = False) -> Tuple[Graph, ParsedStatus]:
    try:
        graph_ = graph = fix_and_make_graph(tokens)
    except Exception as e:
        if verbose:
            print("Building failure:", file=sys.stderr)
            print(tokens, file=sys.stderr)
            print(e, file=sys.stderr)
        return BACKOFF, ParsedStatus.BACKOFF
    try:
        graph, status = connect_graph_if_not_connected(graph)
        if status == ParsedStatus.BACKOFF:
            if verbose:
                print("Reconnection 1 failure:")
                print(tokens, file=sys.stderr)
                print(graph_, file=sys.stderr)
        return graph, status
    except Exception as e:
        print("Reconnction 2 failure:", file=sys.stderr)
        if verbose:
            print(e, file=sys.stderr)
            print(tokens, file=sys.stderr)
            print(graph_, file=sys.stderr)
        return BACKOFF, ParsedStatus.BACKOFF
