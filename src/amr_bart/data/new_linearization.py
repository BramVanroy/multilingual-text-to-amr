from os import PathLike
from typing import List, Iterable, Union

from .penman import load

import penman
from penman import Tree
from penman.tree import is_atomic

"""
TODO: build DFS linearizer from scratch based on the paper description page 12566 and figure 1
https://ojs.aaai.org/index.php/AAAI/article/view/17489
- Of course use the linearized graph as the labels and (shifted) as the decoder inputs
- update tokenizer with special tokens like :ARG and all others (probably gotta check the guidelines for all possible tags)
- add special tokens for opening/closing brackets

https://github.com/amrisi/amr-guidelines/blob/master/amr.md
Voc; We can try:
    - putting all tokens in the new vocabulary as full tokens (e.g. believe-01); OR
    - using special tokens for the sense-dash and sense-id so that the concept itself can be of multiple subword
     tokens (e.g., [bel, ieve, <sensedash>, <sense1>])
Custom constrained beam search? (if a token is not allowed in a position, set its logits to 0?)
    - a sense dash must be followed by a sense
    - special reference token (<R0>) can only follow 1. an opening bracket or 2. an arg:
"""
def read_graphs(
    paths: Union[List[Union[str, PathLike]], Union[str, PathLike]],
    use_recategorization=False,
    dereify=True,
    remove_wiki=False,
):
    if not paths:
        raise ValueError("Cannot read AMRs. Paths is empty")

    if not isinstance(paths, Iterable):
        paths = [paths]

    graphs = [graph for path in paths for graph in load(path, dereify=dereify, remove_wiki=remove_wiki)]

    if use_recategorization:
        for graph in graphs:
            metadata = graph.metadata
            metadata["snt_orig"] = metadata["snt"]
            tokens = eval(metadata["tokens"])
            metadata["snt"] = " ".join(
                [t for t in tokens if not ((t.startswith("-L") or t.startswith("-R")) and t.endswith("-"))]
            )

    return graphs




def format_sense(idx: int):
    if 0 < idx < 25:
        return f"<sense-id:{idx}>"
    else:
        return "<sense-id:x>"


def format_relation(relation_type: str):
    if relation_type == "/":
        return "<relation_type:instance>"
    else:
        return f"<sense-id:{relation_type.replace(':', '')}>"


def serialize_node(parent_node, descriptor=None, varname=None, is_root=True):
    if is_root:
        print("OPEN TREE")
    if is_atomic(parent_node):
        print("TERMINAL", parent_node, descriptor, varname)
    else:
        if not isinstance(parent_node, Tree):
            parent_node = Tree(parent_node)

        varname, branches = parent_node.node

        if descriptor is not None and varname is not None:
            print("OPEN", descriptor, varname)

        for descriptor, target in branches:
            serialize_node(target, descriptor, varname, is_root=False)

        if descriptor is not None and varname is not None:
            print("CLOSE", descriptor, varname)

    if is_root:
        print("CLOSE TREE")


if __name__ == '__main__':
    penman_str = """
    (s / sleep
        :ARG0 (d / dog
            :ARG0-of (b / bark-01)
        )
    )
    """
    tree = penman.parse(penman_str)

    serialize_node(tree)
