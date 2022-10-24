import re
from dataclasses import dataclass, field
from os import PathLike
from typing import List, Iterable, Union, Optional, ClassVar

# from .penman import load

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

def serialize_sense(str_idx: Optional[str]):
    idx = int(str_idx) if str_idx is not None else None
    if idx is None:
        return "<sense-id:NO>"
    elif 0 < idx < 25:
        return f"<sense-id:{str_idx}>"
    else:  # Maybe we don't want to add potentially hundreds of senses to our vocabulary so rest class
        return "<sense-id:x>"


def deserialize_sense(sense_id: str) -> str:
    sense_id = sense_id.split(":")[1].strip().replace(">", "")

    if sense_id.lower() == "no":
        return ""
    else:
        # Try to convert to int to make sure there is no error
        sense_id = int(sense_id)
        if sense_id < 10:
            return f"0{sense_id}"
        else:
            return str(sense_id)


def serialize_relation(relation_type: str):
    if relation_type == "/":
        return "<relation_type:instance>"
    else:
        relation_type = relation_type if relation_type.startswith(":") else f":{relation_type}"
        return f"<relation_type{relation_type}>"


def deserialize_relation(relation_type: str):
    print(relation_type)
    relation_type = relation_type.split(":")[1].strip().replace(">", "")

    if relation_type.lower() == "instance":
        return "/"
    else:
        return relation_type


@dataclass
class Serializer:
    penman_tree: penman.tree.Tree = field(default=None)
    serialized: str = field(default=None)
    amr_str: str = field(default=None)  # TODO: also save the AMR string/penman serialization
    regex: ClassVar[re.Pattern] = re.compile(r"(?:([A-Z]+)\(\(([^)]*)\)(.*)\)\/\1)|(?:(TERM)\(([^)]*)\))", flags=re.DOTALL | re.IGNORECASE)

    def __post_init__(self):
        if self.penman_tree and not self.serialized:
            self.serialized = self.serialize_node(self.penman_tree)
        elif self.serialized and not self.penman_tree:
            pass
        elif not (self.penman_tree is not None and self.serialized is not None):
            raise ValueError("Either 'penman_tree' or 'serialized' has to be given!")

    def serialize_node(self, parent_node, descriptor=None, is_root=True, level=0, pretty: bool = True):
        serialized = "\t" * level if pretty else ""
        if is_root:
            serialized += f"TREE(({serialize_relation('ROOT')}, {parent_node.node[0]}) "

        if is_atomic(parent_node):  # Terminals
            serialized = ("\n" + serialized) if pretty else serialized
            if "-" in parent_node:
                node_name, sense_id = parent_node.rsplit("-", 1)
                sense_id = serialize_sense(sense_id)
            else:
                node_name = parent_node
                sense_id = serialize_sense(None)
            serialized += f"TERM({node_name}{sense_id}, {serialize_relation(descriptor)}) "
        else:  # Branches
            if not isinstance(parent_node, Tree):
                parent_node = Tree(parent_node)

            varname, branches = parent_node.node

            if descriptor is not None and varname is not None:
                serialized = ("\n" + serialized) if pretty else serialized
                serialized += f"REL(({serialize_relation(descriptor)}, {varname}), "

            for descriptor, target in branches:
                serialized += self.serialize_node(target, descriptor, is_root=False, level=level+1)

            if descriptor is not None and varname is not None and not is_root:
                serialized += ("\n" + ("\t" * level)) if pretty else ""
                serialized += ")/REL "

        if is_root:
            serialized += "\n)/TREE" if pretty else ")/TREE"
            self.serialized = serialized

        return serialized

    @classmethod
    def deserialize(cls, text: str):
        if text.count("(") != text.count(")"):
            raise ValueError("Serialized tree is malformed. Opening and closing parentheses do not match")

        amr_str = ""
        for match in re.finditer(cls.regex, text):
            is_terminal = match.group(4) and match.group(4).strip().lower() == "term"
            if is_terminal:
                node_type = "TERM"
                token, relation_type = [item.strip() for item in match.group(5).split(",")]
                # TODO: probably makes more sense to just get the sense id with a capture group
                token, sense_id = re.split("<sense-id:", token, maxsplit=1, flags=re.IGNORECASE)
                sense_id = f"<sense-id:{sense_id}"

                relation_type = deserialize_relation(relation_type)
                sense_id = deserialize_sense(sense_id)
                amr_str += f"{relation_type} {token}-{sense_id} " if sense_id else f"{relation_type} {token} "
            else:
                node_type = match.group(1).strip()
                relation_type, varname = [item.strip() for item in match.group(2).split(",")]
                relation_type = deserialize_relation(relation_type)
                descendants = match.group(3)

                amr_str += f":{relation_type} ({varname} " if relation_type.lower() != "root" else f"({varname} "
                amr_str += cls.deserialize(descendants)
                amr_str += ")"

        return amr_str




if __name__ == '__main__':
    penman_str = """
    (s / sleep
        :ARG0 (d / dog
            :ARG0-of (b / bark-01)
        )
    )
    """
    tree = penman.parse(penman_str)

    serializer = Serializer(penman_tree=tree)
    serialized_tree = serializer.serialized
    deserialized_tree = Serializer.deserialize(serialized_tree)


    print(serializer.serialized)
    print(deserialized_tree)
    re_tree = penman.parse(deserialized_tree)

    print(tree == re_tree)