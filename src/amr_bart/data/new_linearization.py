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
    relation_type = relation_type.split(":")[1].strip().replace(">", "")

    if relation_type.lower() == "instance":
        return "/"
    else:
        return relation_type

# TODO: use these in (de)serialization
SPECIAL_TOKENS = {
    "open_tree": "<tree>",
    "close_tree": "</tree>",
    "open_term": "<term>",
    "close_term": "</term>",
    "open_rel": "<rel>",
    "close_rel": "</rel>",
}

REV_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS.items()}


@dataclass
class Serializer:
    penman_tree: penman.tree.Tree = field(default=None)
    serialized: str = field(default=None)
    amr_str: str = field(default=None)  # TODO: also save the AMR string/penman serialization
    argument_to_idx: None = field(default_factory=dict, init=False)

    regex_node: ClassVar[re.Pattern] = re.compile(r"(?:([A-Z]+)\(\(([^)]*)\)(.*)\)\/\1)|(?:TERM\(([^)]*)\))",
                                                  flags=re.DOTALL | re.IGNORECASE)
    regex_sense: ClassVar[re.Pattern] = re.compile(r"(.*)(<sense-id:.*>)", flags=re.IGNORECASE)
    regex_args: ClassVar[re.Pattern] = re.compile(r"(<R\d+>) / (\w)", flags=re.IGNORECASE)

    def __post_init__(self):
        if self.penman_tree and not self.serialized:
            self.serialized = self.serialize_node(self.penman_tree)
        elif self.serialized and not self.penman_tree:
            pass
        elif not (self.penman_tree is not None and self.serialized is not None):
            raise ValueError("Either 'penman_tree' or 'serialized' has to be given!")

    @property
    def r_idx_to_arg(self):
        return {v: k for k, v in self.argument_to_idx.items()}

    def set_arg_idx(self, arg: str):
        """argument_to_idx is a dictionary of the variable (often just a character) to its index
        which we can then use when creating <RO> etc."""
        # Don't do anything if arg is already in the dictionary
        if not self.argument_to_idx:
            self.argument_to_idx[arg] = 0
        elif arg not in self.argument_to_idx:
            max_idx = max(self.argument_to_idx.values())
            self.argument_to_idx[arg] = max_idx+1

    def get_serialized_arg(self, arg: str):
        idx = self.argument_to_idx[arg]
        return f"<R{idx}>"

    def serialize_node(self, parent_node, descriptor=None, is_root=True, level=0, pretty: bool = True):
        serialized = "\t" * level if pretty else ""
        if is_root:
            self.set_arg_idx(parent_node.node[0])
            serialized += f"TREE(({serialize_relation('ROOT')}, {self.get_serialized_arg(parent_node.node[0])}) "

        if is_atomic(parent_node):  # Terminals
            serialized = ("\n" + serialized) if pretty else serialized
            if descriptor.strip() == "/":  # Instances
                if "-" in parent_node:
                    node_name, sense_id = parent_node.rsplit("-", 1)
                    sense_id = serialize_sense(sense_id)
                    serialized += f"TERM({node_name}{sense_id}, {serialize_relation(descriptor)}) "
                else:
                    sense_id = serialize_sense(None)
                    serialized += f"TERM({parent_node}{sense_id}, {serialize_relation(descriptor)}) "
            else:  # References to other variables
                self.set_arg_idx(parent_node)
                serialized += f"TERM({self.get_serialized_arg(parent_node)}, {serialize_relation(descriptor)}) "
        else:  # Branches
            if not isinstance(parent_node, Tree):
                parent_node = Tree(parent_node)

            varname, branches = parent_node.node

            if descriptor is not None and varname is not None:
                self.set_arg_idx(varname)
                serialized = ("\n" + serialized) if pretty else serialized
                serialized += f"REL(({serialize_relation(descriptor)}, {self.get_serialized_arg(varname)}), "

            for descriptor, target in branches:
                serialized += self.serialize_node(target, descriptor, is_root=False, level=level+1)

            if descriptor is not None and varname is not None and not is_root:
                serialized += ("\n" + ("\t" * level)) if pretty else ""
                serialized += ")/REL "

        if is_root:
            serialized += "\n)/TREE" if pretty else ")/TREE"
            self.serialized = serialized

        return serialized

    def deserialize(self, text: str, is_root: bool = True):
        if text.count("(") != text.count(")"):
            raise ValueError("Serialized tree is malformed. Opening and closing parentheses do not match")

        amr_str = ""
        for match in re.finditer(self.regex_node, text):
            is_terminal = match.group(4)  # If we match a terminal group (not None)
            if is_terminal:  # Terminal
                token, relation_type = [item.strip() for item in match.group(4).split(",")]
                relation_type = deserialize_relation(relation_type)
                if token_match := re.match(self.regex_sense, token):  # Token
                    token = token_match.group(1)
                    sense_id = token_match.group(2)
                    sense_id = deserialize_sense(sense_id)
                    amr_str += f"{relation_type} {token}-{sense_id} " if sense_id else f"{relation_type} {token} "
                else:  # Variable
                    print(match)
                    amr_str += f":{relation_type} {token} "
            else: # Relation
                print(match)
                relation_type, varname = [item.strip() for item in match.group(2).split(",")]
                relation_type = deserialize_relation(relation_type)
                descendants = match.group(3)

                # Do not add relation type prefix for the root
                amr_str += f":{relation_type} ({varname} " if relation_type.lower() != "root" else f"({varname} "
                amr_str += self.deserialize(descendants, is_root=False)
                amr_str += ") "

        # if is_root:
            # amr_str = self.replace_args_in_str(amr_str)

        return amr_str

    def replace_args_in_str(self, text: str):
        for r_idx, repl in self.r_idx_to_arg.items():
            text = text.replace(f"<R{r_idx}>", repl)

        return text


if __name__ == '__main__':
    penman_str = """
     ( t / tell-01 
        :ARG0 ( y / you )
        :ARG1 (w / wash-01
            :ARG0 i
            :ARG1 ( d / dog ) )
        :ARG2 ( i / i ) )

    """
    tree = penman.parse(penman_str)
    serializer = Serializer(penman_tree=tree)

    serialized_tree = serializer.serialized
    print(serialized_tree)
    deserialized_tree = serializer.deserialize(serialized_tree)
    print(deserialized_tree)
    # re_tree = penman.parse(deserialized_tree)
    #
    # assert re_tree == tree
