import re
from dataclasses import dataclass, field
from os import PathLike
from sys import stdout
from typing import List, Iterable, Union, Optional, ClassVar, Dict

# from .penman import load

import penman
from penman import Tree
from penman.tree import is_atomic
import lxml.etree as ET

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


def maybe_convert_relation(relation: str):
    if relation is not None and relation.strip() == "/":
        return "instance"
    else:
        return relation


# TODO: use these in (de)serialization
SPECIAL_TOKENS = {
    "tree": "<tree>",
    "/tree": "</tree>",
    "term": "<term>",
    "/term": "</term>",
    "rel": "<rel>",
    "/rel": "</rel>",
}

REV_SPECIAL_TOKENS = {v: k for k, v in SPECIAL_TOKENS.items()}


@dataclass
class Serializer:
    penman_tree: penman.tree.Tree = field(default=None)
    _xml: ET.ElementBase = field(default=None, init=False)
    _linearized: str = field(default=None, init=False)

    variable_to_ridx: Dict[str, str] = field(default_factory=dict, init=False)

    regex_rx: ClassVar[re.Pattern] = re.compile(r"<R(\d+)>", flags=re.IGNORECASE)

    def set_variable_ridx(self, varname: str):
        """argument_to_idx is a dictionary of the variable (often just a character) to its index
        which we can then use when creating <RO> etc."""
        # Don't do anything if arg is already in the dictionary
        if not self.variable_to_ridx:
            self.variable_to_ridx[varname] = "R0"
        elif varname not in self.variable_to_ridx:
            idxs = [int(r[1:]) for r in self.variable_to_ridx.values()]  # remove "R" and get int value
            max_idx = max(idxs)
            self.variable_to_ridx[varname] = "R" + str(max_idx + 1)

    @property
    def ridx_to_variable(self):
        return {v: k for k, v in self.variable_to_ridx.items()}

    @property
    def xml(self):
        if self._xml is None:
            self._xml = self.xmlify()
        return self._xml

    @property
    def linearized(self):
        if self._linearized is None:
            self._linearized = self.linearize()

        return self._linearized

    def print_xml(self, pretty_print: bool = True):
        xml_str = ET.tostring(self.xml, pretty_print=pretty_print)
        print(xml_str.decode())

    def xmlify(self, parent_node=None, descriptor: Optional[str] = None, is_root: bool = True):
        parent_node = self.penman_tree if parent_node is None else parent_node
        serialized = ""
        if is_root:
            self.set_variable_ridx(parent_node.node[0])
            serialized += f'<tree value="{self.variable_to_ridx[parent_node.node[0]]}" varname="{parent_node.node[0]}" relation_type="ROOT">'

        descriptor = maybe_convert_relation(descriptor)

        if is_atomic(parent_node):  # Terminals
            if descriptor == "instance":  # Instances
                if "-" in parent_node:
                    node_name, sense_id = parent_node.rsplit("-", 1)
                    serialized += f'<term token="{node_name}" sense_id="{sense_id}" />'
                else:  # Without explicit sense-id
                    serialized += f'<term token="{parent_node}" sense_id="NO" />'
            else:  # References to other variables
                self.set_variable_ridx(parent_node)
                serialized += f'<termref ref="{self.variable_to_ridx[parent_node]}" varname="{parent_node}" relation_type="{descriptor}" />'
        else:  # Branches
            if not isinstance(parent_node, Tree):
                parent_node = Tree(parent_node)

            varname, branches = parent_node.node

            if descriptor is not None and varname is not None:
                self.set_variable_ridx(varname)
                serialized += f'<rel ref="{self.variable_to_ridx[varname]}" varname="{varname}" relation_type="{descriptor}">'

            for descriptor, target in branches:
                serialized += self.xmlify(target, descriptor, is_root=False)

            if descriptor is not None and varname is not None and not is_root:
                serialized += "</rel>"

        if is_root:
            serialized += "</tree>"
            return ET.fromstring(serialized)
        else:
            return serialized

    def xml_to_penman(self):
        pass

    def linearize(self, xml: Optional[ET.ElementBase] = None, is_root: bool = True):
        xml = xml if xml is not None else self.xml
        linearized = ""
        if is_root or xml.tag.lower() == "rel":
            if is_root:
                linearized += f"{SPECIAL_TOKENS['tree']}<{xml.attrib['value']}>"
            else:
                linearized += f"{xml.attrib['relation_type']}{SPECIAL_TOKENS['rel']}<{xml.attrib['ref']}>"

            for node in xml:
                linearized += self.linearize(node, is_root=False)

            if is_root:
                linearized += SPECIAL_TOKENS["/tree"]
            else:
                linearized += SPECIAL_TOKENS["/rel"]
        elif xml.tag.lower() == "term":
            linearized += f"{xml.attrib['token']}<sense-id:{xml.attrib['sense_id']}>"
        elif xml.tag.lower() == "termref":
            linearized += f"{xml.attrib['relation_type']}<{xml.attrib['ref']}>"
        else:
            raise ValueError("Unrecognized XML tag")

        return linearized

    @classmethod
    def delinearize(cls, string: str):
        xml_str = "<tree"

        tempstr = ""
        for char in string:
            tempstr += char

            if tempstr in REV_SPECIAL_TOKENS:
                xml_str += REV_SPECIAL_TOKENS[tempstr]
                tempstr = ""

            if match := cls.regex_rx.match(tempstr):
                ref = match.group(1)

        xml_str += "</tree>"

if __name__ == '__main__':
    penman_str = """
     ( t / tell-01 
        :ARG0 ( y / you )
        :ARG1 (w / wash-01
            :ARG0 i
            :ARG1 ( d / dog ) )
        :ARG2 ( i / I ) )

    """
    tree = penman.parse(penman_str)
    serializer = Serializer(penman_tree=tree)
    serializer.print_xml()
    print(serializer.linearized)
    print(serializer.variable_to_ridx)
