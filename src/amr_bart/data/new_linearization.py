import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Optional, ClassVar, Dict, Counter
from collections import Counter, deque

import penman
from penman import Tree, Triple, Graph
from penman.tree import is_atomic, _default_variable_prefix
import lxml.etree as ET
from tqdm import tqdm

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

# TODO: check special arguments, especially :opX and things related non-core roles: https://github.com/amrisi/amr-guidelines/blob/master/amr.md
# TODO: check in amr guidelines how variable assignment for p, p2, p3 is prioritized. First the deepest token, or which one is the first p? Here the "reset_Variables" function in penman seems relevant


def elements_equal(e1, e2):
    """https://stackoverflow.com/a/24349916/1150683"""
    if e1.tag != e2.tag: return False
    if e1.text != e2.text: return False
    if e1.tail != e2.tail: return False
    if e1.attrib != e2.attrib: return False
    if len(e1) != len(e2): return False
    return all(elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


def escape_xml(str_xml: str):
    str_xml = str_xml.replace("&", "&amp;")
    str_xml = str_xml.replace("<", "&lt;")
    str_xml = str_xml.replace(">", "&gt;")
    str_xml = str_xml.replace("\"", "&quot;")
    str_xml = str_xml.replace("'", "&apos;")
    return str_xml


def maybe_convert_relation(relation: str):
    if relation is not None and relation.strip() == "/":
        return "instance"
    else:
        return relation

def remove_wiki(penman_str: str):
    return re.sub(r'\s+:wiki\s+(?:\"[^\"]+\"|-)', "", penman_str)

def remove_metadata(penman_str: str):
    return re.sub(r'^#.*\n', "", penman_str, flags=re.MULTILINE)


def xml2penman_str(xml: Optional[ET.ElementBase] = None, is_root: bool = True):
    penman_str = ""
    if is_root or xml.tag.lower() == "rel":
        if is_root:
            penman_str += f'({xml.attrib["varname"]} '
        else:
            penman_str += f'{xml.attrib["relation_type"]} ( {xml.attrib["varname"]} '

        for node in xml:
            penman_str += xml2penman_str(node, is_root=False)

        penman_str += ") "
    elif xml.tag.lower() == "term":
        if xml.attrib["sense_id"].lower() == "no":
            penman_str += f'/ {xml.attrib["token"]} '
        else:
            penman_str += f'/ {xml.attrib["token"]}-{xml.attrib["sense_id"]} '

    elif xml.tag.lower() == "termref":
        penman_str += f'{xml.attrib["relation_type"]} {xml.attrib["varname"]} '

    elif xml.tag.lower() == "termrel":
        penman_str += f'{xml.attrib["relation_type"]} {xml.attrib["token"]} '
    else:
        raise ValueError("Unrecognized XML tag")

    return penman_str


def linearize_from_xml(xml: Optional[ET.ElementBase] = None, is_root: bool = True):
    linearized = ""
    if is_root or xml.tag.lower() == "rel":
        if is_root:
            linearized += f'<tree><termid value="{xml.attrib["value"]}"/>'
        else:
            linearized += f'<rel><reltype value="{xml.attrib["relation_type"]}"/><termid value="{xml.attrib["ref"]}"/>'

        for node in xml:
            linearized += linearize_from_xml(node, is_root=False)

        if is_root:
            linearized += "</tree>"
        else:
            linearized += "</rel>"
    elif xml.tag.lower() == "term":
        linearized += f'{xml.attrib["token"]}<sense_id value="{xml.attrib["sense_id"]}"/>'
    elif xml.tag.lower() == "termref":
        linearized += f'<reltype value="{xml.attrib["relation_type"]}"/><termref value="{xml.attrib["ref"]}"/>'
    elif xml.tag.lower() == "termrel":
        linearized += f'<reltype value="{xml.attrib["relation_type"]}"/>{xml.attrib["token"]}<sense_id value="NO"/>'
    else:
        raise ValueError("Unrecognized XML tag")

    return linearized


def set_varnames(xml: ET.ElementBase):
    var_counter = Counter()
    for node in xml.xpath("//*[@varname]/term"):
        varname = _default_variable_prefix(node.attrib["token"])
        var_counter[varname] += 1
        node.getparent().attrib["varname"] = varname if var_counter[varname] == 1 else f"{varname}{var_counter[varname]}"

    for node in xml.xpath("//termref"):
        ref = xml.xpath(f'//rel[@ref="{node.attrib["ref"]}"]')
        if ref:
            node.attrib["varname"] = ref[0].attrib["varname"]

    return xml


def delinearize_to_xml(linearized_amr: str):
    # TODO: if an error occurs here during parsing, try to fix the tree
    # Maybe with BS4, which is supposedly more lenient (?)
    xml = ET.fromstring(linearized_amr)

    def _iterate(node, parent_tag: str = None, prev_sibling=None):
        xml_str = ""
        prev_tail = None if prev_sibling is None or not prev_sibling.tail else prev_sibling.tail.strip()

        if node.tag.lower() == "termid":
            if parent_tag == "tree":
                xml_str += f'<tree value="{node.attrib["value"]}" varname="" relation_type="ROOT">'
            elif parent_tag == "rel":
                if prev_sibling is not None and prev_sibling.tag.lower() == "reltype":  # relation_type might be empty if malformed
                    xml_str += f'<rel ref="{node.attrib["value"]}" varname="" relation_type="{prev_sibling.attrib["value"]}">'
                else:
                    xml_str += f'<rel ref="{node.attrib["value"]}" varname="" relation_type="">'

        elif node.tag.lower() == "sense_id":
            token_str = prev_tail if prev_tail else ""  # token might be empty if malformed
            if prev_sibling.tag.lower() == "reltype":  # termrels
                xml_str += f'<termrel token="{token_str}" relation_type="{prev_sibling.attrib["value"]}" />'
            else:  # terminals
                xml_str += f'<term token="{prev_tail}" sense_id="{node.attrib["value"]}"/>'

        elif node.tag.lower() == "termref":
            if prev_sibling is not None and prev_sibling.tag.lower() == "reltype":  # relation_type might be empty if malformed
                xml_str += f'<termref ref="{node.attrib["value"]}" varname="" relation_type="{prev_sibling.attrib["value"]}"/>'
            else:
                xml_str += f'<termref ref="{node.attrib["value"]}" varname="" relation_type=""/>'

        for child_idx, child in enumerate(node):
            sibling = node[child_idx - 1] if 0 < child_idx < len(node) else None
            xml_str += _iterate(child, node.tag.lower(), sibling)

        if parent_tag is None:
            xml_str += "</tree>"
        elif node.tag.lower() == "rel":
            xml_str += "</rel>"

        return xml_str

    final_xml_str = _iterate(xml)

    xml = ET.fromstring(final_xml_str)
    xml = set_varnames(xml)
    return xml


@dataclass
class Linearizer:
    penman_tree: penman.tree.Tree = field(default=None)
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

    @cached_property
    def xml(self):
        return self.penman_tree2xml()

    @cached_property
    def linearized(self):
        return linearize_from_xml(self.xml)

    @cached_property
    def penman_str(self):
        return penman.format(self.penman_tree)

    def pprint_xml(self):
        xml_str = ET.tostring(self.xml, pretty_print=True)
        print(xml_str.decode())

    def pprint_linearized(self):
        """We cannot use ET.tostring because we have text right next to other nodes (as `tail`s)
        so LXML will not be able to correctly prettify the code. Alright, I'll do it myself.
        """
        xml = ET.fromstring(self.linearized)

        def _iterate(xml, indent: int = 0):
            xml_str = "\n" + "\t" * indent
            attribs = " ".join([f'{k}="{v}"' for k, v in xml.attrib.items()])
            xml_str += f'<{xml.tag} {attribs}' if attribs else f'<{xml.tag}'

            if len(xml):
                xml_str += ">"
                for node in xml:
                    xml_str += _iterate(node, indent=indent+1)
                xml_str += "\n" + "\t" * indent
                xml_str += f'</{xml.tag}>'
            else:
                xml_str += "/>"

            xml_str += xml.tail if xml.tail else ""

            return xml_str

        pprint_lin = _iterate(xml)
        print(pprint_lin)

    def penman_tree2xml(self, parent_node=None, descriptor: Optional[str] = None, is_root: bool = True):
        parent_node = self.penman_tree if parent_node is None else parent_node
        xml_str = ""
        if is_root:
            self.set_variable_ridx(parent_node.node[0])
            xml_str += f'<tree value="{self.variable_to_ridx[parent_node.node[0]]}" varname="{escape_xml(parent_node.node[0])}" relation_type="ROOT">'

        descriptor = maybe_convert_relation(descriptor)

        if is_atomic(parent_node):  # Terminals
            if descriptor == "instance":  # Instances
                if "-" in parent_node:
                    node_name, sense_id = parent_node.rsplit("-", 1)
                    xml_str += f'<term token="{escape_xml(node_name)}" sense_id="{sense_id}" />'
                else:  # Without explicit sense-id
                    xml_str += f'<term token="{escape_xml(parent_node)}" sense_id="NO" />'
            else:
                if descriptor.startswith(":op"):
                    if re.match(r"^[a-z]\d+$", parent_node):  # Reference to other variable
                        xml_str += f'<termref ref="{self.variable_to_ridx[parent_node]}" varname="{escape_xml(parent_node)}" relation_type="{descriptor}" />'
                    else:  # A terminal token which is also a some special relation (e.g. `:op1 100000`)
                        # NOTE: it _seems_ to me that this can only occur for numbers. A string will always be a token "instance" (?)
                        xml_str += f'<termrel token="{escape_xml(parent_node)}" relation_type="{descriptor}" />'
                else: # References to other variables
                    self.set_variable_ridx(parent_node)
                    if descriptor == ":wiki":
                        raise ValueError("Wiki items are currently not supported")

                    xml_str += f'<termref ref="{self.variable_to_ridx[parent_node]}" varname="{escape_xml(parent_node)}" relation_type="{descriptor}" />'
        else:  # Branches
            if not isinstance(parent_node, Tree):
                parent_node = Tree(parent_node)

            varname, branches = parent_node.node

            if descriptor is not None and varname is not None:
                self.set_variable_ridx(varname)
                xml_str += f'<rel ref="{self.variable_to_ridx[varname]}" varname="{escape_xml(varname)}" relation_type="{descriptor}">'

            for descriptor, target in branches:
                xml_str += self.penman_tree2xml(target, descriptor, is_root=False)

            if descriptor is not None and varname is not None and not is_root:
                xml_str += "</rel>"

        if is_root:
            xml_str += "</tree>"
            print(xml_str)
            return ET.fromstring(xml_str)
        else:
            return xml_str

    @classmethod
    def from_linearized(cls, linearized_amr: str):
        xml = delinearize_to_xml(linearized_amr)
        return cls.from_xml(xml)

    @classmethod
    def from_xml(cls, xml: ET.ElementBase):
        penman_str = xml2penman_str(xml)
        return cls.from_penman_str(penman_str)

    @classmethod
    def from_penman_str(cls, penman_str: str, keep_wiki: bool = False, keep_metadata: bool = False):
        if not keep_wiki:
            penman_str = remove_wiki(penman_str)
        if not keep_metadata:
            penman_str = remove_metadata(penman_str)

        penman_tree = penman.parse(penman_str)
        return cls(penman_tree=penman_tree)


stri = """
(m / multi-sentence
   :snt1 (m2 / many
             :ARG0-of (s / sense-01
                         :ARG1 (u / urgency)
                         :time (w / watch-01
                                  :ARG0 m2
                                  :ARG1 (t3 / thing
                                            :manner-of (d / develop-02
                                                          :ARG0 (t / thing)))
                                  :manner (q / quiet-04
                                             :ARG1 m2))))
   :snt2 (d2 / dragon
             :domain (y / you)
             :ARG0-of (c / coil-01))
   :snt3 (t2 / tiger
             :domain (y2 / you)
             :ARG0-of (c2 / crouch-01))
   :snt4 (a / admire-01
            :ARG0 (i / i)
            :ARG1 (p / patriot
                     :poss-of (m3 / mind
                                  :mod (n / noble)))))
"""

stri2 = """
(m2 / multi-sentence
    :snt1 (g / give-01
             :ARG0 (h / history)
             :ARG1 (l / lesson
                      :ARG1-of (h2 / have-quant-91
                                   :ARG2 (m / many)
                                   :ARG3 (t / too)))
             :ARG2 (w / we)
             :polarity (a2 / amr-unknown))
    :snt2 (a / and
             :op1 530
             :op2 412
             :op3 64))
"""

if __name__ == "__main__":
    run_dir_test = True

    if run_dir_test:
        pdin = Path(r"D:\corpora\amr_annotation_3.0\data\amrs_fixed")

        valid_trees = 0
        invalid_trees = 0
        for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
            with pfin.open(encoding="utf-8") as fhin:
                for tree in penman.iterparse(fhin):
                    tree_str = penman.format(tree)
                    linearizer = Linearizer.from_penman_str(tree_str)
                    xml = linearizer.xml
                    linearized = linearizer.linearized
                    penman_str = linearizer.penman_str
                    relinearizer = Linearizer.from_linearized(linearized)

                    if linearizer.penman_tree != relinearizer.penman_tree:
                        print(linearizer.penman_str)
                        print(relinearizer.penman_str)
                        input("Type something here to contiue")
    else:
        pnman_str = """# ::id bolt12_64545_0529.2 ::date 2012-12-23T19:59:16 ::annotator SDL-AMR-09 ::preferred
# ::snt What is more they are considered traitors of China, which is a fact of cultural tyranny in the cloak of nationalism and patriotism.
# ::save-date Sun Dec 8, 2013 ::file bolt12_64545_0529_2.txt
(c / consider-01
      :ARG1 (p2 / person
            :domain (t2 / they)
            :ARG0-of (b / betray-01
                  :ARG1 (c2 / country :wiki "China"
                        :name (n2 / name :op1 "China"))))
      :mod (m / more)
      :mod (t4 / tyrannize-01
            :ARG2 (c3 / culture)
            :ARG1-of (c4 / cloak-01
                  :ARG2 (a / and
                        :op1 (n / nationalism)
                        :op2 (p / patriotism)))))
"""
        linearizer = Linearizer.from_penman_str(pnman_str)
        linearizer.pprint_linearized()
        # new_linearizer = Linearizer.from_linearized(linearizer.linearized)
