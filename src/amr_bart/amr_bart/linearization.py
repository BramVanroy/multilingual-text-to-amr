import re
from dataclasses import dataclass, field
from typing import Optional, ClassVar, Dict, Counter
from collections import Counter

from ftfy import fix_text
import penman
from penman import Tree
from penman.tree import is_atomic, _default_variable_prefix
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
    - using special tokens for the sense-id so that the concept itself can be of multiple subword
     tokens (e.g., [bel, ieve, <sense1>])
Custom constrained beam search? (if a token is not allowed in a position, set its logits to 0?)
    - a sense dash must be followed by a sense
    - special reference token (<R0>) can only follow 1. an opening bracket or 2. an arg:
"""

# TODO: check special arguments, especially :opX and things related non-core roles: https://github.com/amrisi/amr-guidelines/blob/master/amr.md
# TODO: for :op values, remove " surrounding quotes when linearized and add them again when delinearizing. But do not add them for numbers where the whole string.isdigit() is true!
# TODO: clean up the "termid"s: not all terms NEED a termid if they are not referred to, and they may add a lot of noise. However, this means we also have to make sure that the "delinearization" works well when no termid is given

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

def unescape_xml(str_xml: str):
    str_xml = str_xml.replace("&amp;", "&")
    str_xml = str_xml.replace("&lt;", "<")
    str_xml = str_xml.replace("&gt;", ">")
    str_xml = str_xml.replace("&quot;", "\"")
    str_xml = str_xml.replace("&apos;", "'")
    return str_xml


def remove_wiki(penman_str: str):
    return re.sub(r'\s+:wiki\s+(?:\"[^\"]+\"|-)', "", penman_str)


def remove_metadata(penman_str: str):
    return re.sub(r'^#.*\n', "", penman_str, flags=re.MULTILINE)


def xml2penman_str(node: Optional[ET.ElementBase] = None, is_root: bool = True):
    penman_str = ""
    if is_root or node.tag.lower() == "rel":
        if is_root:
            penman_str += f'({node.attrib["varname"]} '
        else:
            penman_str += f'{node.attrib["relation_type"]} ( {node.attrib["varname"]} '

        for node in node:
            penman_str += xml2penman_str(node, is_root=False)

        penman_str += ") "
    elif node.tag.lower() == "term":
        if node.attrib["sense_id"].lower() == "no":
            penman_str += f'/ {node.attrib["token"]} '
        else:
            penman_str += f'/ {node.attrib["token"]}-{node.attrib["sense_id"]} '

    elif node.tag.lower() == "termref":
        penman_str += f'{node.attrib["relation_type"]} {node.attrib["varname"]} '

    elif node.tag.lower() == "termrel":
        penman_str += f'{node.attrib["relation_type"]} {node.attrib["token"]} '
    elif node.tag.lower() == "negation":
        penman_str += ':polarity - '

    else:
        raise ValueError("Unrecognized XML tag")

    return penman_str


def xml_to_linearized(node: Optional[ET.ElementBase] = None, is_root: bool = True):
    linearized = ""
    if is_root or node.tag.lower() == "rel":
        if is_root:
            linearized += f'<tree type="linearized_xml"><termid value="{node.attrib["value"]}"/>'
        else:
            linearized += f'<rel><reltype value="{node.attrib["relation_type"]}"/><termid value="{node.attrib["ref"]}"/>'

        for node in node:
            linearized += xml_to_linearized(node, is_root=False)

        if is_root:
            linearized += "</tree>"
        else:
            linearized += "</rel>"
    elif node.tag.lower() == "term":
        linearized += f'{escape_xml(node.attrib["token"])}<sense_id value="{node.attrib["sense_id"]}"/>'
    elif node.tag.lower() == "termref":
        linearized += f'<reltype value="{node.attrib["relation_type"]}"/><termref value="{node.attrib["ref"]}"/>'
    elif node.tag.lower() == "termrel":
        linearized += f'<reltype value="{node.attrib["relation_type"]}"/>{escape_xml(node.attrib["token"])}<sense_id value="NO"/>'
    elif node.tag.lower() == "negation":
        linearized += '<negation/>'
    else:
        raise ValueError("Unrecognized XML tag")

    return linearized


def set_varnames(xml: ET.ElementBase):
    var_counter = Counter()
    for node in xml.xpath("//*[@varname]/term"):  # TODO: check that xpath traversal is depth-first, otherwise use penman reset variables
        varname = _default_variable_prefix(node.attrib["token"])
        var_counter[varname] += 1
        node.getparent().attrib["varname"] = varname if var_counter[varname] == 1 else f"{varname}{var_counter[varname]}"

    for node in xml.xpath("//termref"):
        refs = xml.xpath(f'//rel[@ref="{node.attrib["ref"]}"]|//tree[@value="{node.attrib["ref"]}"]')
        if len(refs) > 1:
            raise ValueError("XML malformed: there are multiple rel/tree nodes with the same ref attribute. These must be unique")
        elif refs:
            node.attrib["varname"] = refs[0].attrib["varname"]

    return xml


def linearize_to_xml(linearized_amr: str):
    # TODO: if an error occurs here during parsing, try to fix the tree
    # Maybe with BS4, which is supposedly more lenient (?)
    xml = ET.fromstring(linearized_amr)

    def _iterate(node, parent_tag: str = None, prev_sibling=None):
        xml_str = ""
        prev_tail = None if prev_sibling is None or not prev_sibling.tail else prev_sibling.tail.strip()

        if node.tag.lower() == "termid":
            if parent_tag == "tree":
                xml_str += f'<tree type="intermediate_xml" value="{node.attrib["value"]}" varname="" relation_type="ROOT">'
            elif parent_tag == "rel":
                if prev_sibling is not None and prev_sibling.tag.lower() == "reltype":  # relation_type might be empty if malformed
                    xml_str += f'<rel ref="{node.attrib["value"]}" varname="" relation_type="{prev_sibling.attrib["value"]}">'
                else:
                    xml_str += f'<rel ref="{node.attrib["value"]}" varname="" relation_type="">'

        elif node.tag.lower() == "sense_id":
            token_str = escape_xml(prev_tail) if prev_tail else ""  # token might be empty if malformed
            if prev_sibling.tag.lower() == "reltype":  # termrels
                xml_str += f'<termrel token="{token_str}" relation_type="{prev_sibling.attrib["value"]}"/>'
            else:  # terminals
                xml_str += f'<term token="{token_str}" sense_id="{node.attrib["value"]}"/>'

        elif node.tag.lower() == "termref":
            if prev_sibling is not None and prev_sibling.tag.lower() == "reltype":
                xml_str += f'<termref ref="{node.attrib["value"]}" varname="" relation_type="{prev_sibling.attrib["value"]}"/>'
            else:  # relation_type might be empty if malformed
                xml_str += f'<termref ref="{node.attrib["value"]}" varname="" relation_type=""/>'
        elif node.tag.lower() == "negation":
            xml_str += '<negation/>'

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


def is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


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

    @property
    def xml(self):
        return self.penman_tree2xml()

    @property
    def linearized(self):
        return xml_to_linearized(self.xml)

    @property
    def penman_str(self):
        return penman.format(self.penman_tree)

    def pprint_xml(self):
        xml_str = ET.tostring(self.xml, pretty_print=True, encoding="utf-8")
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
            xml_str += f'<tree type="intermediate_xml" value="{self.variable_to_ridx[parent_node.node[0]]}" varname="{escape_xml(fix_text(parent_node.node[0]))}" relation_type="ROOT">'

        if is_atomic(parent_node):  # Terminals
            if descriptor == "/":  # Instances
                if re.match(r"\S+-\d{2,}", parent_node):  # Match on "throw-up-01" but not on "even-if"
                    node_name, sense_id = parent_node.rsplit("-", 1)
                    xml_str += f'<term token="{escape_xml(fix_text(node_name))}" sense_id="{sense_id}"/>'
                else:  # Without explicit sense-id
                    xml_str += f'<term token="{escape_xml(fix_text(parent_node))}" sense_id="NO"/>'
            else:
                if descriptor.startswith(":ARG"):  # TODO: here we make some if/elif distinctions but their realization is the same. We may want to make that different in the future
                    if parent_node.isdigit():  # In quantitative relationships, where `:ARG1 2` indicates that something is.happens x2
                        xml_str += f'<termrel token="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                    elif parent_node in ["-", "+"]:  # In polarity relationships of "have-polarity", `:ARG2 -` means negative, or in be-polite relationsships `:ARG2 +`
                        xml_str += f'<termrel token="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                    elif parent_node.startswith('"') and parent_node.endswith('"'):  # In certain quantities, e.g., `:ARG3 "2/3"`
                        xml_str += f'<termrel token="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                    elif is_number(parent_node):  # In certain quantities, e.g., `:ARG2 290.19`
                        xml_str += f'<termrel token="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                    else:
                        # References to other variables with core roles
                        self.set_variable_ridx(parent_node)
                        if descriptor == ":wiki":
                            raise ValueError("Wiki items are currently not supported")
                        xml_str += f'<termref ref="{self.variable_to_ridx[parent_node]}" varname="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                elif descriptor == ":polarity":
                    if parent_node == "-":
                        xml_str += '<negation />'
                    else:
                        raise ValueError(f"Unexpected value '{parent_node}' for :polarity")
                else:  # Non-core roles
                    if re.match(r"^[a-z]\d+$", parent_node):  # Reference to other variable
                        self.set_variable_ridx(parent_node)
                        xml_str += f'<termref ref="{self.variable_to_ridx[parent_node]}" varname="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'
                    else:  # A terminal token which is also a some special relation (e.g. `:op1 100000`)
                        # NOTE: it _seems_ to me that this can only occur for numbers. A string will always be a token "instance" (?)
                        xml_str += f'<termrel token="{escape_xml(fix_text(parent_node))}" relation_type="{descriptor}"/>'

        else:  # Branches
            if not isinstance(parent_node, Tree):
                parent_node = Tree(parent_node)

            varname, branches = parent_node.node

            if descriptor is not None and varname is not None:
                self.set_variable_ridx(varname)
                xml_str += f'<rel ref="{self.variable_to_ridx[varname]}" varname="{escape_xml(fix_text(varname))}" relation_type="{descriptor}">'

            for descriptor, target in branches:
                xml_str += self.penman_tree2xml(target, descriptor, is_root=False)

            if descriptor is not None and varname is not None and not is_root:
                xml_str += "</rel>"

        if is_root:
            xml_str += "</tree>"
            xml = ET.fromstring(xml_str)
            # TODO: restructure this class so that this is not necessary.
            # Now this is necessary because we want to do escape/unescape and especially ftfy in the current method
            self.penman_tree = penman.parse(xml2penman_str(xml))
            return xml
        else:
            return xml_str

    @classmethod
    def from_linearized(cls, linearized_amr: str):
        xml = linearize_to_xml(linearized_amr)
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
