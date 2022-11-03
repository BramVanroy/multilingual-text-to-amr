from pathlib import Path
from pprint import pprint

from tqdm import tqdm
import penman
import lxml.etree as ET

from amr_bart.data.new_linearization import Linearizer


def get_tags(tree):
    uniq_els = set()
    for node in tree.iter():
        if node.tag == "tree":
            uniq_els.update(['<tree type="linearized_xml">', "</tree>"])
        elif node.tag == "rel":
            uniq_els.update(["<rel>", "</rel>"])
        elif node.tag == "reltype":
            uniq_els.add(f'<reltype value="{node.attrib["value"]}"/>')
        elif node.tag == "sense_id":
            uniq_els.add(f'<sense_id value="{node.attrib["value"]}"/>')
        elif node.tag == "termid":
            uniq_els.add(f'<termid value="{node.attrib["value"]}"/>')
        elif node.tag == "termref":
            uniq_els.add(f'<termref value="{node.attrib["value"]}"/>')
        elif node.tag == "negation":
            uniq_els.add(f'<negation/>')

    return uniq_els


def main(din: str):
    pdin = Path(din)
    special_tags = set()
    for pfin in tqdm(list(pdin.rglob("*.txt")), unit="file"):
        with pfin.open(encoding="utf-8") as fhin:
            for tree in penman.iterparse(fhin):
                tree_str = penman.format(tree)
                linearizer = Linearizer.from_penman_str(tree_str)
                tree = ET.fromstring(linearizer.linearized)
                special_tags.update(get_tags(tree))

    pprint(sorted(special_tags))




if __name__ == "__main__":
    main(r"D:\corpora\amr_annotation_3.0\data\amrs_fixed")
