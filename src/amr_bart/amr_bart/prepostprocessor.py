import re
from typing import List


def linearized2inputstr(linearized: str) -> str:
    """
    <reltype value=":ARG0"/> -> :ARG0; if idx > 10, just keep it, e.g. :ARG12. The tokenizer will split it up
    <sense_id value="NO"/> -> :senseNO
    <sense_id value="12"/> -> :sense12; if idx > 100, just keep it, e.g. :sense128. The tokenizer will split it up
    <rel> -> :startrel
    </rel> -> :endrel
    <termid value="R28"/> -> :term28; if idx > 100, just keep it, e.g. :term128. The tokenizer will split it up
    <termref value="R28"/> -> :ref28; if idx > 100, just keep it, e.g. :ref128. The tokenizer will split it up
    <tree type="linearized_xml"> -> "" (removes, SOS is start of tree)
    </tree> -> "" (removes, SOS is start of tree)
    """

    linearized = linearized.replace('<tree type="linearized_xml">', "")
    linearized = linearized.replace('<tree>', "")  # Just in case the type attribute was not added
    linearized = linearized.replace('</tree>', "")

    linearized = re.sub(r'<reltype value="([^"]+)"\/>', "$1", linearized)
    linearized = re.sub(r'<sense_id value="([^"]+)"\/>', ":sense$1", linearized)

    linearized = linearized.replace("<rel>", ":startrel")
    linearized = linearized.replace("</rel>", ":endrel")

    linearized = re.sub(r'<termid value="R([^"]+)"\/>', ":term$1", linearized)
    linearized = re.sub(r'<termref value="R([^"]+)"\/>', ":ref$1", linearized)

    linearized = linearized.replace("<negation/>", ":negation")

    return linearized


def inputstr2linearized(inputstr: str) -> str:
    """
    Do the inverse of linearized2inputstr, e.g. :senseNO -> <sense_id value="NO"/>; add <tree>, etc.
    """
    # TODO: find a way to capture all possible relations. Probably first get all occurrences of r":\S+" and then
    # verify that they are in our list of new tokens. For some tokens, e.g. :ARG24, it might be that it is not explicitly
    # in our list so we will have to check for possibilities per-tag, .e.g. r":ARG\d+" etc.

    inputstr = re.sub(r':sense(NO|\d+)', '<sense_id value="$1"/>', inputstr)

    inputstr = inputstr.replace(":startrel", "<rel>")
    inputstr = inputstr.replace(":endrel", "</rel>")

    inputstr = re.sub(r":term(\d+)", r'<termid value="$1"/>', inputstr)
    inputstr = re.sub(r":ref(\d+)", r'<termref value="$1"/>', inputstr)

    inputstr = inputstr.replace(":negation", "<negation/>")

    inputstr = '<tree type="linearized_xml">' + inputstr + '</tree>'

    return inputstr

def postprocess_tokenids(ids: List[int]) -> List[int]:
    """Postprocess tokenids in such a way that the final result (after tokens2linearized) must be a valid tree"""
    pass
