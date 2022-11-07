import re
from typing import Counter, Union, List
from collections import Counter

import penman
from penman import Tree
from penman.tree import is_atomic, _default_variable_prefix


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


# TODO: instead of dashes, maybe use ~~ for some things? E.g., instead of -91 we can use ~~91 to make it less likely to occur in the wild
# For instance -91, is in the dataset as a regular token as part of a phone number. So the special token should be different.


def do_remove_wiki(penman_str: str):
    """Remove all wiki entrires from a given penman string. These are the items that start with ':wiki' and
    have a value after it that is enclosed in double quotation marks '"'.

    :param penman_str: the given penman string
    :return: a string where all the wiki entries are removed
    """
    return re.sub(r'\s+:wiki\s+(?:\"[^\"]+\"|-)', "", penman_str)


def do_remove_metadata(penman_str: str):
    """Remove the metadata from a given penman string. These are the lines that start with '#'
    :param penman_str: the given penman string
    :return: a string where all the lines that start with '#' are removed
    """
    return re.sub(r'^#.*\n', "", penman_str, flags=re.MULTILINE)


def is_number(maybe_number_str: str) -> bool:
    """Check whether a given string is a number. We do not consider special cases such as 'infinity' and 'nan',
    which technically are floats. We do consider, however, floats like '1.23'.
    :param maybe_number_str: a string that might be a number
    :return: whether the given number is indeed a number
    """
    if maybe_number_str in ["infinity", "nan"]:
        return False

    try:
        float(maybe_number_str)
        return True
    except ValueError:
        return False


# Does not include :ref
ROLES = [':ARG', ':accompanier', ':age', ':beneficiary', ':calendar', ':century', ':concession', ':condition', ':conj-',
         ':consist-of', ':day', ':dayperiod', ':decade', ':degree', ':destination', ':direction', ':domain',
         ':duration', ':endrel', ':era', ':example', ':extent', ':frequency', ':instrument', ':li', ':location',
         ':manner', ':medium', ':mod', ':mode', ':month', ':name', ':negation', ':op', ':ord', ':part', ':path',
         ':polarity', ':polite', ':poss', ':prep-', ':purpose', ':quant', ':quarter', ':range', ':scale', ':season',
         ':sense', ':snt', ':source', ':startrel', ':subevent', ':subset', ':term', ':time', ':timezone',
         ':topic', ':unit', ':value', ':weekday', ':wiki', ':year', ':year2']


def tokenize_except_quotes(input_str: str) -> List[str]:
    """Split a given string into tokens by white-space EXCEPT for the tokens within quotation marks, do not split those.
    E.g.: `"25 bis"` is one token. This is important to ensure that all special values that are enclosed in double
    quotation marks are also considered as a single token.

    :param input_str: string to tokenize
    :return: a list of tokens
    """
    tokens = []
    tmp_str = ""
    quoted_started = False

    for char in input_str:
        is_quote = char == '"'
        if not tmp_str:
            tmp_str += char
            quoted_started = is_quote
        else:
            if quoted_started:
                if is_quote:
                    tmp_str += char
                    tokens.append(tmp_str.strip())
                    tmp_str = ""
                    quoted_started = False
                else:
                    tmp_str += char
            else:
                if char.isspace():
                    tokens.append(tmp_str.strip())
                    tmp_str = ""
                else:
                    tmp_str += char

                if is_quote:
                    quoted_started = True

    tokens.append(tmp_str.strip())
    return tokens


def penmanstr2linearized(penman_str: str, remove_wiki: bool = False, remove_metadata: bool = False) -> str:
    """Linearize a given penman string. Optionally remove the wiki items and/or metadata.

    :param penman_str: the penman string to linearize
    :param remove_wiki: whether to remove wiki entries
    :param remove_metadata: whether to remove metadata
    :return: a linearized string of the given penman string
    """
    if remove_wiki:
        penman_str = do_remove_wiki(penman_str)
    if remove_metadata:
        penman_str = do_remove_metadata(penman_str)

    tree = penman.parse(penman_str)

    return penmantree2linearized(tree)


def penmantree2linearized(penman_tree: Tree) -> str:
    """Linearize a given penman Tree.

    :param penman_tree: a penman Tree
    :return: a linearized string of the given penman Tree
    """
    tokens = []
    # A dictionary to keep track of references, i.e. variables to a generalizable ":refX"
    # Looks like: `{"f2": ":ref2"}`
    references = {}

    def _maybe_add_reference(varname: str):
        """Add a varname to `references` if it is not already in there.
        :param varname: the varname to add, e.g. "d" (for dog) or "f2" for "fight
        """
        if varname not in references:
            if references:
                max_refs = max([int(r.replace(":ref", "")) for r in references.values()])
            else:
                max_refs = 0
            references[varname] = f":ref{max_refs + 1}"

    def _iterate(node, is_instance_type: bool = False):
        """Method to recursively traverse the given penman Tree and while doing so adding new tokens to `tokens`.

        :param node: a Tree or node in a tree
        :param is_instance_type: whether this given node has an instance relation type (/) with its parent
        """
        nonlocal tokens, references
        if is_atomic(node):  # Terminals, node is the token
            # This is_instance_type is explicitly necessary because in some cases, the regex ^[a-z]\d+$
            # below will also match on very rare cases where the match, e.g. `f4`, is not a reference but a
            # real token, e.g. `f / f4`
            if is_instance_type:
                # Token+sense_id. The last condition is to make sure that we do not catch wiki's, too, which may
                # look like that, e.g., "Russian_submarine_Kursk_(K-141)"
                if (match := re.match(r"(\S+)-(\d{2,})", node)) and not (node.startswith('"') and node.endswith('"')):
                    # Special frames. TODO: check explicitly for all potential frames because sense 91 might exist?
                    if match.group(2) == "91":
                        tokens.append(node)
                    else:
                        tokens.extend((match.group(1), f":sense{match.group(2)}"))
                else:
                    tokens.append(node)
            elif re.match(r"^[a-z]\d+$", node):  # In case a terminal refers to another token
                _maybe_add_reference(node)
                tokens.append(references[node])
            # Special "literal" tokens do not have senses. These occur in e.g. :op or :wiki
            elif node.startswith('"') and node.endswith('"'):
                tokens.append(node)
            else:
                tokens.append(node)
        else:
            tokens.append(":startrel")
            if not isinstance(node, Tree):
                node = Tree(node)
            # Varname is e.g. "d" for dog or "d2" for dragon
            # Branches are "child nodes"
            varname, branches = node.node

            _maybe_add_reference(varname)
            tokens.append(references[varname])

            for relation_type, targetnode in branches:
                # We add a special token for negative polarity because it is a strong semantic cue
                if relation_type == ":polarity" and targetnode == "-":
                    tokens.append(":negation")
                    continue

                if relation_type != "/":
                    tokens.append(relation_type)

                _iterate(targetnode, relation_type == "/")

            tokens.append(":endrel")

    _iterate(penman_tree)

    # Remove references that only occur once
    # Every token "occurs" at least once (itself) but if no other occurrences -> remove
    refs_to_keep = sorted([r for r in references.values() if tokens.count(r) > 1],
                          key=lambda x: int(x.replace(":ref", "")))
    tokens = [token for token in tokens if token in refs_to_keep or not token.startswith(":ref")]

    # Re-number references so that the ones that are kept are numbered sequentially starting from :ref1.
    for ref_idx, ref in enumerate(refs_to_keep, 1):
        tokens = [f":ref{ref_idx}" if token == ref else token for token in tokens]

    return " ".join(tokens)


def linearized2penmanstr(tokens: Union[str, List[str]]) -> str:
    """Turn a linearized string or a list of linearized tokens into a penman string representation.
    :param tokens: a linearized string, or a list of tokens. If a string, we will tokenize it automatically
    :return: a penman string
    """
    if isinstance(tokens, str):
        tokens = tokenize_except_quotes(tokens)

    varcounter = Counter()  # To keep track of how often a var occurs so we can decide naming, e.g. dog -> d or d2?
    processed_tokens = set()  # Because we iteratively process tokens, we have to track which tokens we already did

    penman_tokens = []

    def _iterate(first_index: int = 0, level: int = 0):
        """Iterate over all tokens, starting from a first_index. Whenever we encounter :startrel, we call _iterate
        again. That allows us to have more control over how to generate relationships in a recursive, tree-like manner.

        In hopes of making this clearer, here is a visualization of the processing of tokens A rel( B ) C depth-first.
        First we encounter and process token A, then we encounter an opening rel token `srel` (actually `:startrel`).
        Therefore, we will enter a new `_iterate` function, and find and process B. After B, we encounter a `:endrel`
        token to end the relationship (here `erel`), and therefore break out of this _iterate function. That leads us
        back to the initial _iterate function, where then continue with the next token in that loop, which would be B!
        It would be B because this initial _iterate function had not seen B yet, as we had moved from srel into another
        _iterate call. So now that we are back, we need to skip B (because it was processed in the sub call) and move
        to processing C.


        A---> srel ---> B (skip) -----> erel (skip) ---> C
               ↓                          ↑
               -------> B ------------> erel (break)


        :param first_index: the index to start this iteration from. So we started at tokens[first_index]
        :param level: the "depth" that we are in in the tree. Used for indentation
        """
        nonlocal penman_tokens
        indent = "\t" * level
        # Iterate from a given start index to the end of all tokens but we can break out if we encounter an :endrel
        for token_idx in range(first_index, len(tokens)):
            # Outer loops should not process tokens that were processed in internal loops
            if token_idx in processed_tokens:
                continue
            processed_tokens.add(token_idx)

            # SET prev, next, current token values
            token = tokens[token_idx]

            prev_token = tokens[token_idx - 1] if token_idx > 0 and not first_index == token_idx else None
            next_token = None

            if token_idx < len(tokens) - 1:
                # A :ref can be inbetween tokens, but we can ignore those
                if tokens[token_idx + 1].startswith(":ref"):
                    next_token = tokens[token_idx + 2]
                else:
                    next_token = tokens[token_idx + 1]

            # Process each token
            if token == ":startrel":
                varname = ""
                if next_token:  # Can be false if the string is malformed
                    varname = _default_variable_prefix(next_token)
                    varcounter[varname] += 1
                    varname = varname if varcounter[varname] < 2 else f"{varname}{varcounter[varname]}"
                penman_tokens.append("(" + varname)
                _iterate(token_idx + 1, level + 1)
                penman_tokens.append(")")

            # End of a relation
            elif token == ":endrel":  # Stop processing because we have now processed a whole :startrel -> :endrel chunk
                break
            # Handle the special token :negation, which indicate s negative polarity
            elif token == ":negation":
                penman_tokens.append(":polarity -")
            # SENSE IDs: add the sense to the previous token
            elif match := re.match(r"^:sense(.+)", token):
                penman_tokens[-1] = f"{penman_tokens[-1]}-{match.group(1)}"
            # ROLES (that are not :refs)
            elif token.startswith(tuple(ROLES)):
                penman_tokens.append(f"\n{indent}{token}")
            # REFS
            elif token.startswith(":ref"):
                if prev_token and prev_token.startswith(":"):  # This is a reference to another token
                    penman_tokens.append(token)
                else:  # This is the original token that others refer to
                    penman_tokens.append(f"{token}-canonicalref")
            else:  # TOKENS: many exceptions where we do not need the instance / separator
                if token.isdigit() or is_number(token):  # Numbers are mostly dealt with without /
                    penman_tokens.append(f" {token}")
                # Certain quantities, wiki entries.... E.g. "2/3"
                elif token.startswith('"') and token.endswith('"'):
                    penman_tokens.append(f" {token}")
                # Many special roles have special values like "-" or numbers. E.g. `polarity -`, `:value 1`
                elif prev_token is not None and prev_token in [":polarity", ":mode", ":mod", ":polite", ":value",
                                                               ":quant"]:
                    penman_tokens.append(f" {token}")
                # Wikis do not have an instance relation. Empty wikis look like `:wiki -`
                elif prev_token is not None and prev_token in [":wiki"]:
                    penman_tokens.append(f" {token}")
                # Exceptionally, ARG can be unspecified - or +
                elif token in ["-", "+"] and prev_token is not None and prev_token.startswith((":ARG")):
                    penman_tokens.append(f" {token}")
                elif prev_token is not None and prev_token.startswith(":ref"):
                    penman_tokens.append(f"/ {token}")
                # Variable names. If the previous token starts with a (, then that prev token is the variable
                # and the current token is the actual token. This can be important for one-word tokens, e.g., (i / i)
                elif re.match(r"^[a-z]\d*$", token) and (prev_token is not None and not prev_token.startswith("(")):
                    penman_tokens.append(f" {token}")
                else:
                    penman_tokens.append(f"/ {token}")

    _iterate()

    # Link references: first get all unique references and the accompanying canonicalrefs
    # Then, for every canonical reference, find the variable name that is associated with it
    all_refs = set([t.replace("-canonicalref", "") for t in penman_tokens if t.startswith(":ref")])
    canon_refs = [f"{t}-canonicalref" for t in all_refs]
    ref2varname = {}

    for canon_ref in canon_refs:
        idx = penman_tokens.index(canon_ref)
        # The opening bracket is attached to the token, so remove that
        prev_token = penman_tokens[idx - 1].replace("(", "")
        ref2varname[canon_ref.replace("-canonicalref", "")] = prev_token

    # Replace :ref tokens with the found varnames for that token, and remove the canonical ref tokens
    penman_tokens = [ref2varname[t] if t in all_refs else t for t in penman_tokens if
                     not t.endswith("-canonicalref")]

    return " ".join(penman_tokens)


def linearized2penmantree(tokens: Union[str, List[str]]) -> Tree:
    """Turn a linearized representation back into a penman Tree

    :param tokens: a linearized string, or a list of tokens. If a string, we will tokenize it automatically
    :return: a penman Tree based on the linearized string or tokens
    """
    penman_str = linearized2penmanstr(tokens)
    return penman.parse(penman_str)


test_str = """
# ::annotator SDL-AMR-09
# ::date 2012-12-23T19:59:16
# ::id bolt12_64545_0529.2
# ::snt What is more they are considered traitors of China, which is a fact of cultural tyranny in the cloak of nationalism and patriotism.
# ::file bolt12_64545_0529_2.txt
# ::save-date Sun Dec 8, 2013
(c / consider-01
   :ARG1 (p / person
            :domain (t / they)
            :ARG0-of (b / betray-01
                        :ARG1 (c2 / country
                                  :wiki "China"
                                  :name (n / name
                                           :op1 "China"))))
   :mod (m / more)
   :mod (t2 / tyrannize-01
            :ARG2 (c3 / culture)
            :ARG1-of (c4 / cloak-01
                         :ARG2 (a / and
                                  :op1 (n2 / nationalism)
                                  :op2 (p2 / patriotism)))))

"""

if __name__ == '__main__':
    tree = penman.parse(test_str)
    tree.reset_variables()
    linearized = penmantree2linearized(tree)
    print(linearized)