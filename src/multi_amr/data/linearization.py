import re
from collections import Counter
from typing import List, Union

import penman
from multi_amr.data.tokens import ENDLIT, ENDREL, OF_SUFFIX, PREP_PREFIX, STARTSWITH_ROLES, STARTLIT, STARTREL
from multi_amr.utils import is_number
from penman import Tree
from penman.tree import _default_variable_prefix, is_atomic


def do_remove_wiki(penman_str: str):
    """Remove all wiki entrires from a given penman string. These are the items that start with ':wiki' and
    have a value after it that is enclosed in double quotation marks '"'.
    TODO: in spring, they replace the wiki value with "+". Should we do that too?

    :param penman_str: the given penman string
    :return: a string where all the wiki entries are removed
    """
    return re.sub(r"\s+:wiki\s+(?:\"[^\"]+\"|-)", "", penman_str)


def do_remove_metadata(penman_str: str):
    """Remove the metadata from a given penman string. These are the lines that start with '#'
    :param penman_str: the given penman string
    :return: a string where all the lines that start with '#' are removed
    """
    return re.sub(r"^#.*\n", "", penman_str, flags=re.MULTILINE)


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


def replace_of(relation_token: str, reverse: bool = False):
    """We want to use -of as a generic token because all roles can be inversed with the addition of -of, e.g.,
    :ARG0-of. However, `-of` also often occurs in other strings that we do not want to mess with, e.g. `jet-off-01`.
    To avoid these issues, we replace roles that end with `-of` with `~~of` (except for `:consist-of`, which stands
    on its own). Using `reverse=True` replaces `~~of` with `-of` again.

    :param relation_token: token to replace
    :param reverse: if true, turns replaces ~~of with -of instead of the other way around
    :return: the potentially modified token
    """
    if reverse:
        if relation_token.startswith(":") and relation_token.endswith(OF_SUFFIX):
            return relation_token.replace(OF_SUFFIX, "-of")
    else:
        if relation_token.startswith(":") and relation_token.endswith("-of") and relation_token != ":consist-of":
            return relation_token.replace("-of", OF_SUFFIX)
    return relation_token


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
                tokens.append(node)
            elif re.match(r"^[a-z]\d*$", node):  # In case a terminal refers to another token
                _maybe_add_reference(node)
                tokens.append(references[node])
            # Special "literal" tokens that occur in e.g. :op or :wiki
            elif node.startswith('"') and node.endswith('"'):
                tokens.extend([STARTLIT, node[1:-1], ENDLIT])
            else:
                tokens.append(node)
        else:
            tokens.append(STARTREL)
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
                    tokens.append(replace_of(relation_type))

                _iterate(targetnode, relation_type == "/")

            tokens.append(ENDREL)

    _iterate(penman_tree)

    # Remove opening and closing :startrel :endrel tokens, because we do not _really_ need them
    # and they take up space in the sequences
    tokens = tokens[1:-1]

    # Remove references that only occur once
    # Every token "occurs" at least once (itself) but if no other occurrences -> remove
    refs_to_keep = sorted(
        [r for r in references.values() if tokens.count(r) > 1], key=lambda x: int(x.replace(":ref", ""))
    )
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

    # Remove any empty tokens, just to be sure
    tokens = [t.strip() for t in tokens if t.strip() != ""]

    # It is expected that the first token is not :startrel, because in linearlization we
    # explicitly remove first and last :start/endrels because they take up space in the sequence
    # So here we can add them again if needed
    if tokens[0] != STARTREL:
        tokens = [STARTREL] + tokens + [ENDREL]

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
        lit_open = False
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
            if token == STARTREL:
                varname = ""
                if next_token:  # Can be false if the string is malformed
                    varname = _default_variable_prefix(next_token)
                    varcounter[varname] += 1
                    varname = varname if varcounter[varname] < 2 else f"{varname}{varcounter[varname]}"
                penman_tokens.append("(" + varname)
                _iterate(token_idx + 1, level + 1)
                penman_tokens.append(")")

            # End of a relation
            elif token == ENDREL:  # Stop processing because we have now processed a whole :startrel -> :endrel chunk
                break
            elif token == STARTLIT:
                lit_open = True
                penman_tokens.append('"')
            elif token == ENDLIT:
                lit_open = False
                penman_tokens[-1] = f'{penman_tokens[-1]}"'  # just add a closing quote
            # Handle the special token :negation, which indicate s negative polarity
            elif token == ":negation":
                penman_tokens.append(":polarity -")
            # STARTSWITH_ROLES (that are not :refs)
            elif (token.startswith(STARTSWITH_ROLES) and not token.startswith(":ref")) or token.startswith(
                PREP_PREFIX
            ):
                penman_tokens.append(f"\n{indent}{replace_of(token, reverse=True)}")
            # REFS
            elif token.startswith(":ref"):
                if prev_token and prev_token.startswith(":"):  # This is a reference to another token
                    penman_tokens.append(token)
                else:  # This is the original token that others refer to
                    penman_tokens.append(f"{token}-canonicalref")
            else:  # TOKENS: many exceptions where we do not need the instance / separator
                if lit_open:  # Literals
                    if penman_tokens[-1] == '"':  # If this is the first token after the opening quote, no space needed
                        penman_tokens[-1] = f"{penman_tokens[-1]}{token}"
                    else:
                        penman_tokens[-1] = f"{penman_tokens[-1]} {token}"
                elif token.isdigit() or is_number(token):  # Numbers are mostly dealt with without /
                    penman_tokens.append(token)
                # Certain quantities, wiki entries.... E.g. "2/3"
                elif token.startswith('"') and token.endswith('"'):
                    penman_tokens.append(token)
                # Many special roles have special values like "-" or numbers. E.g. `polarity -`, `:value 1`
                # that do not need an instance /
                elif prev_token is not None and prev_token in [
                    ":polarity",
                    ":mode",
                    ":mod",
                    ":polite",
                    ":value",
                    ":quant",
                    ":wiki",
                ]:
                    penman_tokens.append(token)
                # OPs do not have an instance relation
                elif prev_token is not None and prev_token.startswith(":op"):
                    penman_tokens.append(token)
                # Exceptionally, ARG can be unspecified - or +
                elif token in ["-", "+"] and prev_token is not None and prev_token.startswith(":ARG"):
                    penman_tokens.append(token)
                elif prev_token is not None and prev_token.startswith(":ref"):
                    penman_tokens.append(f"/ {token}")
                # Variable names. If the previous token starts with a (, then that prev token is the variable
                # and the current token is the actual token. This can be important for one-word tokens, e.g., (i / i)
                elif re.match(r"^[a-z]\d*$", token) and (prev_token is not None and not prev_token.startswith("(")):
                    penman_tokens.append(token)
                else:
                    penman_tokens.append(f"/ {token}")

    _iterate()

    # Link references: first get all unique references and the accompanying canonicalrefs
    # Then, for every canonical reference, find the variable name that is associated with it
    canon_refs = set([t for t in penman_tokens if t.startswith(":ref") and t.endswith("-canonicalref")])
    tokens_with_refs = set([canon.replace("-canonicalref", "") for canon in canon_refs])
    ref2varname = {}

    # For each canonical reference, get the corresponding token
    for canon_ref in canon_refs:
        idx = penman_tokens.index(canon_ref)

        # The opening bracket is attached to the token, so remove that
        varname = penman_tokens[idx - 1].replace("(", "")
        ref2varname[canon_ref.replace("-canonicalref", "")] = varname

    # Replace :ref tokens with the found varnames for that token, and remove the canonical ref tokens
    penman_tokens = [
        ref2varname[t] if t in tokens_with_refs else t for t in penman_tokens if not t.endswith("-canonicalref")
    ]

    return " ".join(penman_tokens)


def linearized2penmantree(tokens: Union[str, List[str]]) -> Tree:
    """Turn a linearized representation back into a penman Tree

    :param tokens: a linearized string, or a list of tokens. If a string, we will tokenize it automatically
    :return: a penman Tree based on the linearized string or tokens
    """
    penman_str = linearized2penmanstr(tokens)
    return penman.parse(penman_str)
