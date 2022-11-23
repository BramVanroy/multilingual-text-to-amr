import re
from collections import Counter
from typing import Counter, List, Union

import penman
from penman import Tree
from penman.tree import _default_variable_prefix, is_atomic

from mbart_amr.data.tokens import ROLE_PREFIXES


def do_remove_wiki(penman_str: str):
    """Remove all wiki entrires from a given penman string. These are the items that start with ':wiki' and
    have a value after it that is enclosed in double quotation marks '"'.
    TODO: in spring, they replace the wiki value with "+". Should we do that too? https://github.com/BramVanroy/spring/blob/main/spring_amr/penman.py

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
        if relation_token.startswith(":") and relation_token.endswith("~~of"):
            return relation_token.replace("~~of", "-of")
    else:
        if relation_token.startswith(":") and relation_token.endswith("-of") and relation_token != ":consist-of":
            return relation_token.replace("-of", "~~of")
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
            elif re.match(r"^[a-z]\d*$", node):  # In case a terminal refers to another token
                _maybe_add_reference(node)
                tokens.append(references[node])
            # Special "literal" tokens do not have senses. These occur in e.g. :op or :wiki
            elif node.startswith('"') and node.endswith('"'):
                tokens.append(node)
            else:
                tokens.append(node)
        else:
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

            tokens.append(":endrel")

    _iterate(penman_tree)

    tokens = tokens[:-1]  # Remove final :endrel (which closes the tree), which we do not need
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

    varcounter = Counter()  # To keep track of how often a var occurs so we can decide naming, e.g. dog -> d or d2?
    processed_tokens = set()  # Because we iteratively process tokens, we have to track which tokens we already did

    penman_tokens = []

    def _iterate(first_index: int = 0, level: int = 0):
        """Iterate over all tokens, starting from a first_index. Whenever we encounter a ROLE, we call _iterate
        again. That allows us to have more control over how to generate relationships in a recursive, tree-like manner.

        In hopes of making this clearer, here is a visualization of the processing of tokens A rel( B ) C depth-first.
        First we encounter and process token A, then we encounter a ROLE token.
        Therefore, we will enter a new `_iterate` function, and find and process B. After B, we encounter a `:endrel`
        token to end the relationship (here `erel`), and therefore break out of this _iterate function. That leads us
        back to the initial _iterate function, where then continue with the next token in that loop, which would be B!
        It would be B because this initial _iterate function had not seen B yet, as we had moved from ROLE into another
        _iterate call. So now that we are back, we need to skip B (because it was processed in the sub call) and move
        to processing C.


        A---> :ROLE ---> B (skip) -----> erel (skip) ---> C
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
                next_token = tokens[token_idx + 1]

            # Indicates whether there is a next token that does not start with :
            # There can be a :ref token in between this next token and the current token though!
            # So "next_not_ref_token" can be either the next token (+1) or the token next to that! (+2)
            # Valid cases:
            # :ARG0 :ref1 we: next token ref, token next to that does not start with :
            # :purpose drink :sense01: next token not a ref, but it also does not start with :
            next_not_ref_token = None
            # A :ref can be inbetween tokens, but we can ignore those
            if token_idx < len(tokens) - 2 and tokens[token_idx + 1].startswith(":ref") and not tokens[token_idx + 2].startswith(":"):
                next_not_ref_token = tokens[token_idx + 2]
            elif token_idx < len(tokens) - 1 and not tokens[token_idx + 1].startswith(":"):
                next_not_ref_token = tokens[token_idx + 1]

            # Process each token
            if token.startswith(ROLE_PREFIXES):
                penman_tokens.append(f"\n{indent}\t{replace_of(token, reverse=True)}")

                # Exceptions in which case we do not need to start a new branch
                if next_token:
                    if next_token.isdigit() or is_number(next_token):  # Numbers are mostly dealt with without /
                        continue
                    # Certain quantities, wiki entries.... E.g. "2/3"
                    elif next_token.startswith('"') and next_token.endswith('"'):
                        continue
                    # Mode is always a single token, e.g. ":mode imperative"
                    elif token == ":mode":
                        continue
                    # Wikis do not have an instance relation. Empty wikis look like `:wiki -`
                    elif token == ":wiki":
                        continue
                    # Exceptionally, ARG can be unspecified - or +
                    elif next_token in ("-", "+"):
                        continue

                # Add a branch when next_not_ref_token is not None, so either it is the next token or the token after
                # that, depending on whether:
                # - the next token is a ref, in which case the token after that can be selected but ONLY IF:
                #       - the token after that (so current token + 2) does not start with :, e.g. ":ARG0 :ref1 we"
                # - if the next token does not start with :, that is also a new branch (e.g. ":purpose drink :sense01")
                if next_not_ref_token:
                    penman_tokens.append("(")
                    _iterate(token_idx + 1, level + 1)
                    penman_tokens.append(")")

            # End of a relation
            elif (
                token == ":endrel"
            ):  # Stop processing because we have now processed a whole :startrel -> :endrel chunk
                break
            # Handle the special token :negation, which indicate s negative polarity
            elif token == ":negation":
                penman_tokens.append(":polarity -")
            # SENSE IDs: add the sense to the previous token
            elif match := re.match(r"^:sense(.+)", token):
                penman_tokens[-1] = f"{penman_tokens[-1]}-{match.group(1)}"
            # REFS
            elif token.startswith(":ref"):
                if next_token and next_token.startswith(":") or not next_token:  # This is a reference to another token
                    penman_tokens.append(token)
                else:  # This is the original token that others refer to
                    penman_tokens.append(f"{token}-canonicalref")
            else:  # TOKENS: many exceptions where we do not need the instance / separator
                if token.isdigit() or is_number(token):  # Numbers are mostly dealt with without /
                    penman_tokens.append(f" {token}")
                # Certain quantities, wiki entries.... E.g. "2/3"
                elif token.startswith('"') and token.endswith('"'):
                    penman_tokens.append(f" {token}")
                # Mode is always a single token, e.g. ":mode imperative"
                elif prev_token is not None and prev_token == ":mode":
                    penman_tokens.append(f" {token}")
                # Wikis do not have an instance relation. Empty wikis look like `:wiki -`
                elif prev_token is not None and prev_token == ":wiki":
                    penman_tokens.append(f" {token}")
                # Exceptionally, ARG can be unspecified - or +
                elif token in ("-", "+"):
                    penman_tokens.append(f" {token}")
                # Variable names. If the previous token is a ref, then that prev token is the variable
                # and the current token is the actual token. This can be important for one-word tokens, e.g., (i / i)
                elif re.match(r"^[a-z]\d*$", token) and (prev_token is not None and not prev_token.startswith(":ref")):
                    penman_tokens.append(f" {token}")
                else:  # A token and its reference
                    varname = _default_variable_prefix(token)
                    varcounter[varname] += 1
                    varname = varname if varcounter[varname] < 2 else f"{varname}{varcounter[varname]}"
                    penman_tokens.append(f"{varname} / {token}")

    _iterate()

    penman_tokens = ["("] + penman_tokens + [")"]  # Add tree opening/closing parens

    # Link references: first get all unique references and the accompanying canonicalrefs
    # Then, for every canonical reference, find the variable name that is associated with it
    canon_refs = set([t for t in penman_tokens if t.startswith(":ref") and t.endswith("-canonicalref")])
    tokens_with_refs = set([canon.replace("-canonicalref", "") for canon in canon_refs])
    ref2varname = {}

    # For each canonical reference, get the corresponding token
    for canon_ref in canon_refs:
        idx = penman_tokens.index(canon_ref)

        # The variable name is attached to the token that follows on the canonical reference,
        # E.g., [..., ':ref1-canonicalref', 'w / we', ...]. So we get the next token and extract the varname
        varname = penman_tokens[idx + 1].split("/")[0].strip()
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


