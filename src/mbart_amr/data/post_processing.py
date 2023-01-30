from typing import List

from mbart_amr.data.linearization import (penmanstr2linearized,
                                          tokenize_except_quotes)
from mbart_amr.data.tokens import ROLE_PREFIXES


example = """and :op1 :startrel issue :sense01 :ARG0 :startrel :ref1 person :wiki "Jupiter_(mythology)" :name :startrel name :op1 "Jupiter" :endrel :endrel :ARG1 :startrel thing :ARG1~~of :startrel proclaim :sense01 :endrel :endrel :ARG2 :startrel beast :mod :startrel all :endrel :location :startrel forest :endrel :endrel :endrel :op2 :startrel promise :sense01 :ARG0 :ref1 :ARG2 :startrel reward :sense01 :ARG1 :startrel one :ARG0~~of :startrel have :sense03 :ARG1 :startrel offspring :ARG1~~of :startrel have-degree-91 :ARG2 :startrel handsome :endrel :ARG3 :startrel most :endrel :ARG1~~of :startrel deem :sense01 :endrel :endrel :endrel :endrel :endrel :mod :startrel royal :endrel :endrel :endrel
"""


def is_valid(tokens: List[str]):
    if tokens[0].startswith(ROLE_PREFIXES) and tokens[1] == ":startrel" and tokens[-1] == "endrel":
        if len(tokens) == 3:
            return True
        # Third token has to be a ref or NOT a role
        elif tokens[2].startswith(":ref") or not tokens[2].startswith(ROLE_PREFIXES):
            for idx, token in enumerate(tokens):
                # Tokens starting with " have to be preceded by wiki tokens and the other way around
                if (token.startswith('"') and not tokens[idx - 1] == ":wiki") or (
                    token == ":wiki" and not tokens[idx + 1].startswith('"')
                ):
                    return False

        else:
            return False
    else:
        return False

    return True


def post_process(linearized: str):
    # TODO: find a better way to rejoin fixed subtrees, because this is not straightforward with just
    # indexes. They are not linear (but trees) so need to find an alternative way. OO?
    tokens = tokenize_except_quotes(linearized)
    # Collect all indices of :startrel
    starts = [idx for idx, t in enumerate(tokens) if t == ":startrel"]

    fixed_tokens = []
    # Iterate over all starts (all subtrees)
    while starts:
        start_idx = starts.pop(-1)
        end_idx = next(idx for idx, t in enumerate(tokens[start_idx:], start_idx) if t == ":endrel")
        subtokens = tokens[start_idx - 1 : end_idx + 1]

        while not is_valid(subtokens):
            # Do stuff to subtokens to try and fix it
            pass

        fixed_tokens.append((start_idx, subtokens))
        del tokens[start_idx - 1 : end_idx + 1]


if __name__ == "__main__":
    post_process(linearized=example)
