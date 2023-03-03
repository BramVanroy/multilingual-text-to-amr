from collections import defaultdict
from pathlib import Path
from pprint import pprint

from mbart_amr.data.linearization import tokenize_except_quotes
from mbart_amr.data.tokens import *


def main(fin: str):
    pfin = Path(fin)

    tokens_d = defaultdict(set)
    with pfin.open(encoding="utf-8") as fhin:
        for line in fhin:
            line = line.strip()
            if not line:
                continue

            tokens = tokenize_except_quotes(line)

            for idx in range(len(tokens)):
                token = tokens[idx]
                if token.startswith(TOKENS_TO_ADD) or token.endswith((FRAME_91_ID, OF_SUFFIX) + SPECIAL_SUFFIXES):
                    next_token = tokens[idx + 1] if idx < len(tokens) - 1 else None

                    if next_token and next_token.startswith(TOKENS_TO_ADD):
                        tokens_d[token].add(next_token)

        tokens_d = dict(tokens_d)

    role_follow_roles = defaultdict(set)
    for token, follow_tokens in tokens_d.items():
        token_role = None
        for role in TOKENS_TO_ADD:
            if token.startswith(role):
                token_role = role
                break

        if not token_role:
            continue

        if token_role != ":year2":
            token_role = re.sub(r"\d+$", "", token_role)

        for follow_token in follow_tokens:
            for role in TOKENS_TO_ADD:
                if follow_token.startswith(role):
                    if role != ":year2":
                        role = re.sub(r"\d+$", "", role)
                    role_follow_roles[token_role].add(role)
                    break

    role_follow_roles = {k: tuple(v) for k, v in role_follow_roles.items()}
    pprint(role_follow_roles)


if __name__ == "__main__":
    main(r"F:\python\multilingual-text-to-amr\all-linearized.txt")
