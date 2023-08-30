import re
from typing import List


def _is_url(text: str) -> bool:
    # Modified from django https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
    match = re.match(
        r"^(?:(?:http|ftp)s?://)?"  # http:// or https:// (bv: made this optional)
        r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
        r"localhost|"  # localhost...
        r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
        r"(?::\d+)?"  # optional port
        r"(?:/?|[/?]\S+)$",
        text,
        re.IGNORECASE,
    )

    return match is not None


def _is_email(text: str) -> bool:
    # Taken from https://stackoverflow.com/a/201378/1150683
    match = re.match(
        r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])',
        text,
        re.IGNORECASE,
    )

    return match is not None


def _is_filename(text: str) -> bool:
    # Taken from https://stackoverflow.com/a/201378/1150683
    match = re.match(r"^.*?\.[a-z0-9]+$", text, re.IGNORECASE)

    return match is not None


def postprocess_str_after_linearization(linearized: str, verbose: bool = False) -> str:
    linearized = linearized.replace(":polarity -", ":negation")
    linearized = re.sub(r"-of(?=\s|$)", " </of>", linearized)

    # Re-implementation of SPRING, which also replaces "_" with " "
    # https://github.com/SapienzaNLP/spring/blob/39079940d028ba0dde4c1af60432be49f67d76f8/spring_amr/tokenization_bart.py#L143-L144
    # But do not replace _ inside of URLs
    def replace_literal(match):
        content = match.group(1)
        if not (_is_url(content) or _is_email(content) or _is_filename(content)):
            content = content.replace("_", " ")
        return f"<lit> {content} </lit>"

    linearized = re.sub(r'"(.*?)"', replace_literal, linearized)
    if verbose:
        print("after repl lit", linearized)

    # Replace parentheses but only if they are not inside <lit>
    def replace_rel(match):
        return " <rel> " if match.group() == "(" else " </rel> "

    linearized = re.sub(r"(?<!<lit>)\((?![^<]*<\/lit>)|(?<!<lit>)\)(?![^<]*<\/lit>)", replace_rel, linearized)
    if verbose:
        print("after repl rel", linearized)

    # Remove duplicate spaces
    linearized = " ".join(linearized.split())

    return linearized


def postprocess_str_after_delinearization(delinearized: str) -> str:
    delinearized = (
        # AMR specific
        delinearized.replace(" -quantity", "-quantity")
        .replace(" -entity", "-entity")
        .replace(" </of>", "</of>")
    )

    # AMR specific
    # Generic prepositions/conjunctions, e.g. `:prep-by` or `:conj-as-if`
    delinearized = re.sub(r":(prep|conj)-\s+(\w+)", r":\1-\2", delinearized)
    delinearized = delinearized.replace(":negation", " :polarity - ")
    delinearized = delinearized.replace("</of>", "-of ")

    # Add spaces around :-roles like :op1
    # Include `-` for e.g. :prep-with
    delinearized = re.sub(r"(:[a-zA-Z][a-zA-Z0-9-]+)", r" \1 ", delinearized)

    # Glue role digits together, e.g. ':op1 0 <rel>' -> :op10 <rel>
    delinearized = re.sub(r"(:[a-zA-Z][a-zA-Z0-9]+)\s+(\d+)\s*<(rel|lit)>", r"\1\2 <\3>", delinearized)

    def reverse_literal(match):
        rel = match.group(1).strip()
        content = match.group(2).strip()
        if rel.startswith(("wiki", "op", "value")):
            # E.g. `<lit> Russian submarine Yury Dolgorukiy (K-53 5) </lit>` -> (K-535)
            # But will probably lead to quite some false positives
            # Luckily this will mostly occur inside wiki, which we mostly do not use
            content = re.sub(r"(\d+)\s+(\d+)", r"\1\2", content)
            #  <lit> Russian submarine Yury Dolgorukiy (K -535) </lit> -> K-535
            content = re.sub(r"\s+-\s*(\d+)", r"-\1", content)
            content = content.replace(" ", "_")

        return f':{rel} "{content}"'

    delinearized = re.sub(r":([a-zA-Z][a-zA-Z0-9]+)\s*<lit>(.*?)</lit>", reverse_literal, delinearized)

    # Glue numbers back together, e.g. ':quant -54 7' -> ':quant -547'
    # but should not trigger for literal values, like ':value "34 61 09 91 12 135"'
    # Should not consider glueing things back to roles (like `:op1 12 "hello"` -> `:op112`); that is dealt with earlier
    delinearized = re.sub(r"(?<![\"\D])(-?\d*\.?\d*)\s+(\d+)(?!\s*[\d\"<:])", r"\1\2", delinearized)

    # Add -of back to utterances for regular words (NOT for -of roles)
    # E.g., `:mod <rel> <pointer:4> first-of -all </rel>` -> `first-of-all </rel>`
    # E.g., `jet-of f-01 :ARG1` -> `jet-off-01 :ARG1`
    delinearized = re.sub(r"-of\s+([-a-zA-Z0-9]+)\s*(?=[<:])", r"-of\1 ", delinearized)

    def fix_dashes(match):
        # Glue together concepts with dashes that might have been split
        # E.g. `<pointer:6> take-into-ac count-04 :ARG0` -> `take-into-account-04`
        # E.g. `<pointer:3> have-degree -of -resemblance -91 :ARG1` -> `have-degree-of-resemblance`
        pointer = match.group(1).strip()
        content = re.sub(r"\s", "", match.group(2).strip())

        return f'{pointer} {content} '

    delinearized = re.sub(r"(<pointer:\d+>)\s*([-a-z\d\s]+)\s*(?=[<:])", fix_dashes, delinearized)

    delinearized = delinearized.replace("<rel>", " ( ")
    delinearized = delinearized.replace("</rel>", " ) ")
    # Remove duplicate spaces
    delinearized = " ".join(delinearized.split())

    return delinearized


def tokenize_except_quotes_and_angles(input_str: str) -> list[str]:
    """Split a given string into tokens by white-space EXCEPT for the tokens within quotation marks, do not split those.
    E.g.: `"25 bis"` is one token. This is important to ensure that all special values that are enclosed in double
    quotation marks are also considered as a single token. Also does not tokenize things inside tags like <rel>.

    <rel><pointer:0>end-01</rel> -> ['<rel>', '<pointer:0>', 'end-01', '</rel>']
    :param input_str: string to tokenize
    :return: a list of tokens
    """
    tokens = []
    tmp_str = ""
    quoted_started = False
    angled_started = False

    for char in input_str:
        is_quote = char == '"'
        is_open_angled = char == '<'
        is_close_angled = char == '>'
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
            elif angled_started:
                if is_close_angled:
                    tmp_str += char
                    tokens.append(tmp_str.strip())
                    tmp_str = ""
                    angled_started = False
                else:
                    tmp_str += char
            else:
                if char.isspace():
                    tokens.append(tmp_str.strip())
                    tmp_str = ""
                elif is_open_angled:
                    tokens.append(tmp_str.strip())
                    angled_started = True
                    tmp_str = "<"
                else:
                    tmp_str += char

                if is_quote:
                    quoted_started = True

    if tmp_str.strip():
        tokens.append(tmp_str.strip())
    return tokens
