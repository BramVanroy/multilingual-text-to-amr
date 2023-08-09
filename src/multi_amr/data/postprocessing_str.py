import re
from typing import List

def _is_url(text: str) -> bool:
    # Modified from django https://github.com/django/django/blob/stable/1.3.x/django/core/validators.py#L45
    match = re.match(
        r'^(?:(?:http|ftp)s?://)?' # http:// or https:// (bv: made this optional)
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', text, re.IGNORECASE)

    return match is not None


def _is_email(text: str) -> bool:
    # Taken from https://stackoverflow.com/a/201378/1150683
    match = re.match(r'(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])', text, re.IGNORECASE)

    return match is not None

def _is_filename(text: str) -> bool:
    # Taken from https://stackoverflow.com/a/201378/1150683
    match = re.match(r"^.*?\.[a-z0-9]+$", text, re.IGNORECASE)

    return match is not None

def clean_up_amr_tokenization(out_string: str) -> str:
    """Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    :param out_string: the text to clean up
    :return: the cleaned-up string
    """
    out_string = (
        # AMR specific
        out_string.replace(" -quantity", "-quantity")
        .replace(" -entity", "-entity")
        .replace(" </of>", "</of>")
    )
    # Clean-up whitespaces before doing regexes (this is important!)
    out_string = " ".join(out_string.split())

    # AMR specific
    # Generic prepositions/conjunctions, e.g. `:prep-by` or `:conj-as-if`
    out_string = re.sub(r":(prep|conj)-\s+(\w+)", r":\1-\2", out_string)

    # Clean-up whitespaces
    out_string = " ".join(out_string.split())

    return out_string


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
    linearized = re.sub(r'(?<!<lit>)\((?![^<]*<\/lit>)|(?<!<lit>)\)(?![^<]*<\/lit>)', replace_rel, linearized)
    if verbose:
        print("after repl rel", linearized)

    # Remove duplicate spaces
    linearized = " ".join(linearized.split())

    return linearized


def postprocess_str_after_delinearization(delinearized: str) -> str:
    delinearized = delinearized.replace(":negation", ":polarity -")
    delinearized = delinearized.replace("</of>", "-of")

    # Glue role digits together, e.g. ':op1 0 <rel>' -> :op10 <rel>
    delinearized = re.sub(r"(:[a-zA-Z0-9]+)\s+(\d+) <(rel|lit)>", r"\1\2 <\3>", delinearized)

    def reverse_literal(match):
        rel = match.group(1).strip()
        content = match.group(2).strip()
        if rel.startswith(("wiki", "op")):
            content = content.replace(" ", "_")

        return f':{rel} "{content}"'

    delinearized = re.sub(r":([a-zA-Z0-9]+)\s+<lit>(.*?)</lit>", reverse_literal, delinearized)

    # Glue numbers back together, e.g. ':quant -547' -> ':quant -547'
    # but should not trigger for literal values, like ':value "34 61 09 91 12 135"'
    delinearized = re.sub(r"(?<![\"\d])(\s+-?\d*\.?\d*) (\d+)", r"\1\2", delinearized)

    delinearized = delinearized.replace("<rel>", "(")
    delinearized = delinearized.replace("</rel>", ")")

    return delinearized


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
