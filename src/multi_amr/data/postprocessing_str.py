import re


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


def postprocess_str_after_linearization(linearized: str) -> str:
    linearized = linearized.replace(":polarity -", ":negation")
    linearized = re.sub(r"-of\b", " </of>", linearized)

    # Re-implementation of SPRING, which also replaces "_" with " "
    # https://github.com/SapienzaNLP/spring/blob/39079940d028ba0dde4c1af60432be49f67d76f8/spring_amr/tokenization_bart.py#L143-L144
    def replace_literal(match):
        content = match.group(1).replace("_", " ")
        return f"<lit> {content} </lit>"

    linearized = re.sub(r'"(.*?)"', replace_literal, linearized)
    linearized = linearized.replace("(", " <rel>")
    linearized = linearized.replace(")", " </rel>")

    return linearized


def postprocess_str_after_delinearization(delinearized: str) -> str:
    delinearized = delinearized.replace(":negation", ":polarity -")
    delinearized = delinearized.replace("</of>", "-of")

    def reverse_literal(match):
        content = match.group(1).strip().replace(" ", "_")
        return f'"{content}"'

    delinearized = re.sub(r"<lit>(.*?)</lit>", reverse_literal, delinearized)
    delinearized = delinearized.replace("<rel>", "(")
    delinearized = delinearized.replace("</rel>", ")")

    return delinearized
