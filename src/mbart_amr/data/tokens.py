# fmt: off
import re


AMR_LANG_CODE = "amr_XX"
STARTREL = ":startrel"
ENDREL = ":endrel"
STARTLIT = ":startlit"
ENDLIT = ":endlit"

# -00 is a valid sense!
SENSES = (":sense0", ":sense1", ":sense2", ":sense3", ":sense4", ":sense5", ":sense6", ":sense7", ":sense8", ":sense9")
# ORDER IS IMPORTANT, :ref1 is the only special token that can occur at start of sequence so is used in constraints
REFS = (":ref1", ":ref2", ":ref3", ":ref4", ":ref5", ":ref6", ":ref7", ":ref8", ":ref9")
ARGS = (":ARG0", ":ARG1", ":ARG2", ":ARG3", ":ARG4", ":ARG5", ":ARG6", ":ARG7", ":ARG8", ":ARG9")
OPS = (":op1", ":op2", ":op3", ":op4", ":op5", ":op6", ":op7", ":op8", ":op9")
SENTS = (":snt1", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":snt9")

OTHER_ROLES = (
    ":subset", ":accompanier", ":age", ":beneficiary", ":concession", ":condition", ":consist-of", ":degree",
    ":destination", ":direction", ":domain", ":duration", ":example", ":extent", ":frequency", ":instrument", ":li",
    ":location", ":manner", ":medium", ":mod", ":mode", ":name", ":ord", ":part", ":path", ":polarity", ":polite",
    ":poss", ":purpose", ":quant", ":range", ":scale", ":source", ":subevent", ":time", ":topic", ":unit", ":value",
    ":wiki", ":calendar", ":century", ":day", ":dayperiod", ":decade", ":era", ":month", ":quarter", ":season",
    ":timezone", ":weekday", ":year", ":year2", ":conj-as-if"
)

PREP_PREFIX = ":prep-"
MULTI_SENTENCE = "multi-sentence"
UNKOWN = "amr-unknown"
CHOICE = "amr-choice"
NEGATION = ":negation"

FRAME_91_ID = "-91"
OF_SUFFIX = "~~of"
SPECIAL_SUFFIXES = ("-quantity", "-entity")


TOKENS_TO_ADD = ((AMR_LANG_CODE, STARTREL, ENDREL, STARTLIT, ENDLIT) + SENSES + REFS + ARGS + OPS + SENTS + OTHER_ROLES
                 + (PREP_PREFIX, FRAME_91_ID, OF_SUFFIX, MULTI_SENTENCE, UNKOWN, CHOICE, NEGATION) + SPECIAL_SUFFIXES)

# NUMBERED PREFIXES: special tokens that are also valid if they have another number after them
# Only for SENSE and ARG these are different than their full list: we only want those ending in 1...9
# SO THESE DO NOT INCLUDE :sense0 NOR :ARG0!!!
SENSE_NUM_PREFIXES = tuple(sense for sense in SENSES if "0" not in sense)  # :sense1 ... :sense9
REF_NUM_PREFIXES = REFS
ARG_NUM_PREFIXES = tuple(arg for arg in ARGS if "0" not in arg)  # :ARG1 ... :ARG9
OP_NUM_PREFIXES = OPS
SENT_NUM_PREFIXES = SENTS


def _make_prefixes():
    """Make a tuple of numberless prefixes (e.g. :ARG but not :ARG1) to be used
    in the delinearization process Using regex/len checking to be as robust as possible
    so that the vocabulary can be changed if wanted.

    Included: args, ops, sents, "other_roles", and special_prefixes
    """
    no_number_prefixes = []
    for token_type in (ARGS, OPS, SENTS):
        unique = tuple(set(re.sub(r"\d+", "", token) for token in token_type))

        if len(unique) > 1:
            raise ValueError("Expected unique numberless roles, instead got", unique)
        no_number_prefixes.append(unique[0])

    no_number_prefixes = OTHER_ROLES + tuple(no_number_prefixes) + (PREP_PREFIX, )
    return no_number_prefixes


# Prefixes to roles, used in delinearization
# Do not include :ref, :startrel, :endrel, :startlit, :endlit, :sense
ROLE_NONUM_PREFIXES = _make_prefixes()

OF_ABLES = ARGS + OPS + OTHER_ROLES

# fmt: on
