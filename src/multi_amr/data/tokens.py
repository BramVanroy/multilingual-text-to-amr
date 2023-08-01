OTHER_ROLES = (
    ":subset",
    ":accompanier",
    ":age",
    ":beneficiary",
    ":concession",
    ":condition",
    ":consist-of",
    ":degree",
    ":destination",
    ":direction",
    ":domain",
    ":duration",
    ":example",
    ":extent",
    ":frequency",
    ":instrument",
    ":li",
    ":location",
    ":manner",
    ":medium",
    ":mod",
    ":mode",
    ":name",
    ":ord",
    ":part",
    ":path",
    ":polarity",
    ":polite",
    ":poss",
    ":purpose",
    ":quant",
    ":range",
    ":scale",
    ":source",
    ":subevent",
    ":time",
    ":topic",
    ":unit",
    ":value",
    ":wiki",
    ":calendar",
    ":century",
    ":day",
    ":dayperiod",
    ":decade",
    ":era",
    ":month",
    ":quarter",
    ":season",
    ":timezone",
    ":weekday",
    ":year",
    ":year2",
    ":conj-as-if",
    ":role",
    ":superset",
    ":meaning",
    ":cost",
    ":cause",
    ":employed-by",
    ":relation",
)


NUMABLE_PREFIXES = (":ARG", ":snt", ":ref", ":op")
STARTSWITH_ROLES = OTHER_ROLES + NUMABLE_PREFIXES

ARGS = (
    ":ARG0",
    ":ARG1",
    ":ARG2",
    ":ARG3",
    ":ARG4",
    ":ARG5",
    ":ARG6",
    ":ARG7",
    ":ARG8",
    ":ARG9",
)
SNTS = (
    ":snt1",
    ":snt2",
    ":snt3",
    ":snt4",
    ":snt5",
    ":snt6",
    ":snt7",
    ":snt8",
    ":snt9",
)
REFS = (
    ":ref1",
    ":ref2",
    ":ref3",
    ":ref4",
    ":ref5",
    ":ref6",
    ":ref7",
    ":ref8",
    ":ref9",
)
OPS = (
    ":op1",
    ":op2",
    ":op3",
    ":op4",
    ":op5",
    ":op6",
    ":op7",
    ":op8",
    ":op9",
)
SUFFIXES = ("-quantity", "-entity")

AMR_LANG_CODE = "AMR_lang"
STARTREL = ":startrel"
ENDREL = ":endrel"
STARTLIT = ":startlit"
ENDLIT = ":endlit"

PREP_PREFIX = ":prep-"
MULTI_SENTENCE = "multi-sentence"
UNKOWN = "amr-unknown"
CHOICE = "amr-choice"
NEGATION = ":negation"
OF_SUFFIX = ":of-rel"

TOKENS_TO_ADD = (
    OTHER_ROLES
    + ARGS
    + SNTS
    + REFS
    + OPS
    + SUFFIXES
    + (
        AMR_LANG_CODE,
        STARTREL,
        ENDREL,
        STARTLIT,
        ENDLIT,
        PREP_PREFIX,
        MULTI_SENTENCE,
        UNKOWN,
        CHOICE,
        NEGATION,
        OF_SUFFIX,
    )
)
