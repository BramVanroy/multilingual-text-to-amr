OTHER_ROLES = (
    ":subset", ":accompanier", ":age", ":beneficiary", ":concession", ":condition", ":consist-of", ":degree",
    ":destination", ":direction", ":domain", ":duration", ":example", ":extent", ":frequency", ":instrument", ":li",
    ":location", ":manner", ":medium", ":mod", ":mode", ":name", ":ord", ":part", ":path", ":polarity", ":polite",
    ":poss", ":purpose", ":quant", ":range", ":scale", ":source", ":subevent", ":time", ":topic", ":unit", ":value",
    ":wiki", ":calendar", ":century", ":day", ":dayperiod", ":decade", ":era", ":month", ":quarter", ":season",
    ":timezone", ":weekday", ":year", ":year2", ":conj-as-if", ":role", ":superset", ":meaning", ":cost", ":cause",
    ":employed-by", ":relation"
)


NUMABLE_PREFIXES = (":ARG", ":snt", ":ref", ":op")
ROLES = OTHER_ROLES + NUMABLE_PREFIXES

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

TOKENS_TO_ADD = (OTHER_ROLES +
                 NUMABLE_PREFIXES +
                 SUFFIXES +
                 (
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
                     OF_SUFFIX
                 )
                 )