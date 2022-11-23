# fmt: off
# Tokens to add to the tokenizer
TOKENS_TO_ADD = (
    # ARGs
    ":ARG0", ":ARG1", ":ARG2", ":ARG3", ":ARG4", ":ARG5", ":ARG6", ":ARG7", ":ARG8", ":ARG9",
    # Other roles
    ":subset", ":accompanier", ":age", ":beneficiary", ":concession", ":condition", ":consist-of", ":degree",
    ":destination", ":direction", ":domain", ":duration", ":example", ":extent", ":frequency", ":instrument", ":li",
    ":location", ":manner", ":medium", ":mod", ":mode", ":name", ":ord", ":part", ":path", ":polarity", ":polite",
    ":poss", ":purpose", ":quant", ":range", ":scale", ":source", ":subevent", ":time", ":topic", ":unit", ":value",
    ":wiki", ":calendar", ":century", ":day", ":dayperiod", ":decade", ":era", ":month", ":quarter", ":season",
    ":timezone", ":weekday", ":year", ":year2",
    # OPs 
    ":op1", ":op2", ":op3", ":op4", ":op5", ":op6", ":op7", ":op8", ":op9",
    # Sentence IDs
    ":snt1", ":snt2", ":snt3", ":snt4", ":snt5", ":snt6", ":snt7", ":snt8", ":snt9",
    # Special tokens and frames
    "-91", "-quantity", "-entity", "~~of", ":prep-", ":conj-", "amr-unknown", "amr-choice", "multi-sentence", ":negation",
    # Word senses
    ":sense00", ":sense01", ":sense02", ":sense03", ":sense04", ":sense05", ":sense06", ":sense07", ":sense08", ":sense09",
    ":sense1", ":sense2", ":sense3", ":sense4", ":sense5", ":sense6", ":sense7", ":sense8", ":sense9",
    # References to track coreference
    ":ref1", ":ref2", ":ref3", ":ref4", ":ref5", ":ref6", ":ref7", ":ref8", ":ref9",
    # To indicate the end of a branch
    ":endrel",
    # Special language token for AMR
    "amr_XX"
)

# Prefixes to roles, used in delinearization
# Do not include :ref, :endrel, :sense
ROLE_PREFIXES = (
    ":ARG",
    ":accompanier",
    ":age",
    ":beneficiary",
    ":calendar",
    ":century",
    ":concession",
    ":condition",
    ":conj-",
    ":consist-of",
    ":day",
    ":dayperiod",
    ":decade",
    ":degree",
    ":destination",
    ":direction",
    ":domain",
    ":duration",
    ":era",
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
    ":month",
    ":name",
    ":op",
    ":ord",
    ":part",
    ":path",
    ":polarity",
    ":polite",
    ":poss",
    ":prep-",
    ":purpose",
    ":quant",
    ":quarter",
    ":range",
    ":scale",
    ":season",
    ":snt",
    ":source",
    ":subevent",
    ":subset",
    ":time",
    ":timezone",
    ":topic",
    ":unit",
    ":value",
    ":weekday",
    ":wiki",
    ":year",
    ":year2",
)
# fmt: on
