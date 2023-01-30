"""
Also read AMR guidelines
Parse corpus as linearized and check what's possible and what's not
---

prev_token = tokens[idx-1]
num_startrels = prev_tokens.count...
num_endrels = prev_tokens.count...
num_quotes = prev_tokens.count...

1. Only be able to generate a new endrel if less endrels than startrels
2. Sense-ids only possible after tokens that are not special :ARG tokens
3. Opened `"` quotes need to be closed before a relation is started
4. When REF?
5. Something about :wiki? (always followed by `"` or "-")
6. Sometimes :ARG can be followed by "-"
7. Check whether/which relations are always followed by a non-special token


...
# If the current length is already close to max length
# and num_startrels != num_endrels, make sure that we have the room to generate
# Same with open double quotes

"""