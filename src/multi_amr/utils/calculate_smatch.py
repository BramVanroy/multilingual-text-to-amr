import logging
from typing import List

import smatch


def calculate_smatch(refs_penman: List[str], preds_penman: List[str]):
    total_match_num = total_test_num = total_gold_num = 0
    n_invalid = 0

    for sentid, (ref_penman, pred_penman) in enumerate(zip(refs_penman, preds_penman), 1):
        try:
            best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
                ref_penman, pred_penman, sent_num=sentid
            )
        except Exception:
            n_invalid += 1
            # At this point, any error is probably caused by the prediction
            continue

        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        smatch.match_triple_dict.clear()

    if n_invalid > 0:
        logging.warning(
            f"{n_invalid:,} ({n_invalid / len(preds_penman) * 100:.2f}%) prediction(s) were not valid AMR. "
            f" Smatch  scores only reflect the performance on valid AMR structures! Invalid structures have"
            f" been appended to invalid-amrs.txt in the output directory."
        )

    score = smatch.compute_f(total_match_num, total_test_num, total_gold_num)

    return {
        "smatch_precision": score[0],
        "smatch_recall": score[1],
        "smatch_fscore": score[2],
        "ratio_invalid_amrs": n_invalid / len(preds_penman) * 100,
    }
