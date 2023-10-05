from multi_amr.data.postprocessing_graph import BACKOFF
from smatchpp import Smatchpp, util
import penman


class BackOffSmatchpp(Smatchpp):
    def process_corpus(self, amrs, amrs2):
        status = []
        match_dict = {}
        # Track the indices where we had to backoff
        back_offed_idxs = []
        for i, a in enumerate(amrs):
            try:
                match, tmpstatus, _ = self.process_pair(a, amrs2[i])
            except Exception as exc:
                back_offed_idxs.append(i)
                match, tmpstatus, _ = self.process_pair(a, penman.encode(BACKOFF))
            status.append(tmpstatus)
            util.append_dict(match_dict, match)
        return match_dict, status, back_offed_idxs

    def score_corpus(self, amrs, amrs2):
        match_dict, status, new_back_offed_idxs = self.process_corpus(amrs, amrs2)

        # pairwise statistic
        if self.printer.score_type is None:
            final_result = []
            for i in range(len(amrs)):
                match_dict_tmp = {k: [match_dict[k][i]] for k in match_dict.keys()}
                result = self.printer.get_final_result(match_dict_tmp)
                final_result.append(result)

        # aggregate statistic (micro, macro...)
        else:
            final_result = self.printer.get_final_result(match_dict)
        return final_result, status, new_back_offed_idxs
