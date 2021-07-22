# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import Levenshtein


class RecMetric(object):
    def __init__(self, main_indicator='acc', **kwargs):
        self.main_indicator = main_indicator
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0.0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            norm_edit_dis += Levenshtein.distance(pred, target) / max(
                len(pred), len(target), 1)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / all_num
        }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / self.all_num
        norm_edit_dis = 1 - self.norm_edit_dis / self.all_num
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0

class RecFullMetric(object):
    def __init__(self, main_indicator='acc_luc_full', long_word_min_len=10, **kwargs):
        self.main_indicator = main_indicator
        self.long_word_min_len = long_word_min_len
        self.cases = ['luc_full', 'luc_long', 'mc_full', 'mc_long']
        self.reset()

    def __call__(self, pred_label, *args, **kwargs):
        preds, labels = pred_label

        correct_num = {case: 0 for case in self.cases}
        edit_dis = {case: 0.0 for case in self.cases}
        norm_edit_dis = {case: 0.0 for case in self.cases}
        long_num = 0
        all_num = 0
        long_char = 0
        all_char = 0

        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            pred_target = {'luc': (pred, target), 'mc': (pred.lower(), target.lower())}

            is_long = (len(target) >= self.long_word_min_len)

            for case, (pred, target) in pred_target.items():
                cur_edit_dis = Levenshtein.distance(pred, target)
                is_correct = int(pred == target)

                correct_num[case + '_full'] += is_correct
                edit_dis[case + '_full'] += cur_edit_dis
                norm_edit_dis[case + '_full'] += cur_edit_dis / len(target)

                if is_long:
                    correct_num[case + '_long'] += is_correct
                    edit_dis[case + '_long'] += cur_edit_dis
                    norm_edit_dis[case + '_long'] += cur_edit_dis / len(target)

            all_char += len(target)
            all_num += 1

            if is_long:
                long_char += len(target)
                long_num += 1

        for case in self.cases:
            self.correct_num[case] += correct_num[case]
            self.norm_edit_dis[case] += norm_edit_dis[case]
            self.edit_dis[case] += edit_dis[case]

        self.all_num += all_num
        self.long_num += long_num
        self.all_char += all_char
        self.long_char += long_char

        acc = {'acc_' + case: 1.0 * correct_num / max(all_num if (case[-4:] == 'full') else long_num, 1)
                for case, correct_num in correct_num.items()}
        norm_edit_dis = {'norm_edit_dis_' + case: 1 - dis / max(all_num if (case[-4:] == 'full') else long_num, 1)
                for case, dis in norm_edit_dis.items()}
        edit_dis = {'edit_dis_' + case: 1 - dis / max(all_char if (case[-4:] == 'full') else long_char, 1)
                for case, dis in edit_dis.items()}
        return dict(**acc, **norm_edit_dis, **edit_dis)

    def get_metric(self):
        acc = {'acc_' + case: 1.0 * correct_num / max(self.all_num if (case[-4:] == 'full') else self.long_num, 1)
                for case, correct_num in self.correct_num.items()}
        norm_edit_dis = {'norm_edit_dis_' + case: 1 - dis / max(self.all_num if (case[-4:] == 'full') else self.long_num, 1)
                for case, dis in self.norm_edit_dis.items()}
        edit_dis = {'edit_dis_' + case: 1 - dis / max(self.all_char if (case[-4:] == 'full') else self.long_char, 1)
                for case, dis in self.edit_dis.items()}
        self.reset()
        return dict(**acc, **norm_edit_dis, **edit_dis)

    def reset(self):
        self.correct_num = {case: 0 for case in self.cases}
        self.edit_dis = {case: 0.0 for case in self.cases}
        self.norm_edit_dis = {case: 0.0 for case in self.cases}
        self.long_num = 0
        self.all_num = 0
        self.long_char = 0
        self.all_char = 0

