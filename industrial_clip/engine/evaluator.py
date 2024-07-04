import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score

from dassl.evaluation.build import EVALUATOR_REGISTRY
from dassl.evaluation.evaluator import EvaluatorBase


@EVALUATOR_REGISTRY.register()
class ExtendedClassification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)

        self.cfg = cfg
        self.generate_top_k = 5
        self._lab2cname = lab2cname
        
        self.reset()

    def reset(self):
        self._correct = [0 for _ in range(self.generate_top_k)]
        self._total = [0 for _ in range(self.generate_top_k)]

        # only for the topk=1
        self._y_true = []
        self._y_pred = []
        self._y_logits = []

        if self.cfg.TEST.PER_CLASS_RESULT or self.cfg.VALIDATION.PER_CLASS_RESULT:
            assert self._lab2cname is not None
            self._per_class_res = defaultdict(list)
        else:
            self._per_class_res = None

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]

        if isinstance(gt, (tuple)):
            label = gt[1]
            gt = gt[1][gt[0]]
        else:
            label = gt

        top = [[] for _ in range(self.generate_top_k)]
        for i in range(self.generate_top_k):
            # if we have less than i+1 classes, we repeat the last possible guess
            if mo.shape[1] < self.generate_top_k:
                top[i] = torch.topk(mo, min(i + 1, mo.shape[1]))[1]
                last_column = top[i][:,-1:]
                repeated_last_column = last_column.repeat_interleave((i + 1) - top[i].shape[1], dim=1)
                top[i] = torch.cat((top[i], repeated_last_column), dim=1)
            # else we take the top i+1 classes
            else:
                top[i] = torch.topk(mo, i + 1)[1]

        for i in range(self.generate_top_k):
            pred = label[top[i]]
            gtk = gt.unsqueeze(1).repeat(1, pred.shape[1])
            matches = pred.eq(gtk).float()
            correct = matches.sum(dim=1).bool().sum()

            self._correct[i] += int(correct)
            self._total[i] += gt.shape[0]

            # only for the topk=1
            if i == 0:
                self._y_true.extend(gt.data.cpu().numpy().tolist())
                self._y_pred.extend(pred.data.cpu().numpy().tolist())
                if self.cfg.TEST.REPORT_LOGITS:
                    self._y_logits.extend([mo.data.cpu().numpy()])

                if self._per_class_res is not None:
                    for i, l in enumerate(gt):
                        matches_i = int(matches[i].item())
                        self._per_class_res[l.item()].append(matches_i)


    def evaluate(self):

        results = OrderedDict()

        # only for the topk=1
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        results["topk_accuracy"] = [0 for _ in range(self.generate_top_k)]
        results["topk_total"] = [0 for _ in range(self.generate_top_k)]
        results["topk_correct"] = [0 for _ in range(self.generate_top_k)]
        for i in range(self.generate_top_k):
            acc = 100.0 * self._correct[i] / self._total[i]
            results["topk_accuracy"][i] = acc
            results["topk_total"][i] = self._total[i]
            results["topk_correct"][i] = self._correct[i]
            print(f"Top-{i + 1} accuracy: {acc:.1f}% ({self._correct[i]:,}/{self._total[i]:,})")

        # only for the topk=1
        results["macro_f1"] = macro_f1
        print(f"Macro F1: {macro_f1:.1f}%")

        # derive per-class result (only for the topk=1)
        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            results["perclass_accuracy"] = []
            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                if total == 0:
                    acc = 0
                else:
                    acc = 100.0 * correct / total
                accs.append(acc)
                results["perclass_accuracy"].append(
                    f"* acc: {acc:.1f}%\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"class: {label} ({classname})"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["mean_perclass_accuracy"] = mean_acc

        # pass results for cmat (only for the topk=1)
        results["y_true"] = self._y_true
        results["y_pred"] = [a[0] for a in self._y_pred]
        unique_classname_indices = list(dict.fromkeys(results["y_true"] + results["y_pred"]))
        results["classnames"] = [b for a, b in self._lab2cname.items() if a in unique_classname_indices]
        # rearrange y_true and y_pred to match the classnames
        index_mapping = {old_i: new_i for new_i, old_i in enumerate(unique_classname_indices)}
        results["y_true"] = [index_mapping[i] for i in results["y_true"]]
        results["y_pred"] = [index_mapping[i] for i in results["y_pred"]]

        if self.cfg.TEST.REPORT_LOGITS:
            results["y_logits"] = self._y_logits

        return results
