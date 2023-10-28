import torch
from torch.nn.functional import cross_entropy
from torch.nn.modules.loss import _WeightedLoss

import numpy as np
import torch.nn as nn


EPSILON = 1e-32


class LogNLLLoss(_WeightedLoss):
    __constants__ = ['weight', 'reduction', 'ignore_index']

    def __init__(self, weight=None, size_average=None, reduce=None, reduction=None,
                 ignore_index=-100):
        super(LogNLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, y_input, y_target):
        # y_input = torch.log(y_input + EPSILON)
        return cross_entropy(y_input, y_target, weight=self.weight,
                             ignore_index=self.ignore_index)


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    dims = (0, *range(2, len(output.shape)))
    gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output*gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() + EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """

    epsilon = 1e-20
    n_classes = output.shape[1]

    output = torch.argmax(output, dim=1)
    true_positives = torch.tensor([((output == i) * (gt == i)).sum() for i in range(n_classes)]).float()
    selected = torch.tensor([(output == i).sum() for i in range(n_classes)]).float()
    relevant = torch.tensor([(gt == i).sum() for i in range(n_classes)]).float()

    precision = (true_positives + epsilon) / (selected + epsilon)
    recall = (true_positives + epsilon) / (relevant + epsilon)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (0, *range(2, len(output.shape)))

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError("The number of weights must match with the number of classes")
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return classwise_scores 

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)


class BinaryMetrics():
    """
    Compute common metrics for binary segmentation, including overlap metrics, distance metrics and MAE
    NOTE: batch size must be set to one for accurate measurement, batch size larger than one may cause errors!
    """
    def __init__(self, eps=1e-5, resolution=(1, 1), inf_result=np.nan):
        self.eps = eps
        self.resolution = resolution
        self.inf_result = inf_result

    def _check_inf(self, result):
        if result == np.inf:  # inf 无穷
            return self.inf_result
        else:
            return result

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)

        metric_dict = dict()
        metric_dict['pixel_acc'] = pixel_acc.item()
        metric_dict['dice'] = dice.item()
        metric_dict['precision'] = precision.item()
        metric_dict['recall'] = recall.item()
        metric_dict['specificity'] = specificity.item()

        return metric_dict

    def _calculate_distance_metrics(self, gt, pred):
        # shape: (N, C, H, W)
        gt_class = gt[0, ...].cpu().numpy().astype(np.int).astype(np.bool)  # (H, W)
        pred_class = pred[0, 0, ...].cpu().numpy().astype(np.int).astype(np.bool)  # (H, W)
        # surface_distance_dict = compute_surface_distances(gt_class, pred_class, self.resolution)
        # distances = surface_distance_dict['distances_pred_to_gt']
        # mean_surface_distance = self._check_inf(np.mean(distances))

        # compute Hausdorff distance (95 percentile)
        # hd95 = self._check_inf(compute_robust_hausdorff(surface_distance_dict, percent=95))

        metric_dict = dict()
        # metric_dict['mean_surface_distance'] = mean_surface_distance
        # metric_dict['hd95'] = hd95

        return metric_dict

    def _calculate_mae(self, gt, pred):
        # shape: (N, C, H, W)
        residual = torch.abs(gt.unsqueeze(1) - pred)
        mae = torch.mean(residual, dim=(2, 3)).squeeze().detach().cpu().numpy()

        metric_dict = dict()
        metric_dict['mae'] = mae
        return metric_dict

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        sigmoid_pred = nn.Sigmoid()(y_pred)
        class_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)

        assert class_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'

        overlap_metrics = self._calculate_overlap_metrics(y_true.to(y_pred.device, dtype=torch.float), class_pred)
        distance_metrics = self._calculate_distance_metrics(y_true, class_pred)
        mae = self._calculate_mae(y_true, sigmoid_pred)

        metrics = {**overlap_metrics, **distance_metrics, **mae}

        return metrics


class MetricMeter(object):
    """
    Metric记录器
    """
    def __init__(self, metrics):
        self.metrics = metrics
        self.initialization()

    def initialization(self):
        for metric in self.metrics:
            exec('self.' + metric + '=[]')

    def update(self, metric_dict):
        """
        将新的metric字典传入，更新记录器
        :param metric_dict: 指标字典
        :return: None
        """
        for metric_key, metric_value in metric_dict.items():
            try:
                exec('self.' + metric_key + '.append(metric_value)')
                # exec 执行储存在字符串或文件中的 Python 语句，相比于 eval，exec可以执行更复杂的 Python 代码
                # exec(object[, globals[, locals]])
                # object：必选参数，表示需要被指定的 Python 代码。它必须是字符串或 code 对象。
                # 如果 object 是一个字符串，该字符串会先被解析为一组 Python 语句，然后再执行（除非发生语法错误）。如果 object 是一个 code 对象，那么它只是被简单的执行。
                # globals：可选参数，表示全局命名空间（存放全局变量），如果被提供，则必须是一个字典对象。
                # locals：可选参数，表示当前局部命名空间（存放局部变量），如果被提供，可以是任何映射对象。
                # 如果该参数被忽略，那么它将会取与 globals 相同的值。
            except:
                continue

    def report(self, print_stats=True):
        """
        汇报目前记录的指标信息
        :param print_stats: 是否将指标信息打印在终端
        :return: report_str
        """
        report_str = ''
        for metric in self.metrics:
            metric_mean = np.nanmean(eval('self.' + metric), axis=0)  # 沿着指定的轴计算算数平均值，NAN忽略
            metric_std = np.nanstd(eval('self.' + metric), axis=0)
            if print_stats:
                stats = metric + ': {} ± {};'.format(np.around(metric_mean, decimals=4),  # 四舍五入到小数点后的位数
                                                     np.around(metric_std, decimals=4))
                print(stats, end=' ')
                report_str += stats
        return report_str





if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
