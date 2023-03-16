import torch
import shutil

def topacc(output, target, topk=(1,), predict = False):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        if predict:
            return res, pred[:1].reshape(-1).cpu().numpy()
        else:
            return res

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def bceacc(output, target):
    with torch.no_grad():
        assert output.size() == target.size()
        batch_size = target.size(0)
        res = []
        correct = (output > 0.5).eq(target.bool()).float().sum(0, keepdim = True)
        res.append(correct.mul_(100.0 / batch_size))

        return res
