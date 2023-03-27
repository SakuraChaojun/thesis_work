import tqdm  # 进度更新包
import torch
import logging
import os
from sklearn import metrics

logger = logging.getLogger('main.eval')
input_parameters = 7 # 参数修改点


def __load_model__(ckpt):
    '''
    ckpt: Path of the checkpoint
    return: Checkpoint dict
    '''
    if os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt)
        print("Successfully loaded checkpoint '%s'" % ckpt)
        return checkpoint
    else:
        raise Exception("No checkpoint found at '%s'" % ckpt)


def train_epoch(model, trainLoader, optimizer, device):
    model.to(device)
    for batch in tqdm.tqdm(trainLoader, desc='Training:    ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, input_parameters, 2)
        # 这里的torch.chunk（）的作用是把一个tensor均匀分割成若干个小tensor 源码定义:torch.chunk(intput,chunks,dim=0)
        optimizer.zero_grad()  # 模型训练断点
        loss, prediction, ground_truth = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2].squeeze(2),
                                               datas[3].squeeze(2),
                                               datas[4].squeeze(2), datas[5].squeeze(2), datas[6].squeeze(2))
        loss.backward()
        optimizer.step()
    return model, optimizer


def test_epoch(model, testLoader, device, ckpt=None):
    model.to(device)
    if ckpt is not None:
        checkpoint = __load_model__(ckpt)
        model.load_state_dict(checkpoint['state_dict'])
    ground_truth = torch.tensor([], device=device)
    prediction = torch.tensor([], device=device)
    for batch in tqdm.tqdm(testLoader, desc='Testing:     ', mininterval=2):
        batch = batch.to(device)
        datas = torch.chunk(batch, input_parameters, 2)
        loss, p, label = model(datas[0].squeeze(2), datas[1].squeeze(2), datas[2].squeeze(2), datas[3].squeeze(2),
                               datas[4].squeeze(2), datas[5].squeeze(2), datas[6].squeeze(2))
        prediction = torch.cat([prediction, p])
        ground_truth = torch.cat([ground_truth, label])
    acc = metrics.accuracy_score(torch.round(ground_truth).detach().cpu().numpy(),
                                 torch.round(prediction).detach().cpu().numpy())
    auc = metrics.roc_auc_score(ground_truth.detach().cpu().numpy(), prediction.detach().cpu().numpy())
    logger.info('auc: ' + str(auc) + ' acc: ' + str(acc))
    print('auc: ' + str(auc) + ' acc: ' + str(acc))
    return auc
