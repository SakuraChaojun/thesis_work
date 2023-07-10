import torch
from torch.utils import data
from .readdata_new import DataReader

# assist2015/assist2015_train.txt assist2015/assist2015_test.txt
# assist2017/assist2017_train.txt assist2017/assist2017_test.txt question:102
# assist2009/builder_train.csv assist2009/builder_test.csv    question_dim : 124

# pathway:../data/raw_data/assist_2017_test_time.csv
# pathway: ../dataloader/assist_2009_train_time.csv

def getDataLoader(batch_size, num_of_questions, max_step):
    handle = DataReader('../data/raw_data/assist_2017_train_time.csv',
                        '../data/raw_data/assist_2017_test_time.csv', max_step,
                        num_of_questions)

    train, vali = handle.getTrainData()

    dtrain = torch.tensor(train.astype(int).tolist(), dtype=torch.long) # 转换为 tensor
    dvali = torch.tensor(vali.astype(int).tolist(), dtype=torch.long)
    dtest = torch.tensor(handle.getTestData().astype(int).tolist(), dtype=torch.long)

    trainLoader = data.DataLoader(dtrain, batch_size=batch_size, shuffle=True)
    valiLoader = data.DataLoader(dvali, batch_size=batch_size, shuffle=True)
    testLoader = data.DataLoader(dtest, batch_size=batch_size, shuffle=False)

    return trainLoader, valiLoader, testLoader
