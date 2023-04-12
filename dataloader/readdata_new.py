import numpy as np
import itertools
from sklearn.model_selection import KFold


class DataReader():  # 初始化，使用了魔法函数，self为惯例第一个参数名称。https://www.runoob.com/python3/python3-class.html
    def __init__(self, train_path, test_path, maxstep, num_ques):
        self.train_path = train_path  # 读取path地址 train
        self.test_path = test_path  # 读取path 地址 test
        self.maxstep = maxstep  # 习题集最大长度 ， default =200
        self.num_ques = num_ques  # 回答问题的数目，assist2009_updated数据集里是110
        # 以上参数被dataloader 调用

    def getData(self, file_path):  # path : data location  路径：数据位置
        datas = []  # 初始化datas 为空 list
        with open(file_path, 'r') as file:  # open 函数读取文件 with as 上下文管理器 通过上下文管理器来自动控制
            for len, ques, ans, times, attempt, hint, hintTotal in itertools.zip_longest(
                    *[file] * 7):  # 迭代控制 longest返回最长的序列为准 # TODO times 修改点
                # print(len)
                len = int(len.strip().strip(','))  # 去掉length中的逗号，空格等信息
                ques = [int(q) for q in ques.strip().strip(',').split(',')]  # 去掉逗号空格信息后 按照逗号返回list
                ans = [int(a) for a in ans.strip().strip(',').split(',')]  # 去掉逗号空格信息后 按照逗号返回list
                times = [int(t) for t in times.strip().strip(',').split(',')]  # TODO times 修改点

                attempt = [int(e) for e in attempt.strip().strip(',').split(',')]
                hint = [int(h) for h in hint.strip().strip(',').split(',')]
                hintTotal = [int(o) for o in hintTotal.strip().strip(',').split(',')]

                slices = len // self.maxstep + (1 if len % self.maxstep > 0 else 0)  # 按照超参定义分块 如果还有余数则加一个块
                for i in range(slices):
                    data = np.zeros(shape=[self.maxstep, 7])  # 使用numpy存储数据 一块有维度 200 *3  # 0 ->question and answer(1->)
                    if len > 0:  # 1->question (1->)
                        if len >= self.maxstep:  # 如果len 大于超参则结束点为len          # 2->label (0->1, 1->2)
                            steps = self.maxstep
                        else:
                            steps = len
                        for j in range(steps):  # 最大步长循环
                            data[j][0] = ques[i * self.maxstep + j] + 1  # TODO 加1 知识点mapping从0开始
                            data[j][2] = ans[i * self.maxstep + j] + 1
                            data[j][3] = times[i * self.maxstep + j]  # TODO times 修改点
                            data[j][4] = attempt[i * self.maxstep + j]
                            data[j][5] = hint[i * self.maxstep + j]
                            data[j][6] = hintTotal[i * self.maxstep + j]
                            if ans[i * self.maxstep + j] == 1:
                                data[j][1] = ques[i * self.maxstep + j] + 1  # [j][1] 代表问题和答案镶嵌入
                            else:
                                data[j][1] = ques[i * self.maxstep + j] + self.num_ques + 1
                        len = len - self.maxstep  # 读数据检查点
                    datas.append(data.tolist())  # 为了打印出维度信息
            print('done: ' + str(np.array(datas).shape))

        return datas

    def getTrainData(self):
        print('loading train data...')
        kf = KFold(n_splits=5, shuffle=True, random_state=3)  # 5折交叉划分训练
        Data = np.array(self.getData(self.train_path))  # get data 方法
        for train_indexes, vali_indexes in kf.split(Data):  # 5折随机交叉
            valiData = Data[vali_indexes].tolist()  # 划分训练集
            trainData = Data[train_indexes].tolist()  # 划分验证集
        return np.array(trainData), np.array(valiData)

    def getTestData(self):
        print('loading test data...')
        testData = self.getData(self.test_path)
        return np.array(testData)
