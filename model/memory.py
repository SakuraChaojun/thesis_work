import torch
from torch import nn

class DKVMNHeadGroup(nn.Module):  # 用于读写数据的读写头
    def __init__(self, memory_size, memory_state_dim, is_write):  # 定义记忆矩阵 20*50  受到超参调控 N
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size  # 记忆矩阵的规模 N,20
        self.memory_state_dim = memory_state_dim  # memory state dim : D_V or D_K 200,50
        self.is_write = is_write
        '''
                       Key matrix or Value matrix K矩阵或者V矩阵
                       Key matrix is used for calculating correlation weight(attention weight) K阵计算相关权重
        '''
        if self.is_write:  # 初始化定义key矩阵不能写入
            self.erase = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(self.memory_state_dim, self.memory_state_dim, bias=True)

            nn.init.kaiming_normal_(self.erase.weight)  # 初始化权重
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    def addressing(self, control_input, memory):
        # control_input: embedded memory state dim (d_K)
        # key_matrix memory size * memory state dim (d_k)
        # correlation_weight: w(i) = k * key matrix (i)
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim) 16 * 20
            memory:                 Shape (memory_size, memory_state_dim) 20 * 20
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        # embedding_result : [batch size, memory size], each row contains each concept correlation weight for 1 question
        similarity_score = torch.matmul(control_input, torch.t(memory))  # 计算相似性地方可以改,行列转置。然后相乘
        correlation_weight = torch.nn.functional.softmax(similarity_score, dim=1)  # Shape: (batch_size, memory_size) softmax 方法可以改进
        # Shape: (batch_size, memory_size) 归一化到0到1之间
        return correlation_weight

    # 计算读取内容 read content： 相关权重和答题情况的乘积
    # Getting read content 读过程，参数为32*20*200的value矩阵和32*20的w注意力权重阵
    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        debug = []
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)  # + time
        return read_content

    def valueread(self, memory, control_input=None, valueread_weight=None):
        if valueread_weight is None:
            valueread_weight = self.value_attention(control_input = control_input, memory = memory)
        valueread_weight = valueread_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        value_c = torch.mul(valueread_weight, memory)
        valueread_content = value_c.view(-1, self.memory_size, self.memory_state_dim)
        valueread_content = torch.sum(valueread_content, dim=1)
        debug = []
        return valueread_content

    def write(self, control_input, memory, write_weight):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory

class DKVMN(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(control_input=control_input, memory=self.memory_key)
        return correlation_weight

    def value_attention(self, control_input):
        value_memory = self.memory_value
        unbind_memory = torch.chunk(value_memory, 16, dim=0)
        unbind_question = torch.chunk(control_input, 16, dim=0)
        question_knowledge_corr = []
        for i in range(len(unbind_question)):
            single_valueMemory = torch.t(unbind_memory[i].squeeze(0))
            single_question = unbind_question[i].repeat(1, 4)
            value_similarityScore = torch.matmul(single_question, single_valueMemory)
            value_correlation_weight = torch.nn.functional.softmax(value_similarityScore, dim=1)
            question_knowledge_corr.append(value_correlation_weight)
        init_tensor = question_knowledge_corr[0]
        for j in range(1, len(question_knowledge_corr)):
            init_tensor = torch.cat([init_tensor, question_knowledge_corr[j]], dim=0)
        debug = []
        return init_tensor

    def read(self, read_weight):
        read_content = self.value_head.read(memory=self.memory_value, read_weight=read_weight)
        return read_content

    def valueRead(self,valueread_weight):
        valueread_content = self.value_head.valueread(memory=self.memory_value, valueread_weight= valueread_weight)
        return valueread_content

    def write(self, write_weight, control_input):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)
        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value
