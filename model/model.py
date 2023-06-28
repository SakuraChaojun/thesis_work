import torch
import torch.nn as nn
from model.memory import DKVMN

class MODEL(nn.Module):

    def __init__(self, n_question, batch_size, q_embed_dim, qa_embed_dim, memory_size, final_fc_dim):
        super(MODEL, self).__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size  # memory size
        self.memory_key_state_dim = q_embed_dim
        self.memory_value_state_dim = qa_embed_dim
        self.final_fc_dim = final_fc_dim

        self.read_embed_linear = nn.Linear(
            self.memory_value_state_dim + self.memory_key_state_dim, self.final_fc_dim,
            bias=True)

        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.difficulty_linear = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)
        self.difficulty_linear_final = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)

        self.keyweight_linear = nn.Linear(200, 200, bias=True)
        self.valueweight_linear = nn.Linear(200, 200, bias=True)
        self.keyselfweight_linear = nn.Linear(200, 200, bias=True)
        # alt linear layer
        self.ability_linear = nn.Linear(200, 200, bias=True)

        self.dropout = nn.Dropout(0.5)

        # init the key memory
        self.init_memory_key = nn.Parameter(torch.randn(self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)

        # init the value memory
        self.init_memory_value = nn.Parameter(torch.randn(self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        self.q_embed = nn.Embedding(self.n_question + 1, self.q_embed_dim, padding_idx=0)  # A matrix
        self.qa_embed = nn.Embedding(2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)  # B matrix

        self.time_embed = nn.Embedding(4960, self.q_embed_dim, padding_idx=0)  # 102 for 2017 datasets
        self.attempt_embed = nn.Embedding(100, self.q_embed_dim, padding_idx=0)
        self.hint_embed = nn.Embedding(2, self.q_embed_dim, padding_idx=0)
        self.hint_total_embed = nn.Embedding(60, self.q_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.kaiming_normal_(self.difficulty_linear.weight)
        nn.init.kaiming_normal_(self.difficulty_linear_final.weight)

        nn.init.xavier_normal_(self.ability_linear.weight)
        nn.init.xavier_normal_(self.keyweight_linear.weight)
        nn.init.xavier_normal_(self.keyselfweight_linear.weight)
        nn.init.xavier_normal_(self.valueweight_linear.weight)

        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        nn.init.constant_(self.difficulty_linear.bias, 0)
        nn.init.constant_(self.difficulty_linear_final.bias, 0)
        nn.init.constant_(self.ability_linear.bias, 0)
        nn.init.constant_(self.valueweight_linear.bias, 0)
        nn.init.constant_(self.keyweight_linear.bias, 0)
        nn.init.constant_(self.keyselfweight_linear.bias,0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

        nn.init.kaiming_normal_(self.time_embed.weight)
        nn.init.kaiming_normal_(self.attempt_embed.weight)
        nn.init.kaiming_normal_(self.hint_embed.weight)
        nn.init.kaiming_normal_(self.hint_total_embed.weight)

    def forward(self, q_data, qa_data, target, time_data, attempt_data, hint_data, hintTotal_data):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]

        q_embed_data = self.q_embed(q_data)
        time_embed_data = self.time_embed(time_data)
        attempt_embed_data = self.attempt_embed(attempt_data)
        hint_action_embed_data = self.hint_total_embed(hint_data)
        hint_total_embed_data = self.hint_total_embed(hintTotal_data)

        origin_q_embed_data = q_embed_data

        # begin time linear
        time_init_weight = self.difficulty_linear(time_embed_data + q_embed_data)
        time_weight_update = torch.tanh(time_init_weight)
        time_second_weight = self.difficulty_linear_final(time_weight_update)
        time_final_weight = torch.sigmoid(time_second_weight)

        # begin difficulty layer
        difficulty = self.difficulty_linear(hint_total_embed_data + hint_action_embed_data + attempt_embed_data)
        difficulty_weight = torch.tanh(difficulty)
        difficulty_weight = self.difficulty_linear_final(difficulty_weight)
        difficulty_final = torch.sigmoid(difficulty_weight)

        q_embed_data = self.difficulty_linear(
            q_embed_data + (difficulty_final * (hint_total_embed_data + hint_action_embed_data + attempt_embed_data)))

        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)  # 题目在这里已经带上难度diff作为整体
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        slice_origin_q_embed_data = torch.chunk(origin_q_embed_data, seqlen, 1)

        value_read_content_l = []
        value_kt_read_content_l = []
        kt_self_read_content_l = []

        input_embed_l = []

        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            q_original = slice_origin_q_embed_data[i].squeeze(1)
            # question-concept E-C
            correlation_weight = self.mem.attention(q)
            # value_weight = self.mem.value_attention(q_original)
            value_weight = self.mem.value_attention(q)
            # correlation_weight = torch.cat((correlation_weight , value_weight),0)

            debug = []

            # Read Process
            read_content = self.mem.read(correlation_weight)
            read_content_value = self.mem.valueRead(value_weight)
            value_read_content_l.append(read_content)
            value_kt_read_content_l.append(read_content_value)

            input_embed_l.append(q)
            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        all_read_valueKT_content = torch.cat([value_kt_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        all_read_value_content = self.keyweight_linear(all_read_value_content)
        all_read_valueKT_content = self.valueweight_linear(all_read_valueKT_content)

        all_read_kt_self = self.mem.kt_self_read()
        all_read_kt_self = all_read_kt_self.unsqueeze(0)
        all_read_kt_self = all_read_kt_self.repeat(16,10,4)

        all_read_kt_self = self.keyselfweight_linear(all_read_kt_self)

        #output = self.ability_linear(all_read_kt_self)

        value_norm = nn.LayerNorm([200,200])
        all_read_valueKT_content = value_norm(all_read_valueKT_content)

        output_weight = torch.sigmoid(all_read_value_content+all_read_valueKT_content)

        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1) + (
                time_final_weight * time_embed_data)

        predict_input = torch.cat([output_weight, input_embed_content], -1)

        debug = []
        # predict_input = self.dropout(predict_input)
        # drop put
        # layer norm
        # linear tanh/relu/

        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

        pred = self.predict_linear(read_content_embed)  ## drop out
        # pred = self.dropout(pred)

        target_1d = target.view(-1, 1)  # [batch_size * seq_len, 1]
        mask = target_1d.ge(1)  # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask) - 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target.float())

        return loss, torch.sigmoid(filtered_pred), filtered_target.float()
