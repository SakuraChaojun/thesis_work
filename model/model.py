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

        self.read_embed_linear = nn.Linear(self.memory_value_state_dim + self.memory_key_state_dim, self.final_fc_dim,
                                           bias=True)

        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)

        self.difficulty_linear = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)
        self.difficulty_linear_final = nn.Linear(self.q_embed_dim, self.q_embed_dim, bias=True)

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

        self.time_embed = nn.Embedding(9902, self.q_embed_dim, padding_idx=0)  # 102 for 2017 datasets
        self.attempt_embed = nn.Embedding(4000, self.q_embed_dim, padding_idx=0)
        # hint, hint_total,
        self.hint_embed = nn.Embedding(2, self.q_embed_dim, padding_idx=0)
        self.hint_total_embed = nn.Embedding(60, self.q_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        # begin hint and attempt weight value
        nn.init.kaiming_normal_(self.difficulty_linear.weight)
        nn.init.kaiming_normal_(self.difficulty_linear_final.weight)

        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)
        nn.init.constant_(self.difficulty_linear.bias, 0)
        nn.init.constant_(self.difficulty_linear_final.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

        nn.init.kaiming_normal_(self.time_embed.weight)
        nn.init.kaiming_normal_(self.attempt_embed.weight)
        nn.init.kaiming_normal_(self.hint_embed.weight)
        nn.init.kaiming_normal_(self.hint_total_embed.weight)

    # new function begin here

    def forward(self, q_data, qa_data, target, time_data, attempt_data, hint_data, hintTotal_data):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)

        # begin timedata
        # time_layer = nn.Tanh()
        # time_embed_data_layer = time_layer(self.time_embed(time_data))
        # end timedata

        # begin difficulty linear
        time_embed_data = self.time_embed(time_data)
        attempt_embed_data = self.attempt_embed(attempt_data)
        hint_action_embed_data = self.hint_total_embed(hint_data)
        hint_total_embed_data = self.hint_total_embed(hintTotal_data)

        time_init_weight = self.difficulty_linear(time_embed_data + q_embed_data)
        time_weight_update = torch.tanh(time_init_weight)
        time_second_weight = self.difficulty_linear_final(time_weight_update)
        time_final_weight = torch.sigmoid(time_second_weight)

        difficulty = self.difficulty_linear(
            hint_total_embed_data + hint_action_embed_data + attempt_embed_data + q_embed_data)
        difficulty_weight = torch.tanh(difficulty)
        difficulty_weight = self.difficulty_linear_final(difficulty_weight)
        difficulty_final = torch.sigmoid(difficulty_weight)

        q_embed_data = self.difficulty_linear(q_embed_data  + (
                difficulty_final * (hint_action_embed_data + hint_total_embed_data + attempt_embed_data)))


        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat([self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)  # 注意力机制改进 ？
            correlation_weight = self.mem.attention(q)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat([value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        input_embed_content = torch.cat([input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1) + + (time_final_weight * time_embed_data)
        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(predict_input.view(batch_size * seqlen, -1)))

        pred = self.predict_linear(read_content_embed)
        target_1d = target.view(-1, 1)  # [batch_size * seq_len, 1]
        mask = target_1d.ge(1)  # [batch_size * seq_len, 1]
        pred_1d = pred.view(-1, 1)  # [batch_size * seq_len, 1]

        filtered_pred = torch.masked_select(pred_1d, mask)
        filtered_target = torch.masked_select(target_1d, mask) - 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target.float())

        return loss, torch.sigmoid(filtered_pred), filtered_target.float()
