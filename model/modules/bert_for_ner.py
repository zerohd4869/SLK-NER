import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from model.modules.crf import CRF
from model.modules.linears import PoolerEndLogits, PoolerStartLogits
from model.losses.focal_loss import FocalLoss
from model.losses.label_smoothing import LabelSmoothingCrossEntropy
from transformers.modeling_bert import BertModel
from transformers.modeling_bert import BertPreTrainedModel

from model.modules.gaz_embed import Gaz_Embed
from model.functions.gaz_opt import get_batch_gaz
from model.functions.utils import reverse_padded_sequence


class BertSoftmaxForNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertSoftmaxForNer, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            else:
                loss_fct = CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


# Attention
# An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.  The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.
# We call our particular attention "Scaled Dot-Product Attention".   The input consists of queries and keys of dimension $d_k$, and values of dimension $d_v$.  We compute the dot products of the query with all keys, divide each by $\sqrt{d_k}$, and apply a softmax function to obtain the weights on the values.
def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    # query,key,value:torch.Size([30, 8, 10, 64])
    # decoder mask:torch.Size([30, 1, 9, 9])
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)  # torch.Size([30, 8, 64, 10])
    # torch.Size([30, 8, 10, 10])
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    # decoder scores:torch.Size([30, 8, 9, 9]),        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 64=512//8
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value:torch.Size([30, 10, 512])
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
            nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([30, 10, 512])
        return ret


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, label2id, data, device="cuda"):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(512 * 2 + 50 * 1, len(label2id))
        self.crf = CRF(tagset_size=len(label2id), tag_dictionary=label2id, device=device, is_bert=True)
        self.W = []
        self.rnn = []
        self.k_hop = 1
        for i in range(self.k_hop):
            self.W.append(nn.Linear(512 * 2, 512 * 2))
            self.rnn.append(nn.GRU(config.hidden_size if i == 0 else 512 * 2, 512, num_layers=1, bidirectional=True, batch_first=True).cuda())

        self.W = nn.ModuleList(self.W)
        self.rnn = nn.ModuleList(self.rnn)
        self.init_weights()
        self.label2id = label2id
        self.id2label = {a: b for b, a in label2id.items()}
        self.pooling = nn.Linear(1024, 50)
        self.gaz_embed_all = Gaz_Embed(data, 0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, input_lens=None, mode=None, gaz=None, reverse_gaz=None,
                char_seq_lengths=None):
        # (batch,256,768), (batch,768) <-- (batch,256) * 3
        # sequence_output, pooled_output, (hidden_states), (attentions)
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        # (batch, 256, 768)
        sequence_output = outputs[0]  # all tokens of last hidden state

        # get batch gaz ids
        batch_gaz_ids, batch_gaz_length, batch_gaz_mask = get_batch_gaz(reverse_gaz, input_ids.size(0), input_ids.size(1), input_lens)

        reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask = get_batch_gaz(gaz, input_ids.size(0), input_ids.size(1), input_lens)
        lengths = list(map(int, char_seq_lengths))
        reverse_batch_gaz_ids = reverse_padded_sequence(reverse_batch_gaz_ids, lengths)
        reverse_batch_gaz_length = reverse_padded_sequence(reverse_batch_gaz_length, lengths)
        reverse_batch_gaz_mask = reverse_padded_sequence(reverse_batch_gaz_mask, lengths)

        # gaz embedding (batch,256,3~4,50)
        gaz_embs_1st = self.gaz_embed_all((batch_gaz_ids, batch_gaz_length, batch_gaz_mask))  # .unsqueeze(-2)

        reverse_gaz_embs_1st = self.gaz_embed_all((reverse_batch_gaz_ids, reverse_batch_gaz_length, reverse_batch_gaz_mask))  # .unsqueeze(-2)
        zero_embs = torch.zeros([reverse_gaz_embs_1st.size(0), 2, reverse_gaz_embs_1st.size(2), reverse_gaz_embs_1st.size(3)]).cuda()
        reverse_gaz_embs_1st = torch.cat([zero_embs, reverse_gaz_embs_1st[:, :-2]], dim=1)

        gaz_embs = torch.cat([gaz_embs_1st, reverse_gaz_embs_1st], dim=-2)  # 2'

        for w, rnn in zip(self.W, self.rnn):
            try:
                rnn.flatten_parameters()
            except:
                pass
            # (batch,256,512*2), (2,batch,512)
            sequence_output, hidden = rnn(sequence_output)

        hidden0 = hidden.permute(1, 0, 2).reshape(input_ids.size(0), 1, 1, -1).repeat(1, input_ids.size(1), gaz_embs.size(2), 1).contiguous()
        # (batch, 256, 3-4, 50)
        hidden0 = self.pooling(hidden0)
        weight0 = torch.sum(torch.mul(hidden0, gaz_embs), -1)
        weight0 = nn.functional.softmax(weight0, -1)
        gaz_embs_q = torch.mul(gaz_embs, weight0.unsqueeze(-1))
        # (batch, 256, 50)
        gaz_embs_q = torch.sum(gaz_embs_q, 2)
        sequence_output = torch.cat([sequence_output, gaz_embs_q], dim=-1)
        # (batch,256,1024)
        sequence_output = self.dropout(sequence_output)
        # (batch,256,35)
        logits = self.classifier(sequence_output)
        outputs = (logits,)
        if labels is not None:
            loss = self.crf.calculate_loss(logits, tag_list=labels, lengths=input_lens)
            outputs = (loss,) + outputs
        return outputs  # (loss), scores


class BertSpanForNer(BertPreTrainedModel):
    def __init__(self, config, ):
        super(BertSpanForNer, self).__init__(config)
        self.soft_label = config.soft_label
        self.num_labels = config.num_labels
        self.loss_type = config.loss_type
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.start_fc = PoolerStartLogits(config.hidden_size, self.num_labels)
        if self.soft_label:
            self.end_fc = PoolerEndLogits(config.hidden_size + self.num_labels, self.num_labels)
        else:
            self.end_fc = PoolerEndLogits(config.hidden_size + 1, self.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        outputs = self.bert(input_ids, token_type_ids, attention_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        start_logits = self.start_fc(sequence_output)
        if start_positions is not None and self.training:
            if self.soft_label:
                batch_size = input_ids.size(0)
                seq_len = input_ids.size(1)
                label_logits = torch.FloatTensor(batch_size, seq_len, self.num_labels)
                label_logits.zero_()
                label_logits = label_logits.to(input_ids.device)
                label_logits.scatter_(2, start_positions.unsqueeze(2), 1)
            else:
                label_logits = start_positions.unsqueeze(2).float()
        else:
            label_logits = F.softmax(start_logits, -1)
            if not self.soft_label:
                label_logits = torch.argmax(label_logits, -1).unsqueeze(2).float()
        end_logits = self.end_fc(sequence_output, label_logits)
        outputs = (start_logits, end_logits,) + outputs[2:]

        if start_positions is not None and end_positions is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce']
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy()
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss()
            else:
                loss_fct = CrossEntropyLoss()
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            active_loss = attention_mask.view(-1) == 1
            active_start_logits = start_logits[active_loss]
            active_end_logits = end_logits[active_loss]

            active_start_labels = start_positions.view(-1)[active_loss]
            active_end_labels = end_positions.view(-1)[active_loss]

            start_loss = loss_fct(active_start_logits, active_start_labels)
            end_loss = loss_fct(active_end_logits, active_end_labels)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs
        return outputs
