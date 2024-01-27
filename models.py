import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from data import Vocab
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout=0.3, activation=True):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.LeakyReLU(negative_slope=0.1) if activation else nn.Identity()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class LabelBiaffine(nn.Module):
    def __init__(self, left_features, right_features, out_features, bias=True):
        """
        Args:
            left_features: size of left input
            right_features: size of right input
            out_features: size of output
            bias: If set to False, the layer will not learn an additive bias.
                Default: True
        """
        super(LabelBiaffine, self).__init__()
        self.left_features = left_features
        self.right_features = right_features
        self.out_features = out_features

        self.U = Parameter(torch.Tensor(self.out_features, self.left_features, self.right_features))
        self.weight_left = Parameter(torch.Tensor(self.out_features, self.left_features))
        self.weight_right = Parameter(torch.Tensor(self.out_features, self.right_features))

        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_left)
        nn.init.xavier_uniform_(self.weight_right)
        # nn.init.constant_(self.bias, 0.)
        nn.init.xavier_uniform_(self.U)

    def forward(self, input_left, input_right):
        """
        Args:
            input_left: Tensor
                the left input tensor with shape = [batch1, batch2, ..., left_features]
            input_right: Tensor
                the right input tensor with shape = [batch1, batch2, ..., right_features]
        Returns:
        """

        batch_size = input_left.size()[:-1]
        batch = int(np.prod(batch_size))

        # convert left and right input to matrices [batch, left_features], [batch, right_features]
        input_left = input_left.view(batch, self.left_features)
        input_right = input_right.view(batch, self.right_features)

        # output [batch, out_features]
        output = F.bilinear(input_left, input_right, self.U, self.bias)
        output = output + F.linear(input_left, self.weight_left, None) + F.linear(input_right, self.weight_right, None)

        # convert back to [batch1, batch2, ..., out_features]
        return output.view(batch_size + (self.out_features, ))


class ArcBiaffine(nn.Module):
    def __init__(self, in_dim, out_dim=1):
        super(ArcBiaffine, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = Parameter(torch.Tensor(in_dim, in_dim))
        self.bias = Parameter(torch.Tensor(1))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, h_head, h_dep):
        weight_dep = torch.matmul(h_dep, self.weight.unsqueeze(dim=0))
        weight_bias = weight_dep + self.bias
        arc_scores = torch.matmul(h_head, torch.transpose(weight_bias, 2, 1))
        return arc_scores


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring :cite:`dozat-etal-2017-biaffine`.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y / d^s`,
    where `d` and `s` are vector dimension and scaling factor respectively.
    :math:`x` and :math:`y` can be concatenated with bias terms.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        scale (float):
            Factor to scale the scores. Default: 0.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    """

    def __init__(self, n_in, n_out=1, scale=0, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.scale = scale
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in+bias_x, n_in+bias_y))
        nn.init.xavier_uniform_(self.weight)

    def __repr__(self):
        s = f"n_in={self.n_in}"
        if self.n_out > 1:
            s += f", n_out={self.n_out}"
        if self.scale != 0:
            s += f", scale={self.scale}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1) / self.n_in ** self.scale

        return s


class DeepBiaffine(nn.Module):
    def __init__(self, vocab: Vocab, embedding_dim: int, mtl_dim: int, dep_mlp_dim: int, head_mlp_dim: int,
                 bilstm_dim: int, mtl_task: str, bilstm_layers: int):
        super(DeepBiaffine, self).__init__()
        self.vocab = vocab
        self.dep_mlp_dim = dep_mlp_dim
        self.head_mlp_dim = head_mlp_dim
        self.bilstm_dim = bilstm_dim
        self.mtl_task = mtl_task
        self.mtl_dim = mtl_dim
        self.embedding_dim = embedding_dim

        self.bilstm1 = nn.LSTM(self.embedding_dim, self.bilstm_dim, num_layers=1, bidirectional=True)
        self.bilstm2 = nn.LSTM(self.bilstm_dim * 2, self.bilstm_dim, num_layers=1, bidirectional=True)
        self.bilstm_mtl = nn.LSTM(self.bilstm_dim * 2, self.bilstm_dim, num_layers=bilstm_layers, bidirectional=True)

        self.mlp_arc_h = MLP(self.bilstm_dim * 2, self.head_mlp_dim)
        self.mlp_arc_d = MLP(self.bilstm_dim * 2, self.head_mlp_dim)
        self.mlp_rel_h = MLP(self.bilstm_dim * 2, self.dep_mlp_dim)
        self.mlp_rel_d = MLP(self.bilstm_dim * 2, self.dep_mlp_dim)

        self.arc_attn = Biaffine(n_in=self.head_mlp_dim, scale=0, bias_x=True, bias_y=False)
        self.rel_attn = Biaffine(n_in=self.dep_mlp_dim, n_out=self.vocab.deps_size, bias_x=True, bias_y=True)

        # MTL linear layers
        self.linear_mtl = nn.Linear(self.bilstm_dim * 2, self.mtl_dim)
        self.linear_pos = nn.Linear(self.mtl_dim, self.vocab.pos_size)
        self.linear_gender = nn.Linear(self.mtl_dim, self.vocab.gender_size)
        self.linear_number = nn.Linear(self.mtl_dim, self.vocab.number_size)
        self.linear_person = nn.Linear(self.mtl_dim, self.vocab.person_size)

        self.activation = nn.Tanh()
        self.embed_dropout = nn.Dropout(p=0.3)

    def forward(self, words_embeds, sentences_lens):
        ner_out = None
        pos_out = None
        gender_out = None
        number_out = None
        person_out = None

        self.embed_dropout(words_embeds)
        embedding = [words_embeds]

        bilstm_input = torch.cat(embedding, dim=2)
        bilstm_input_packed = pack_padded_sequence(bilstm_input, sentences_lens, batch_first=True, enforce_sorted=False)
        out, (ht, ct) = self.bilstm1(bilstm_input_packed)
        out_unpacked, _ = pad_packed_sequence(out, batch_first=True)

        if self.mtl_task is not None:
            out, (ht, ct) = self.bilstm_mtl(out)
            mtl_out_unpacked, _ = pad_packed_sequence(out, batch_first=True)

            mtl_out = self.linear_mtl(mtl_out_unpacked)

            if "pos" in self.mtl_task:
                pos_out = self.linear_pos(mtl_out).flatten(0, 1)
            if "gender" in self.mtl_task:
                gender_out = self.linear_gender(mtl_out).flatten(0, 1)
            if "number" in self.mtl_task:
                number_out = self.linear_number(mtl_out).flatten(0, 1)
            if "person" in self.mtl_task:
                person_out = self.linear_person(mtl_out).flatten(0, 1)

        out, (ht, ct) = self.bilstm2(out)
        mlp_input, _ = pad_packed_sequence(out, batch_first=True)

        arc_h = self.mlp_arc_h(mlp_input)
        arc_d = self.mlp_arc_d(mlp_input)
        rel_h = self.mlp_rel_h(mlp_input)
        rel_d = self.mlp_rel_d(mlp_input)

        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)

        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)
        s_rel = s_rel.flatten(0, 1)

        return s_rel, s_arc, pos_out, gender_out, number_out, person_out
