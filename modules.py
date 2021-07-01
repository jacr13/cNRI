import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import numpy as np


import torch
from torch.autograd import Variable
import torch.distributions as tdist

from utils import my_softmax, get_offdiag_indices, gumbel_softmax

_EPS = 1e-10


class MLP(nn.Module):
    """Two-layer fully-connected ELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if len(inputs.shape) > 2:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.bn(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        elif len(inputs.shape) == 2:
            return self.bn(inputs)
        else:
            raise NotImplementedError(
                "Batchnorm for these dimensions not implemented.")

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.elu(self.fc1(inputs))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class MLP3(nn.Module):
    """Three-layer fully-connected RELU net with batch norm."""

    def __init__(self, n_in, n_hid, n_out, do_prob=0.):
        super(MLP3, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, 2 * n_hid)
        self.fc3 = nn.Linear(2 * n_hid, n_out)
        self.dropout_prob = do_prob

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.dropout(F.relu(self.fc1(inputs)), p=self.dropout_prob)
        x = F.dropout(F.relu(self.fc2(x)), p=self.dropout_prob)
        return self.fc3(x)


class MLPEncoder(nn.Module):
    def __init__(self, n_in, n_hid, n_out, do_prob=0., factor=True):
        super(MLPEncoder, self).__init__()

        self.factor = factor
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
        self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
        if self.factor:
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            #print("Using factor graph MLP encoder.")
        else:
            self.mlp4 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
            #print("Using MLP encoder.")
        self.fc_out = nn.Linear(n_hid, n_out)
        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.

        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]

        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]
        x = self.mlp1(x)  # 2-layer ELU net per node
        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        if self.factor:
            x = self.edge2node(x, rel_rec, rel_send)
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        else:
            x = self.mlp3(x)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
        return self.fc_out(x)


class MLPEncoder_multi(nn.Module):
    def __init__(self, n_in, n_hid, edge_types_list, do_prob=0., split_point=1,
                 init_type='xavier_normal', bias_init=0.1):
        super(MLPEncoder_multi, self).__init__()

        self.edge_types_list = edge_types_list
        self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
        self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)

        self.init_type = init_type
        if self.init_type not in ['xavier_normal', 'orthogonal', 'sparse']:
            raise ValueError('This initialization type has not been coded')

        self.bias_init = bias_init

        self.split_point = split_point
        if split_point == 0:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = MLP(n_hid * 3, n_hid, n_hid, do_prob)
            self.fc_out = nn.ModuleList(
                [nn.Linear(n_hid, sum(edge_types_list))])
        elif split_point == 1:
            self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
            self.mlp4 = nn.ModuleList(
                [MLP(n_hid * 3, n_hid, n_hid, do_prob)
                 for _ in edge_types_list])
            self.fc_out = nn.ModuleList([nn.Linear(n_hid, K)
                                         for K in edge_types_list])
        elif split_point == 2:
            self.mlp3 = nn.ModuleList(
                [MLP(n_hid, n_hid, n_hid, do_prob)
                 for _ in edge_types_list])
            self.mlp4 = nn.ModuleList(
                [MLP(n_hid * 3, n_hid, n_hid, do_prob)
                 for _ in edge_types_list])
            self.fc_out = nn.ModuleList(
                [nn.Linear(n_hid, K)
                 for K in edge_types_list])
        else:
            raise ValueError('Split point is not valid, must be 0, 1, or 2')

        self.init_weights()

    @property
    def device(self):
        return next(self.parameters()).device

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data)
                elif self.init_type == 'xavier_normal':
                    nn.init.xavier_normal_(m.weight.data)
                elif self.init_type == 'sparse':
                    nn.init.sparse_(m.weight.data, sparsity=0.1)

                if not math.isclose(self.bias_init, 0, rel_tol=1e-9):
                    m.bias.data.fill_(self.bias_init)

    def edge2node(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)
        return incoming / incoming.size(1)

    def node2edge(self, x, rel_rec, rel_send):
        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([receivers, senders], dim=2)
        return edges

    def forward(self, inputs, rel_rec, rel_send):
        # Input shape: [num_sims, num_atoms, num_timesteps, num_dims]
        x = inputs.view(inputs.size(0), inputs.size(1), -1)
        # New shape: [num_sims, num_atoms, num_timesteps*num_dims]

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self.node2edge(x, rel_rec, rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self.edge2node(x, rel_rec, rel_send)
        if self.split_point == 0:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            x = self.mlp4(x)
            return self.fc_out[0](x)
        elif self.split_point == 1:
            x = self.mlp3(x)
            x = self.node2edge(x, rel_rec, rel_send)
            x = torch.cat((x, x_skip), dim=2)  # Skip connection
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp4[i](x)
                y_list.append(self.fc_out[i](y))
            return torch.cat(y_list, dim=-1)
        elif self.split_point == 2:
            y_list = []
            for i in range(len(self.edge_types_list)):
                y = self.mlp3[i](x)
                y = self.node2edge(y, rel_rec, rel_send)
                y = torch.cat((y, x_skip), dim=2)  # Skip connection
                y = self.mlp4[i](y)
                y_list.append(self.fc_out[i](y))
            return torch.cat(y_list, dim=-1)


class MLPEncoder_SD(nn.Module):
    def __init__(self, skeleton):
        super(MLPEncoder_SD, self).__init__()
        self.PA = nn.Parameter(torch.from_numpy(skeleton.astype(np.float32)))
        nn.init.constant_(self.PA, 1e-6)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, inputs):
        x = torch.repeat_interleave(
            self.PA[np.newaxis, :, :], inputs.shape[0], dim=0)
        return x


class RNNDecoder(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, n_atoms, n_clinical, edge_types, n_hid,
                 cond_hidden, cond_msgs,
                 do_prob=0., skip_first=False):
        super(RNNDecoder, self).__init__()
        self.n_atoms = n_atoms
        self.cond_hidden = cond_hidden
        self.cond_msgs = cond_msgs
        self.n_hid = n_hid
        self.edge_types = edge_types

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])

        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        if self.cond_hidden:
            self.clinical_mlp = MLP3(
                n_clinical, n_hid, n_atoms * n_hid, do_prob)

        if self.cond_msgs:
            self.clinical2msg_mlp = MLP3(
                n_clinical, n_hid, n_atoms * n_hid, do_prob)

        self.in_r = nn.Linear(n_in_node, n_hid)
        self.in_z = nn.Linear(n_in_node, n_hid)
        self.in_n = nn.Linear(n_in_node, n_hid)

        self.hr = nn.Linear(n_hid, n_hid)
        self.hz = nn.Linear(n_hid, n_hid)
        self.hn = nn.Linear(n_hid, n_hid)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        self.dropout_prob = do_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def init_hidden(self, batch_size, clinical_data):
        if self.cond_hidden:
            hidden = self.clinical_mlp(clinical_data)
            hidden = hidden.view(batch_size,
                                 self.n_atoms,
                                 self.msg_out_shape)
        else:
            hidden = Variable(torch.zeros(batch_size,
                                          self.n_atoms,
                                          self.msg_out_shape))
        return hidden

    def step(self, inputs, hidden, rel_rec, rel_send, rel_type, clinical_data):
        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.msg_fc2)) - 1.
        else:
            start_idx = 0
            norm = float(len(self.msg_fc2))

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1
        for i in range(start_idx, len(self.msg_fc2)):
            msg = torch.tanh(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = torch.tanh(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg / norm

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous() / inputs.size(2)  # Average

        if self.cond_msgs:
            cln_emb = self.clinical2msg_mlp(clinical_data)
            cln_emb = cln_emb.view(inputs.size(0),
                                   inputs.size(1),
                                   self.msg_out_shape)
            agg_msgs = agg_msgs + cln_emb

        r = torch.sigmoid(self.in_r(inputs) + self.hr(agg_msgs))
        z = torch.sigmoid(self.in_z(inputs) + self.hr(agg_msgs))
        n = torch.tanh(self.in_n(inputs) + r * self.hn(agg_msgs))
        hidden = (1 - z) * n + z * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/vlocity difference
        pred = inputs + pred
        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, clinical):
        inputs = data.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        # Initializing hidden state with clinical data
        hidden = self.init_hidden(batch_size, clinical)
        hidden = hidden.to(inputs.device)

        pred_all = []

        for step in range(0, time_steps-1):
            if step == 0:
                x = inputs[:, 0, :, :]
                pred_all.append(x)
            else:
                x = pred_all[-1]

            pred, hidden = self.step(
                x, hidden, rel_rec, rel_send, rel_type, clinical)

            pred_all.append(pred)

        predictions = torch.stack(pred_all, dim=1)
        return predictions.transpose(1, 2).contiguous()


class RNNDecoder_multi(nn.Module):
    """Recurrent decoder module."""

    def __init__(self, n_in_node, n_atoms, n_clinical, edge_types,
                 edge_types_list, n_hid, cond_hidden, cond_msgs,
                 do_prob=0., skip_first=False):
        super(RNNDecoder_multi, self).__init__()
        self.n_atoms = n_atoms
        self.edge_types = edge_types
        self.edge_types_list = edge_types_list
        self.cond_hidden = cond_hidden
        self.cond_msgs = cond_msgs
        self.n_hid = n_hid

        self.msg_fc1 = nn.ModuleList(
            [nn.Linear(2 * n_hid, n_hid) for _ in range(edge_types)])
        self.msg_fc2 = nn.ModuleList(
            [nn.Linear(n_hid, n_hid) for _ in range(edge_types)])

        self.msg_out_shape = n_hid
        self.skip_first_edge_type = skip_first

        if self.cond_hidden:
            self.clinical_mlp = MLP3(
                n_clinical, n_hid, n_atoms * n_hid, do_prob)

        if self.cond_msgs:
            self.clinical2msg_mlp = MLP3(
                n_clinical, n_hid, n_atoms * n_hid, do_prob)

        # GRU
        self.in_r = nn.Linear(n_in_node, n_hid)
        self.in_z = nn.Linear(n_in_node, n_hid)
        self.in_n = nn.Linear(n_in_node, n_hid)

        self.hr = nn.Linear(n_hid, n_hid)
        self.hz = nn.Linear(n_hid, n_hid)
        self.hn = nn.Linear(n_hid, n_hid)

        self.out_fc1 = nn.Linear(n_hid, n_hid)
        self.out_fc2 = nn.Linear(n_hid, n_hid)
        self.out_fc3 = nn.Linear(n_hid, n_in_node)

        #print('Using learned recurrent interaction net decoder.')

        self.dropout_prob = do_prob

    @property
    def device(self):
        return next(self.parameters()).device

    def init_hidden(self, batch_size, clinical_data):
        if self.cond_hidden:
            hidden = self.clinical_mlp(clinical_data)
            hidden = hidden.view(batch_size,
                                 self.n_atoms,
                                 self.msg_out_shape)
        else:
            hidden = Variable(torch.zeros(batch_size,
                                          self.n_atoms,
                                          self.msg_out_shape)
                              ).to(clinical_data.device)
        return hidden

    def step(self, inputs, hidden, rel_rec, rel_send, rel_type, clinical_data):

        # node2edge
        receivers = torch.matmul(rel_rec, hidden)
        senders = torch.matmul(rel_send, hidden)
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = Variable(torch.zeros(pre_msg.size(0), pre_msg.size(1),
                                        self.msg_out_shape))

        if inputs.is_cuda:
            all_msgs = all_msgs.cuda()

        non_null_idxs = list(range(self.edge_types))
        if self.skip_first_edge_type:
            edge = 0
            for k in self.edge_types_list:
                non_null_idxs.remove(edge)
                edge += k

        # Run separate MLP for every edge type
        # NOTE: To exclude one edge type, simply offset range by 1

        for i in non_null_idxs:
            msg = F.relu(self.msg_fc1[i](pre_msg))
            msg = F.dropout(msg, p=self.dropout_prob)
            msg = F.relu(self.msg_fc2[i](msg))
            msg = msg * rel_type[:, :, i:i + 1]
            all_msgs += msg

        agg_msgs = all_msgs.transpose(-2, -1).matmul(rel_rec).transpose(-2, -1)
        agg_msgs = agg_msgs.contiguous()

        if self.cond_msgs:
            cln_emb = self.clinical2msg_mlp(clinical_data)
            cln_emb = cln_emb.view(inputs.size(0),
                                   inputs.size(1),
                                   self.msg_out_shape)
            agg_msgs = agg_msgs + cln_emb

        # GRU-style gated aggregation
        r = torch.sigmoid(self.in_r(inputs) + self.hr(agg_msgs))
        z = torch.sigmoid(self.in_z(inputs) + self.hr(agg_msgs))
        n = torch.tanh(self.in_n(inputs) + r * self.hn(agg_msgs))
        hidden = (1 - z) * n + z * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.out_fc1(hidden)), p=self.dropout_prob)
        pred = F.dropout(F.relu(self.out_fc2(pred)), p=self.dropout_prob)
        pred = self.out_fc3(pred)

        # Predict position/velocity difference
        pred = inputs + pred
        return pred, hidden

    def forward(self, data, rel_type, rel_rec, rel_send, clinical):
        inputs = data.transpose(1, 2).contiguous()
        time_steps = inputs.size(1)
        batch_size = inputs.size(0)

        # Initializing hidden state with clinical data
        hidden = self.init_hidden(batch_size, clinical)

        pred_all = []

        for step in range(0, time_steps-1):
            if step == 0:
                x = inputs[:, 0, :, :]
                pred_all.append(x)
            else:
                x = pred_all[-1]

            pred, hidden = self.step(
                x, hidden, rel_rec, rel_send, rel_type, clinical)

            pred_all.append(pred)

        predictions = torch.stack(pred_all, dim=1)
        return predictions.transpose(1, 2).contiguous()
