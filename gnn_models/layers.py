'''
Author: your name
Date: 2021-05-12 12:36:36
LastEditTime: 2021-05-14 13:18:30
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /dglke/gnn_models/layers.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv


def initializer(emb):
    emb.uniform_(-1.0, 1.0)
    return emb


class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.linear_func = nn.Linear(in_feats, out_feats, bias=False)
        self.attention_func = nn.Linear(2 * out_feats, 1, bias=False)

    def edge_attention(self, edges):
        concat_z = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        src_e = self.attention_func(concat_z)
        src_e = F.leaky_relu(src_e)
        return {'e', src_e}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # 边的注意力权重
        a = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(a * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, graph, h):
        z = self.linear_func(h)
        graph.ndata['z'] = z
        graph.apply_edges(self.edge_attention)
        graph.update_all(self.message_func, self.reduce_func)
        return graph.ndata.pop('h')


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embeddin = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases, num_hidden_layers=1, dropout=0.5, use_self_loop=False, use_cuda=False, reg_param=0):
        super(RGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.num_hidden_layers = num_hidden_layers
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.reg_param = reg_param
        self.rgcn_layers = nn.ModuleList()
        self.build_input_layer()
        self.build_hidden_layers()
        self.w_relations = nn.Parameter(torch.Tensor(num_rels, self.h_dim))
        nn.init.xavier_uniform_(
            self.w_relations, gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triples):
        # DistMult
        s = embedding[triples[:, 0]]
        r = self.w_relations[triples[:, 1]]
        o = embedding[triples[:, 2]]
        score = torch.sum(s*r*o, dim=1)
        return score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))+torch.mean(self.w_relations.pow(2))

    def get_loss(self, g, embed, triples, labels):
        score = self.calc_score(embed, triples)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss+self.reg_param*reg_loss

    def build_input_layer(self):
        self.rgcn_layers.append(EmbeddingLayer(self.num_nodes, self.h_dim))

    def build_hidden_layers(self):
        for idx in range(self.num_hidden_layers):
            act = F.relu() if idx < self.num_hidden_layers-1 else None
            rel_graph = RelGraphConv(self.h_dim, self.out_dim, self.num_rels,
                                     activation=act, self_loop=self.use_self_loop, dropout=self.dropout)
            self.rgcn_layers.append(rel_graph)

    def forward(self, g, h, r, norm):
        for layer in self.rgcn_layers:
            h = layer(g, h, r, norm)
        return h
def node_norm_to_edge_norm(g, node_norm):
    g =g.local_var()
    g.ndata['norm']=node_norm
    g.apply_edges(lambda edges : {'norm':edges.dst['norm']})
    return g.edata['norm']
