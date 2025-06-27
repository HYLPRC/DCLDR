import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease


class AttentionFusion(nn.Module):
    def __init__(self, dim, vector_level=False):
        """
        :param dim: 单个特征的维度，比如你的是 gt_out_dim
        :param vector_level: 是否做逐维 attention（True 则更细致，默认 False）
        """
        super(AttentionFusion, self).__init__()
        self.vector_level = vector_level
        if vector_level:
            # 输出一个与 a/b 同维度的权重向量
            self.attn = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Tanh(),
                nn.Linear(dim, dim),
                nn.Sigmoid()
            )
        else:
            # 输出一个 scalar 权重（[0,1]）
            self.attn = nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.Tanh(),
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )

    def forward(self, a, b):
        """
        :param a: 结构特征 tensor [N, dim]
        :param b: 语义特征 tensor [N, dim]
        :return: 加权融合后的表示 tensor [N, dim]
        """
        fused_input = torch.cat([a, b], dim=-1)  # [N, 2*dim]
        w = self.attn(fused_input)               # [N, 1] 或 [N, dim]
        if not self.vector_level:
            w = w.expand_as(a)                   # 扩展成 [N, dim]
        return w * a + (1 - w) * b               # 融合

class DCLDR(nn.Module):
    def __init__(self, args):
        super(DCLDR, self).__init__()
        self.args = args
        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.drug_emb = nn.Embedding(args.drug_number, args.hgt_in_dim)
        self.disease_emb = nn.Embedding(args.disease_number, args.hgt_in_dim)
        self.protein_emb = nn.Embedding(args.protein_number, args.hgt_in_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(self.args.device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(self.args.device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout)
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout)
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer-1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        
        self.drug_attn_fusion = AttentionFusion(dim=args.gt_out_dim, vector_level=False)
        self.disease_attn_fusion = AttentionFusion(dim=args.gt_out_dim, vector_level=False)

        self.mlp = nn.Sequential(
            nn.Linear(args.gt_out_dim * 4, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

        self.gate_layer = nn.Sequential(
            nn.Linear(4 * args.gt_out_dim, 2 * args.gt_out_dim),
            nn.Sigmoid()
        )



    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        # print("CUDA device:", torch.cuda.current_device())
        # print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
        # print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
        # print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")

        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)

        # drug_feature = self.drug_emb(torch.arange(self.args.drug_number).to(self.args.device))
        # disease_feature = self.disease_emb(torch.arange(self.args.disease_number).to(self.args.device))
        # protein_feature = self.protein_emb(torch.arange(self.args.protein_number).to(self.args.device))


        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:self.args.disease_number+self.args.drug_number, :]

        dr = torch.stack((dr_hgt, dr_sim), dim=1)
        di = torch.stack((di_hgt, di_sim), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        # attention

        dr = self.drug_attn_fusion(dr_hgt, dr_sim)        # [num_drugs, dim]
        di = self.disease_attn_fusion(di_hgt, di_sim)     # [num_diseases, dim]
        
        a = dr[sample[:, 0]]
        b = di[sample[:, 1]]
        drdi_embedding = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)  # 

        # mul *2
        # drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])

        # cat *4
        # drdi_embedding = torch.cat([dr[sample[:, 0]], di[sample[:, 1]]], dim=-1)

        # Hadamard + absolute difference + concat *8
        # a = dr[sample[:, 0]]
        # b = di[sample[:, 1]]
        # drdi_embedding = torch.cat([a, b, torch.abs(a - b), a * b], dim=-1)

        # gated 
        # a = dr[sample[:, 0]]
        # b = di[sample[:, 1]]
        # fusion = torch.cat([a, b], dim=-1)
        # gate = self.gate_layer(fusion)  # self.gate = nn.Linear(4 * dim, 2 * dim)
        # drdi_embedding = gate * a + (1 - gate) * b

        output = self.mlp(drdi_embedding)

        return dr_hgt, di_hgt, dr_sim, di_sim, output

