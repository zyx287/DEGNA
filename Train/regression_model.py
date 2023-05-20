import torch
import torch.nn as nn
from torch_scatter import scatter_max
from typing import Tuple


class EGNA(nn.Module):
    """
    EGNA (Empirical Graph neural Network for protein-ligand binding Affinity prediction) predicts
    the pKd of complexes.
    """
    def __init__(
            self,
            evo_node_attr_dim: int,#蛋白质进化特征
            lig_node_attr_dim: int#配体特征
    ):
        """
        :param evo_node_attr_dim: The dimension of the input evolutionary features for each residue.
        :param lig_node_attr_dim: The dimension of the input ligand features for each atom.
        """
        super().__init__()
        self.in_prot_mlp = nn.Sequential(
            #处理蛋白质中的节点特征
            nn.Linear(evo_node_attr_dim, 128),#进化特征映射到一个维度为 128
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.in_lig_mlp = nn.Sequential(
            nn.Linear(lig_node_attr_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.p_gcl = GCL(128, 256)#蛋白质特征和邻接矩阵作为输入
        self.l_gcl = GCL(128, 256)#处理配体特征

        self.t_list = (1, 3, 6, 9, 12)#5 个元素的元组，分别代表 EIRL 中用到的时间间隔
        n_t = len(self.t_list)#计算时间间隔的数量

        self.eirl = EIRL(256, 256, 512, n_t)#输出为更新后的配体特征
        self.eirl_2 = EIRL(512, 512, 512, n_t)#增强网络的表达能力

        out_hidden_dim = 512#输出特征维度

        self.embed_mlp = nn.Sequential(
            #节点特征映射到一个低维空间
            nn.Linear(2048, out_hidden_dim),
            nn.BatchNorm1d(out_hidden_dim),
            nn.LeakyReLU(1e-2, inplace=True),#非线性拟合
            nn.Dropout(0.5),
        )
        self.out_mlp = nn.Linear(out_hidden_dim, 1)#映射为一个标量值pKd

    def forward(
            self,
            px: torch.Tensor,
            lx: torch.Tensor,
            pp_mat: torch.Tensor,
            pl_mat: torch.Tensor,
            ll_mat: torch.Tensor,
            res_idx: torch.Tensor,
            prot_g_idx: torch.Tensor,
            lig_g_idx: torch.Tensor
    ):
        """
        :param px: The input protein vertex features
        :param lx: The input ligand vertex features
        :param pp_mat: The adjacency matrix of the scaffold region in the protein
        :param pl_mat: The adjacency matrix of the interaction between the binding pocket and the ligand
        :param ll_mat: The adjacency matrix of the ligand
        :param res_idx: The indices of the pocket residues in the scaffold
        :param prot_g_idx: The indices used to discriminate different proteins in a batch
        :param lig_g_idx: The indices used to discriminate different ligands in a batch
        :return: predicted pKd
        """
        # MLP
        px = self.in_prot_mlp(px)#节点特征输入到 MLP
        lx = self.in_lig_mlp(lx)
        # GNN
        px = self.p_gcl(px, pp_mat)# MLP 处理后的蛋白质节点特征和邻接矩阵输入到 GCL
        lx = self.l_gcl(lx, ll_mat)

        # fetch binding pocket
        ix = px[res_idx, :]#蛋白质节点特征中属于配体结合位点的部分

        pl_mat_list = [pl_mat ** t for t in self.t_list]#配体结合位点邻接矩阵的不同时间幂次方
        ix_g, lx_g = self.eirl(ix, lx, pl_mat_list)#配体节点特征与配体结合位点邻接矩阵的幂次方输入到 EIRL 网络中
        ix_g2, lx_g2 = self.eirl_2(ix_g, lx_g, pl_mat_list)#deepen

        ix_cat = torch.cat((ix_g, ix_g2), dim=1)#沿着特征维度拼接起来
        lx_cat = torch.cat((lx_g, lx_g2), dim=1)

        # Readout节点特征按照 batch 进行切分，计算每个 batch 中节点特征的最大值
        ix_max, _ = scatter_max(ix_cat, prot_g_idx, dim=0)
        lx_max, _ = scatter_max(lx_cat, lig_g_idx, dim=0)

        ilx = torch.cat((ix_max, lx_max), dim=1)#节点特征最大值沿着特征维度拼接起来
        q_embed = self.embed_mlp(ilx)
        out = self.out_mlp(q_embed).squeeze(1)

        return out


class GCL(nn.Module):
    """
    GCL (Graph Convolutional Layer) is used to represent proteins/ligands
    """
    def __init__(
            self,
            input_dim: int,
            output_dim: int
    ):
        """
        :param input_dim: The dimension of the input embedding
        :param output_dim: The dimension of the output embedding
        """
        super(GCL, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

    def forward(
            self,
            x: torch.Tensor,
            am: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x: Vertex embedding
        :param am: Adjacency matrix
        :return: updated vertex embedding
        """
        x_sum = torch.sparse.mm(am, x)#稀疏矩阵乘法（torch.sparse.mm()）计算邻居节点的特征聚合结果
        x = self.mlp(x_sum)#邻居节点的特征聚合结果输入到 MLP 网络中，生成更新后的节点特征
        return x


class ExchangeLayer(nn.Module):
    """
    Exchange information between residues in proteins and atoms in ligands
    """
    def __init__(
            self,
            in_prot_dim: int,
            in_lig_dim: int,
            out_dim: int
    ):
        """
        :param in_prot_dim: The dimension of the input protein embedding
        :param in_lig_dim: The dimension of the input ligand embedding
        :param out_dim: The dimension of the output protein/ligand embedding
        """
        super(ExchangeLayer, self).__init__()

        self.nonlinear = nn.Sequential(
            nn.Linear(in_prot_dim + in_lig_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
            self,
            x1: torch.Tensor,
            x2: torch.Tensor,
            am: torch.Tensor
    ) -> torch.Tensor:
        """
        :param x1: An input embedding
        :param x2: Another input embedding
        :param am: An adjacency matrices
        :return: output embedding
        """
        x1 = torch.sparse.mm(am, x1)
        x = self.nonlinear(torch.cat((x1, x2), dim=1))
        return x


class PseudoEnergyTerm(nn.Module):
    """
    Pseudo energy terms are used to represent the interaction under a specific interaction adjacency matrix.
    """
    def __init__(
            self,
            in_prot_dim: int,
            in_lig_dim: int,
            out_dim: int
    ):
        """
        :param in_prot_dim: The dimension of the input protein embedding
        :param in_lig_dim: The dimension of the input ligand embedding
        :param out_dim: The dimension of the output protein/ligand embedding
        """
        super(PseudoEnergyTerm, self).__init__()
        self.pl_gcl = ExchangeLayer(in_prot_dim, in_lig_dim, out_dim)
        self.lp_gcl = ExchangeLayer(in_prot_dim, in_lig_dim, out_dim)

    def forward(
            self,
            px: torch.Tensor,
            lx: torch.Tensor,
            pl_mat: torch.Tensor
    ):
        """
        :param px: The input protein embedding
        :param lx: The input ligand embedding
        :param pl_mat: The adjacency matrices of the interaction.
        :return: A Tuple contains the output protein and ligand embedding
        """
        px_p = self.lp_gcl(px, lx, pl_mat.t())
        lx_p = self.pl_gcl(lx, px, pl_mat)
        return px_p, lx_p


class EIRL(nn.Module):
    """
    The EIRL (Empricial Interaction Representation Layer) is used to represent the protein-ligand interaction
    like an empirical scoring function
    """
    def __init__(
            self,
            in_prot_dim: int,
            in_lig_dim: int,
            out_dim: int,
            n_exp: int
    ):
        """
        :param in_prot_dim: The dimension of the input protein embedding
        :param in_lig_dim: The dimension of the input ligand embedding
        :param out_dim: The dimension of the output protein/ligand embedding
        :param n_exp: The number of the types of the exponents
        """
        super(EIRL, self).__init__()
        """
        包含多个 PseudoEnergyTerm 的 ModuleList 
        其中每个 PseudoEnergyTerm 均接收输入蛋白质和配体节点特征，以及一个幂次方邻接矩阵，输出对应的更新后的节点特征
        """
        self.terms = nn.ModuleList([PseudoEnergyTerm(in_prot_dim, in_lig_dim, out_dim) for _ in range(n_exp)])

        self.weight_p = nn.Conv1d(n_exp, 1, (1,))
        self.weight_l = nn.Conv1d(n_exp, 1, (1,))

    def forward(
            self,
            px: torch.Tensor,
            lx: torch.Tensor,
            pl_mat_list: list
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param px: The input protein embedding
        :param lx: The input ligand embedding
        :param pl_mat_list: The list of adjacency matrices which have various exponents.
        :return: A Tuple contains the output protein and ligand embedding
        """
        px_p_list, lx_p_list = [], []
        for i, pl_mat in enumerate(pl_mat_list):
            px_p, lx_p = self.terms[i](px, lx, pl_mat)
            px_p_list.append(lx_p)
            lx_p_list.append(px_p)

        px_pl = torch.stack(px_p_list, dim=1)
        lx_lp = torch.stack(lx_p_list, dim=1)

        px_pl_rep = self.weight_p(px_pl).squeeze(1)
        lx_lp_rep = self.weight_l(lx_lp).squeeze(1)

        return px_pl_rep, lx_lp_rep
