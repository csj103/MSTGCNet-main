import torch
import torch.nn as nn
import numpy as np
import math

import torch.fft as fft
from einops import rearrange, reduce, repeat


class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts

        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        # assigns samples to experts whose gate is nonzero
        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), expert_out[-1].size(2), expert_out[-1].size(3),
                            requires_grad=True, device=stitched.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        # add eps to all zero values in order to avoid nans when going back to log space
        combined[combined == 0] = np.finfo(float).eps
        # back to log space
        return combined.log()
    def expert_to_gates(self):
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(MLP, self).__init__()
        self.fc = nn.Conv2d(in_channels=input_size,
                             out_channels=output_size,
                             kernel_size=(1, 1),
                             bias=True)

    def forward(self, x):
        out = self.fc(x)
        return out



class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1)
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp_multi, self).__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            moving_avg = func(x)
            moving_mean.append(moving_avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1)
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):

    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        # self.d_model = d_model
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""

        if self.output_attention:
            return self.dft_forward(x)

        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(torch.arange(t + self.pred_len, dtype=torch.float),
                          't -> () () t ()').to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)))
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]

        return x_freq, index_tuple

    def dft_forward(self, x):
        T = x.size(1)

        dft_mat = fft.fft(torch.eye(T))
        i, j = torch.meshgrid(torch.arange(self.pred_len + T), torch.arange(T))
        omega = torch.exp(2 * math.pi * 1j / T)
        idft_mat = (torch.pow(omega, i * j) / T).cfloat()

        x_freq = torch.einsum('ft,btd->bfd', [dft_mat, x.cfloat()])

        if T % 2 == 0:
            x_freq = x_freq[:, self.low_freq:T // 2]
        else:
            x_freq = x_freq[:, self.low_freq:T // 2 + 1]

        _, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        indices = indices + self.low_freq
        indices = torch.cat([indices, -indices], dim=1)

        dft_mat = repeat(dft_mat, 'f t -> b f t d', b=x.shape[0], d=x.shape[-1])
        idft_mat = repeat(idft_mat, 't f -> b t f d', b=x.shape[0], d=x.shape[-1])

        mesh_a, mesh_b = torch.meshgrid(torch.arange(x.size(0)), torch.arange(x.size(2)))

        dft_mask = torch.zeros_like(dft_mat)
        dft_mask[mesh_a, indices, :, mesh_b] = 1
        dft_mat = dft_mat * dft_mask

        idft_mask = torch.zeros_like(idft_mat)
        idft_mask[mesh_a, :, indices, mesh_b] = 1
        idft_mat = idft_mat * idft_mask

        attn = torch.einsum('bofd,bftd->botd', [idft_mat, dft_mat]).real
        return torch.einsum('botd,btd->bod', [attn, x]), rearrange(attn, 'b o t d -> b d o t')
    
class PriorLoss(nn.Module):  
    def __init__(self, loss_type='distance', lambda_distance=1.0, lambda_sparsity=1.0, lambda_similarity=1.0):  
        """  
        :param loss_type: 选择的损失类型 ('distance', 'sparsity', 'similarity')  
        :param lambda_distance: 距离正则化项的权重  
        :param lambda_sparsity: 稀疏性正则化项的权重  
        :param lambda_similarity: 相似性正则化项的权重  
        """  
        super(PriorLoss, self).__init__()  
        self.loss_type = loss_type  
        self.lambda_distance = lambda_distance  
        self.lambda_sparsity = lambda_sparsity  
        self.lambda_similarity = lambda_similarity  

    def forward(self, A, K):  
        """  
        计算所选损失  
        
        :param A: 学习到的邻接矩阵 (tensor)，形状为 (N, N)  
        :param K: 先验邻接矩阵 (tensor)，形状为 (N, N)  
        :return: 计算出的损失值  
        """  
        if self.loss_type == 'distance':  
            # 基于距离的正则化 (Frobenius范数)  
            loss = self.lambda_distance * torch.norm(A - K, p='fro')**2  
        elif self.loss_type == 'sparsity':  
            # 基于稀疏性的正则化 (L1范数)  
            mask = 1 - K  
            loss = self.lambda_sparsity * torch.norm(A * mask, p=1)  
        elif self.loss_type == 'similarity':  
            # 基于相似性的正则化  
            loss = -self.lambda_similarity * torch.sum(K * torch.log(A + 1e-10))  # 避免log(0)  
        else:  
            raise ValueError("loss_type must be one of ['distance', 'sparsity', 'similarity']")  
        
        return loss

class AMSDispatcher(object):
    """
    AMSDispatcher 类：实现 AMS 风格的混合专家分派功能。
    根据 gating 权重将输入分派给各个专家，并在合并专家输出时自动跳过没有有效输入的专家。
    """
    def __init__(self, num_experts, gates):
        """
        初始化 AMSDispatcher。
        参数:
            num_experts: 专家数量
            gates: gating 权重，形状为 [batch_size, num_experts]
        """
        self.num_experts = num_experts
        self.gates = gates
        self.batch_size = gates.size(0)
        
        # 参照 SparseDispatcher 的实现
        # 找出所有非零门控值的位置
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # 获取每个专家对应的样本索引
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # 计算每个专家分配的样本数量
        self._part_sizes = (gates > 0).sum(0).tolist()
        # 获取非零门控值
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """
        根据 SparseDispatcher 分派规则，将输入张量分派给各个专家。
        参数:
            inp: 输入张量，形状通常为 [batch_size, ...]
        返回:
            expert_inputs: 长度为 num_experts 的列表，每个元素为对应专家的输入张量
        """
        # 如果没有非零门控值，处理边界情况
        if self._batch_index.numel() == 0:
            empty_expert_inputs = []
            # 确保即使是空输入也有正确的维度
            input_shape = list(inp.shape)
            input_shape[0] = 0  # 设置第一维为0
            for _ in range(self.num_experts):
                empty_expert_inputs.append(torch.zeros(input_shape, dtype=inp.dtype, device=inp.device))
            return empty_expert_inputs
            
        # 参照 SparseDispatcher 的实现
        # 根据批次索引选择样本并进行处理
        inp_exp = inp[self._batch_index.flatten()]
        
        # 按专家分割输入数据
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_outputs, multiply_by_gates=True):
        """
        合并各专家的输出，参照 SparseDispatcher 的实现。
        参数:
            expert_outputs: 长度为 num_experts 的列表，每个元素为对应专家的输出张量
            multiply_by_gates: 是否将专家输出乘以对应的 gating 分值
        返回:
            combined: 合并后的输出张量，其形状为 [batch_size, ...]
        """
        # 如果专家输出为空，处理边界情况
        if not expert_outputs or all(e.numel() == 0 for e in expert_outputs):
            # 找到一个有效的输出形状
            for i, out in enumerate(expert_outputs):
                if out.numel() > 0:
                    output_shape = list(out.shape)
                    output_shape[0] = self.batch_size
                    return torch.zeros(output_shape, device=self.gates.device)
            
            # 如果所有专家输出都为空，返回形状与输入相符的零张量
            return torch.zeros((self.batch_size, 1), device=self.gates.device)
        
        # 参照 SparseDispatcher 的实现
        # 拼接所有专家输出
        stitched = torch.cat(expert_outputs, dim=0)
        
        # 如果需要乘以门控值
        if multiply_by_gates:
            # 处理多维输出的情况
            if stitched.dim() > 2:
                # 扩展门控值的维度以匹配输出
                nonzero_gates = self._nonzero_gates
                for _ in range(stitched.dim() - 2):
                    nonzero_gates = nonzero_gates.unsqueeze(-1)
                stitched = stitched * nonzero_gates
            else:
                stitched = stitched * self._nonzero_gates
        
        # 创建输出张量
        zeros = torch.zeros(
            self.batch_size, 
            *stitched.shape[1:], 
            device=stitched.device, 
            dtype=stitched.dtype
        )
        
        # 使用 index_add 累加专家输出
        combined = zeros.index_add(0, self._batch_index, stitched)
        
        return combined