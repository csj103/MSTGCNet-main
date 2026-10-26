"""Variable-preserving MSTGCNet variants E1--E5.

No learned projection contracts V. Only explicit adjacency propagation mixes V.
Routing aggregates per-variable scores but never replaces expert inputs.
"""
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from layers.GrangerPrior import expand_prior, load_granger_prior
from layers.RevIN import RevIN
from models.MSTGCNet import CCSTGCNExpert, GMoEBlock, PositionalEncoding, TrendSeasonalRouter


class VariableEncoder(nn.Module):
    def __init__(self, num_vars, d_var):
        super().__init__()
        self.temporal = nn.Conv1d(1, d_var, 3, padding=1, padding_mode='replicate')
        self.identity = nn.Parameter(torch.empty(num_vars, d_var))
        nn.init.normal_(self.identity, std=.02)

    def forward(self, x):
        b,l,v = x.shape
        hidden = self.temporal(x.transpose(1,2).reshape(b*v,1,l))
        hidden = hidden.reshape(b,v,-1,l).permute(0,3,1,2)
        return hidden + self.identity[None,None]


class VariableRouter(TrendSeasonalRouter):
    def _routing_features(self, x):
        b,l,v,d = x.shape
        independent = x.permute(0,2,1,3).reshape(b*v,l,d)
        scores, transformed = super()._routing_features(independent)
        # DFT/trend processing shares weights; aggregate before window Top-K/noise.
        scores = scores.reshape(b,v,l).mean(dim=1)
        transformed = transformed.reshape(b,v,l,d).permute(0,2,1,3)
        return scores, transformed


class VariableExpert(nn.Module):
    def __init__(self, args, patch_size, prior=None, max_lag=1):
        super().__init__()
        self.mode = args.experiment_mode
        self.seq_len, self.num_vars, self.d_var = args.seq_len, args.enc_in, args.d_var
        self.patch_size = patch_size
        self.num_patches = math.ceil(self.seq_len / patch_size)
        self.num_nodes = self.num_patches * self.num_vars
        self.knn_k = min(args.knn_k, max(self.num_nodes - 1, 1))
        self.lambda_prior = args.lambda_prior
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, patch_size))
        self.query = nn.Linear(patch_size * self.d_var, args.graph_dim, bias=False)
        self.key = nn.Linear(patch_size * self.d_var, args.graph_dim, bias=False)
        self.temporal_attn = nn.MultiheadAttention(
            self.d_var, max(1, math.gcd(args.attn_heads, self.d_var)),
            dropout=args.dropout, batch_first=True)
        self.graph_weight = nn.Parameter(torch.empty(patch_size, patch_size))
        self.graph_bias = nn.Parameter(torch.zeros(self.num_nodes, patch_size, self.d_var))
        nn.init.xavier_uniform_(self.graph_weight)
        self.norm_graph = nn.LayerNorm(self.d_var)
        self.norm_ffn = nn.LayerNorm(self.d_var)
        self.ffn = nn.Sequential(nn.Linear(self.d_var,args.d_ff), nn.GELU(),
                                 nn.Dropout(args.dropout), nn.Linear(args.d_ff,self.d_var))
        self.dropout = nn.Dropout(args.dropout)
        patch_ids = torch.arange(self.num_patches).repeat_interleave(self.num_vars)
        self.register_buffer('causal_mask',patch_ids[:,None] >= patch_ids[None,:])
        valid = torch.arange(self.num_patches*patch_size).reshape(self.num_patches,patch_size) < self.seq_len
        self.register_buffer('patch_valid',valid.repeat_interleave(self.num_vars,dim=0))
        if prior is None:
            prior = torch.zeros(self.num_vars,self.num_vars)
        self.register_buffer('expanded_prior',expand_prior(prior,self.num_patches,patch_size,max_lag))
        self.capture_graph = False
        self.last_adjacency = None

    def _patch(self,x):
        b,l,v,d = x.shape
        if (l,v,d) != (self.seq_len,self.num_vars,self.d_var):
            raise ValueError('Expert requires [B,L,V,d_var] in original variable order')
        extra = self.num_patches*self.patch_size-l
        if extra: x = torch.cat([x,x.new_zeros(b,extra,v,d)],dim=1)
        return x.reshape(b,self.num_patches,self.patch_size,v,d).permute(0,1,3,2,4).reshape(b,self.num_nodes,self.patch_size,d)

    def _unpatch(self,nodes):
        b=nodes.size(0)
        return nodes.reshape(b,self.num_patches,self.num_vars,self.patch_size,self.d_var).permute(0,1,3,2,4).reshape(b,-1,self.num_vars,self.d_var)[:,:self.seq_len]

    def _temporal_encode(self,nodes):
        b,n,p,d=nodes.shape
        x=nodes.reshape(b*n,p,d)
        padding=(~self.patch_valid).unsqueeze(0).expand(b,-1,-1).reshape(b*n,p)
        h,_=self.temporal_attn(x,x,x,key_padding_mask=padding,need_weights=False)
        # Retain original node information before graph propagation.
        h=(x+self.dropout(h)).reshape(b,n,p,d)
        return h*self.patch_valid[None,:,:,None]

    def _adjacency(self,nodes):
        if self.mode == 'E1':
            # Deliberately reproduce original fixed embeddings/binary Top-K.
            return CCSTGCNExpert._adjacency(self)
        if self.mode == 'E3':
            fixed=self.expanded_prior.clone()
            k=min(self.knn_k,self.num_nodes)
            val,idx=fixed.topk(k,dim=-1)
            fixed=torch.zeros_like(fixed).scatter(-1,idx,val)
            fixed=fixed+torch.eye(self.num_nodes,device=fixed.device,dtype=fixed.dtype)
            return fixed/fixed.sum(-1,keepdim=True).clamp_min(1e-12)
        flat=nodes.flatten(2)
        scores=torch.matmul(self.query(flat),self.key(flat).transpose(1,2))/math.sqrt(self.query.out_features)
        if self.mode in {'E4','E5'} and self.lambda_prior != 0:
            scores=scores+self.lambda_prior*self.expanded_prior
        valid=self.causal_mask.clone(); valid.fill_diagonal_(False)
        scores=scores.masked_fill(~valid,float('-inf'))
        # Preserve selected LOGITS, rather than binary scatter, for Q/K gradients.
        values,indices=scores.topk(self.knn_k,dim=-1)
        selected=torch.full_like(scores,float('-inf')).scatter(-1,indices,values)
        eye=torch.eye(self.num_nodes,device=nodes.device,dtype=torch.bool)
        selected=selected.masked_fill(eye,0.)  # guaranteed finite self path
        return torch.softmax(selected,dim=-1)

    def forward(self,x):
        nodes=self._temporal_encode(self._patch(x))
        adj=self._adjacency(nodes)
        b,n,p,d=nodes.shape
        batched=adj.unsqueeze(0).expand(b,-1,-1) if adj.ndim==2 else adj
        weights,idx=batched.topk(min(self.knn_k+1,n),dim=-1)
        # Sparse gather O(B*N*K*p*d), avoiding dense A @ H propagation.
        neighbors=nodes[torch.arange(b,device=x.device)[:,None,None],idx]
        graph=(neighbors*weights[:,:,:,None,None]).sum(dim=2)
        graph=torch.einsum('bnqd,qp->bnpd',graph,self.graph_weight)+self.graph_bias
        hidden=self._unpatch(nodes+self.dropout(F.gelu(graph)))
        hidden=self.norm_graph(hidden)
        hidden=self.norm_ffn(hidden+self.dropout(self.ffn(hidden)))
        if self.capture_graph: self.last_adjacency=batched[:1].detach().cpu()
        # GMoE does not need a full batch of retained autograd graph matrices.
        return hidden, None


class VariableGMoEBlock(GMoEBlock):
    def __init__(self,args,layer_idx,prior,max_lag):
        nn.Module.__init__(self)
        self.num_experts=args.num_experts_list[layer_idx]
        self.top_k=min(args.top_k,self.num_experts)
        sizes=args.patch_size_list[layer_idx]
        if len(sizes)!=self.num_experts:
            raise ValueError('One patch size per expert is required')
        self.router=VariableRouter(args.seq_len,args.d_var,self.num_experts,self.top_k,
            args.trend_kernel_sizes,args.seasonality_k,args.noisy_gating==1)
        self.experts=nn.ModuleList([VariableExpert(args,p,prior,max_lag) for p in sizes])


class VariableModel(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.enc_in,self.seq_len,self.d_model=args.enc_in,args.seq_len,args.d_var
        self.revin=bool(args.revin); self.loss_coef=args.loss_coef
        self.experiment_mode=args.experiment_mode
        from scripts.preprocess_alfa import FEATURE_COLUMNS
        features=list(getattr(args,'variable_names',FEATURE_COLUMNS))
        if len(features)!=args.enc_in: raise ValueError('Explicit ordered variable names required')
        self.prior_metadata={}; matrix=torch.zeros(args.enc_in,args.enc_in)
        if args.experiment_mode in {'E3','E4','E5'}:
            if not args.granger_prior: raise ValueError('A real training-derived --granger_prior is required')
            matrix,self.prior_metadata=load_granger_prior(args.granger_prior,features,
                allow_synthetic=getattr(args,'_unit_test_synthetic_prior',False))
        permutation=torch.arange(args.enc_in)
        if args.experiment_mode=='E5':
            permutation=torch.randperm(args.enc_in,generator=torch.Generator().manual_seed(args.prior_shuffle_seed))
            matrix=matrix[permutation][:,permutation]
        self.register_buffer('prior_permutation',permutation)
        self.register_buffer('prior_matrix',matrix)
        self.encoder=VariableEncoder(args.enc_in,args.d_var)
        self.position_embedding=PositionalEncoding(args.d_var)
        self.dropout=nn.Dropout(args.dropout)
        self.blocks=nn.ModuleList([VariableGMoEBlock(args,i,matrix,int(self.prior_metadata.get('max_lag',1))) for i in range(args.e_layers)])
        self.reconstruction=nn.Linear(args.d_var,1)
        if self.revin: self.revin_layer=RevIN(args.enc_in,affine=False)

    def forward(self,x_enc,x_mark=None,x_dec=None,x_mark_dec=None):
        x=self.revin_layer(x_enc,'norm') if self.revin else x_enc
        hidden=self.encoder(x)
        hidden=self.dropout(hidden+self.position_embedding(hidden)[:, :, None, :])
        balance=x_enc.new_zeros(())
        for block in self.blocks:
            hidden,aux,_=block(hidden); balance=balance+aux
        out=self.reconstruction(hidden).squeeze(-1)
        if self.revin: out=self.revin_layer(out,'denorm')
        return out,self.loss_coef*balance
