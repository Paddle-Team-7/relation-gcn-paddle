#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-27 11:14:53
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

#import torch
import paddle
#from torch.nn import Parameter
#from torch.nn.modules.module import Module
#from torch import nn
import paddle.nn as nn
#import torch.nn.functional as F
import paddle.nn .functional as F
import math
# USE_CUDA=torch.cuda.is_available()
# device=torch.device("cuda" if USE_CUDA else "cpu")
def _calculate_fan_in_and_fan_out(tensor):
		dimensions = tensor.dim()
		if dimensions < 2:
			raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

		num_input_fmaps = tensor.shape[1]
		num_output_fmaps = tensor.shape[0]
		receptive_field_size = 1
		if tensor.dim() > 2:
			# math.prod is not always available, accumulate the product manually
			# we could use functools.reduce but that is not supported by TorchScript
			for s in tensor.shape[2:]:
				receptive_field_size *= s
		fan_in = num_input_fmaps * receptive_field_size
		fan_out = num_output_fmaps * receptive_field_size

		return fan_in, fan_out

class RGCLayer(nn.Layer):
	def __init__(self,input_dim,h_dim,supprot,num_base,featureless,drop_prob):
		super(RGCLayer,self).__init__()
		self.num_base=num_base
		self.input_dim=input_dim
		self.supprot=supprot
		self.h_dim=h_dim
		self.featureless=featureless
		self.drop_prob=drop_prob
		if num_base>0:
			#self.W=Parameter(torch.empty(input_dim*self.num_base,h_dim,dtype=torch.float32,device=device)) # requires_grad=True
			px = paddle.empty([input_dim*self.num_base,h_dim],dtype='float32')
			self.W = paddle.create_parameter(shape=px.shape,dtype=str(px.numpy().dtype),
										default_initializer=paddle.nn.initializer.Assign(px))
			self.W.stop_gradient = False
			#self.W_comp=Parameter(torch.empty(supprot,num_base,dtype=torch.float32,device=device))
			px1 = paddle.empty([supprot,num_base],dtype='float32')
			self.W_comp = paddle.create_parameter(shape=px1.shape,dtype=str(px1.numpy().dtype),
										default_initializer=paddle.nn.initializer.Assign(px1))
			self.W_comp.stop_gradient = False
		else:
			#self.W=Parameter(torch.empty(input_dim*self.supprot,h_dim,dtype=torch.float32,device=device))
			px2 = paddle.empty([input_dim*self.supprot,h_dim],dtype='float32')
			self.W = paddle.create_parameter(shape=px2.shape,dtype=str(px2.numpy().dtype),
										default_initializer=paddle.nn.initializer.Assign(px2))
			self.W.stop_gradient = False

		#self.B=Parameter(torch.FloatTensor(h_dim))
		px3 = paddle.empty([h_dim],dtype='float32')
		self.B  = paddle.create_parameter(shape=px3.shape,dtype=str(px3.numpy().dtype),
									default_initializer=paddle.nn.initializer.Assign(px3))
		self.B.stop_gradient = False
		#self.reset_parameters()

	
	
	def reset_parameters(self):
		#nn.init.xavier_uniform_(self.W)
		fan_in, fan_out = _calculate_fan_in_and_fan_out(self.W)
		nn.initializer.XavierUniform(fan_in,fan_out)

		if self.num_base>0:
			#nn.init.xavier_uniform_(self.W_comp)
			fan_in, fan_out = _calculate_fan_in_and_fan_out(self.W_comp)
			nn.initializer.XavierUniform(fan_in,fan_out)
		#self.B.data.fill_(0)
		resultb = paddle.full(shape=self.B.detach().shape,fill_value=0)
		self.B = resultb.detach()

	def forward(self,vertex,A):
		supports=[]
		nodes_num=A[0].shape[0]
		for i,adj in enumerate(A):
			if not self.featureless:
				#supports.append(torch.spmm(adj,vertex))
				supports.append(paddle.matmul(adj,vertex))
			else:
				supports.append(adj)
		supports=paddle.concat(supports,axis=1)
		if self.num_base>0:
			#V=paddle.matmul(self.W_comp,paddle.reshape(self.W,(self.num_base,self.input_dim,self.h_dim)).permute(1,0,2))
			aa = paddle.reshape(self.W,(self.num_base,self.input_dim,self.h_dim))
			aa = paddle.transpose(aa,perm=[1,0,2])
			V = paddle.matmul(self.W_comp,aa)

			V = paddle.reshape(V,(self.input_dim*self.supprot,self.h_dim))
			#output=torch.spmm(supports,V)
			output = paddle.matmul(supports,V)
		else:
			#output=torch.spmm(supports,self.W)
			output=paddle.matmul(supports,self.W)
		if self.featureless:
			temp=paddle.ones([nodes_num])
			temp_drop=F.dropout(temp,self.drop_prob)
			output=(output.transpose(1,0)*temp_drop).transpose(1,0)
		output+=self.B
		return output

