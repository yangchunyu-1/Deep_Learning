import numpy as np
from utils import im2col,col2im

#卷积层的实现
class Convolution:
    def __init__(self,W,b,stride=1,pad=0):
        self.W = W  #滤波器（权重）
        self.b = b  #偏置
        self.stride = stride  #步幅
        self.pad = pad  #填充

        self.x = None
        self.col = None
        self.col_W = None

        self.dW = None
        self.db = None

    def forward(self,x):
        FN,C,FH,FW = self.W.shape
        N,C,H,W = x.shape
        #计算输出特征图的高和宽
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FW) / self.stride)

        col = im2col(x,FH,FW,self.stride,self.pad)
        # 滤波器的展开，原本 (FN, C, FH, FW) 的权重被重塑为 (FN, C*FH*FW)，执行 .T（转置），使其形状变为 (C*FH*FW, FN)
        col_W = self.W.reshape(FN,-1).T
        out = np.dot(col,col_W) + self.b

        # transpose 函数会更改多维数组轴的顺序
        out = out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self,dout):
        FN,C,FH,FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1,FN)

        self.db = np.sum(dout,axis=0)
        self.dW = np.dot(self.col.T,dout)
        self.dW = self.dW.transpose(1,0).reshape(FN,C,FH,FW)

        dcol = np.dot(dout,self.col_W.T)
        dx = col2im(dcol,self.x.shape,FH,FW,self.stride,self.pad)

        return dx

#池化层的实现
class Pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None  #最大值的索引

    def forward(self,x):
        N,C,H,W = x.shape
        #池化层的使用中不使用填充
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        #展开
        col = im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        #让每一行代表一个池化窗口内的所有像素点
        col = col.reshape(-1,self.pool_h*self.pool_w)

        #最大值
        arg_max = np.argmax(col,axis=1)
        out = np.max(col,axis=1)

        #转换
        out = out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)

        self.x = x
        self.arg_max = arg_max
        return out

    def backward(self,dout):
        dout = dout.transpose(0,2,3,1)

        pool_size = self.pool_h * self.pool_w
        #创建一个全0矩阵存放反向传播的梯度
        dmax = np.zeros((dout.size,pool_size))
        #只在arg_max处填入dout梯度
        dmax[np.arange(self.arg_max.size),self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2],-1)
        dx = col2im(dcol,self.x.shape,self.pool_h,self.pool_w,self.stride,self.pad)

        return dx

#补充：axis=0为纵向，axis=1为横向