import torch
import pickle
from torch import nn
import torch.nn.functional as F

def adjust_learning_rate(optimizer, epoch, args):
	"""Sets the learning rate to the initial LR decayed every 3 epochs"""
	lr = args.learning_rate * (args.decay ** (epoch // args.stepsize))
	print("Current learning rate is: {:.5f}".format(lr))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

def generate_flip_grid(w, h):
	# used to flip attention maps
	x_ = torch.arange(w).view(1, -1).expand(h, -1)
	y_ = torch.arange(h).view(-1, 1).expand(-1, w)
	grid = torch.stack([x_, y_], dim=0).float().cuda()
	grid = grid.unsqueeze(0).expand(1, -1, -1, -1)  # [1,2,h,w]
	grid[:, 0, :, :] = 2 * grid[:, 0, :, :] / (w - 1) - 1
	grid[:, 1, :, :] = 2 * grid[:, 1, :, :] / (h - 1) - 1

	grid[:, 0, :, :] = -grid[:, 0, :, :]
	return grid

def load_train_per_class_num_pickle(path):
    '''
        读取train中每类的数量dict-pickle文件
    '''
    with open(path, 'rb') as f:
        per_class_num_dict = pickle.load(f)
    f.close()
    return per_class_num_dict

class FocalLoss(nn.Module):
    '''
        定义损失函数：用于类别不均衡和 类别易学与困难;alpha是从0开始到num结束
    '''
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)):
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)

        pt = logpt.data.exp()  # 因为log操作已经做了，下面就不需要了，而前面的需要去掉log，因此exp()，抵消log

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1-pt)**self.gamma * logpt  # 公式是 -1*alpha * (1-pt)**peta * log(pt)， 都需要经过softmax之后

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()




