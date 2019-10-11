import torch
import numpy as np

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append((correct_k.mul_(100. / batch_size)).item())  # 添加 .item() 因为是tensor
    # print(res) # 按照百分制 输出的
    return np.array(res)

def test(model, test_loader, epoch):
	'''
	测试代码，计算输出每个epoch的平均精度
	:param model:
	:param test_loader:
	:return:
	'''
	print("###### testing ... ######")
	# metrics initialization
	batches = 0
	# epoch_loss = 0
	epoch_acc = np.array([0, 0, 0], dtype='float')  # top - 1, 3, 5

	for i, sample in enumerate(test_loader):
		images = sample[0]	# test just large
		labels = sample[-1]

		images = images.cuda()
		labels = labels.cuda()

		test_input = images
		y_pred = model(test_input, is_train=False)

		epoch_acc += accuracy(y_pred, labels, topk=(1, 3, 5))
		# end of this batch
		batches += 1

		# print('Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f' % (epoch_acc[0]/batches, epoch_acc[1]/batches, epoch_acc[2]/batches))

	epoch_acc /= batches

	print('\n')
	print('>>>>>>>>>>>>>>>>>>>>>>>> Accuracy for %d epoch >>>>>>>>>>>>>>>>>>>>>>>>>>>'%epoch )
	print('Accuracy: Top-1 %.2f, Top-3 %.2f, Top-5 %.2f'%(epoch_acc[0], epoch_acc[1], epoch_acc[2]) )
	print('\n')
	return



