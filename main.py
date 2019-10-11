import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.backends import cudnn
from torch.autograd import Variable
from dataset import get_subsets
from utils import adjust_learning_rate, generate_flip_grid, FocalLoss, load_train_per_class_num_pickle
from test import test
import numpy as np
import argparse
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def get_parser():
	parser = argparse.ArgumentParser(description = 'CNN Attention Consistency')
	parser.add_argument("--arch", default="resnet50", type=str,
		help="ResNet architecture")

	parser.add_argument("--num_class", default=2296, type=int,
						help="class's number")
	parser.add_argument('--train_batch_size', default = 4, type = int,
		help = 'default training batch size')
	parser.add_argument('--train_workers', default = 8, type = int,
		help = '# of workers used to load training samples')
	parser.add_argument('--test_batch_size', default = 48, type = int,
		help = 'default test batch size')
	parser.add_argument('--test_workers', default = 8, type = int,
		help = '# of workers used to load testing samples')

	parser.add_argument('--learning_rate', default = 0.001, type = float,
		help = 'base learning rate')
	parser.add_argument('--momentum', default = 0.9, type = float,
		help = "set the momentum")
	parser.add_argument('--weight_decay', default = 0.0005, type = float,
		help = 'set the weight_decay')
	parser.add_argument('--stepsize', default = 3, type = int,
		help = 'lr decay each # of epoches')
	parser.add_argument('--decay', default=0.5, type=float,
		help = 'update learning rate by a factor')

	parser.add_argument('--model_dir',
		default = './ckpt_new_aug',
		type = str,
		help = 'path to save checkpoints')
	parser.add_argument('--model_prefix',
		default = 'model',
		type = str,
		help = 'model file name starts with')

	# optimizer
	parser.add_argument('--optimizer',
		default = 'SGD',
		type = str,
		help = 'Select an optimizer: TBD')

	# general parameters
	parser.add_argument('--epoch_max', default = 30, type = int,
		help = 'max # of epcoh')
	parser.add_argument('--display', default = 1000, type = int,
		help = 'display')
	parser.add_argument('--snapshot', default = 1, type = int,
		help = 'snapshot')
	parser.add_argument('--start_epoch', default = 1, type = int,
		help = 'resume training from specified epoch')
	parser.add_argument('--resume', default = '/home/gengyanlei/Python_work/VAC/ckpt_new_aug/model_resnet50_epoch1.pth', type = str,
		help = 'resume training from specified model state')

	parser.add_argument('--test', default = True, type = bool,
		help = 'conduct testing after each checkpoint being saved')

	return parser

def main():
	parser = get_parser()
	args = parser.parse_args()

	# load data
	trainset, testset = get_subsets(size1=(224,224), size2=(192,192))

	train_loader = torch.utils.data.DataLoader(trainset,
		batch_size = args.train_batch_size,
		shuffle = True,
		num_workers = args.train_workers)
	test_loader = torch.utils.data.DataLoader(testset,
		batch_size = args.test_batch_size,
		shuffle = False,
		num_workers = args.test_workers)

	# path to save models
	if not os.path.isdir(args.model_dir):
		print("Make directory: " + args.model_dir)
		os.makedirs(args.model_dir)

	# prefix of saved checkpoint
	model_prefix = args.model_dir + '/' + args.model_prefix

	# define the model: use ResNet50 as an example
	if args.arch == "resnet50":
		from resnet import resnet50
		model = resnet50(pretrained=True, num_labels=args.num_class)  # 用来测试 训练时
		model_prefix = model_prefix + "_resnet50"
	elif args.arch == "resnet101":
		from resnet import resnet101
		model = resnet101(pretrained=True, num_labels=args.num_class)
		model_prefix = model_prefix + "_resnet101"
	else:
		raise NotImplementedError("To be implemented!")
	# 判断是否需要继续训练
	if args.start_epoch != 0:
		resume_model = torch.load(args.resume)
		resume_dict = resume_model.state_dict()
		model_dict = model.state_dict()
		resume_dict = {k:v for k,v in resume_dict.items() if k in model_dict and k.size() == model_dict[k].size() }
		model_dict.update(resume_dict)
		model.load_state_dict(model_dict)  # 重新导入 更新后的字典
		print('继续训练')

	# 多GPU并行训练
	cudnn.benchmark = True
	model.cuda()
	model = nn.DataParallel(model)

	# 选择优化器optimizer
	if args.optimizer == 'Adam':
		optimizer = optim.Adam(
			model.parameters(),
			lr = args.learning_rate
		)
	elif args.optimizer == 'SGD':
		optimizer = optim.SGD(
			model.parameters(),
			lr = args.learning_rate,
			momentum = args.momentum,
			weight_decay = args.weight_decay
		)
	else:
		raise NotImplementedError("Not supported yet!")

	# training the network
	model.train()

	# attention map size
	size1, size2 = 7, 6
	w1 = size1
	h1 = size1
	grid_l = generate_flip_grid(w1, h1)

	w2 = size2
	h2 = size2
	grid_s = generate_flip_grid(w2, h2)

	# least common multiple
	lcm = w1 * w2

	##################################
	# 根据训练集中，每类数量，计算alpha
	##################################
	per_class_num_dict = load_train_per_class_num_pickle(path='/home/ailab/dataset/new_data/per_class_details/train_per_class_num.pickle')
	# 确保从0到num_classes
	alpha_list = []
	for i in range(args.num_class):
		per_class_num = per_class_num_dict[i]
		if per_class_num == 1:
			per_class_num = 1.1
		alpha_list.append(per_class_num)
	alpha_array = np.array(alpha_list)

	alpha_array = (1 / np.log(alpha_array))
	# for i in range(args.num_class):
	#     if alpha_array[i] > 0.5:
	#         alpha_array[i] = alpha_array[i] / 2
	alpha = alpha_array.tolist()
	alpha = [round(alpha_i, 4) for alpha_i in alpha]

	criterion = FocalLoss(2, alpha=alpha, size_average=True)
	criterion_mse = nn.MSELoss(size_average = True)

	for epoch in range(args.start_epoch, args.epoch_max):
		epoch_start = time.clock()
		if not args.stepsize == 0:
			adjust_learning_rate(optimizer, epoch, args)

		# num1 = 0
		for step, batch_data in enumerate(train_loader):
			# if num1 >10:
			# 	print('############')
			# 	model.eval()
			# 	test(model, test_loader, epoch + 1)
			# 	model.train()
			# 	break
			# num1 += 1

			batch_images_lo = batch_data[0]
			batch_images_lf = batch_data[1]
			batch_images_so = batch_data[2]
			batch_images_sf = batch_data[3]
			batch_images_lc = batch_data[4]  # color 变化, 需要和images_lo 合并
			batch_labels = batch_data[-1]

			batch_images_l = torch.cat((batch_images_lo, batch_images_lf))
			batch_images_c = torch.cat((batch_images_lo, batch_images_lc))  # color
			batch_images_s = torch.cat((batch_images_so, batch_images_sf))
			batch_labels = torch.cat((batch_labels, batch_labels, batch_labels, batch_labels, batch_labels, batch_labels))  # 6个

			batch_images_l = batch_images_l.cuda()
			batch_images_c = batch_images_c.cuda()  # color
			batch_images_s = batch_images_s.cuda()
			batch_labels = batch_labels.cuda()

			inputs_l = batch_images_l
			inputs_c = batch_images_c  # color
			inputs_s = batch_images_s
			labels = batch_labels

			output_l, hm_l = model(inputs_l)
			output_c, hm_c = model(inputs_c)  # color
			output_s, hm_s = model(inputs_s)

			output = torch.cat((output_l, output_s, output_c))
			# output = torch.cat((output_l, output_s))
			loss = criterion(output, labels)

			# flip
			num = hm_l.size(0) // 2  #单独split 按照batch维度，那么这个数需要大于等于一半  小于整体

			hm1, hm2 = hm_l.split(num)
			flip_grid_large = grid_l.expand(num, -1, -1, -1)
			flip_grid_large = Variable(flip_grid_large, requires_grad = False)
			flip_grid_large = flip_grid_large.permute(0, 2, 3, 1)
			hm2_flip = F.grid_sample(hm2, flip_grid_large, mode = 'bilinear',
				padding_mode = 'border')
			flip_loss_l = F.mse_loss(hm1, hm2_flip)  # no size_average

			hm1_small, hm2_small = hm_s.split(num)
			flip_grid_small = grid_s.expand(num, -1, -1, -1)
			flip_grid_small = Variable(flip_grid_small, requires_grad = False)
			flip_grid_small = flip_grid_small.permute(0, 2, 3, 1)
			hm2_small_flip = F.grid_sample(hm2_small, flip_grid_small, mode = 'bilinear',
				padding_mode = 'border')
			flip_loss_s = F.mse_loss(hm1_small, hm2_small_flip)

			# color 变化 对比
			hm1, hm2 = hm_c.split(num)
			# color_loss = torch.FloatTensor([0])
			color_loss = F.mse_loss(hm1, hm2)  # no size_average

			# scale loss
			num = hm_l.size(0)
			hm_l = F.upsample(hm_l, lcm)
			hm_s = F.upsample(hm_s, lcm)
			scale_loss = F.mse_loss(hm_l, hm_s)

			losses = loss + flip_loss_l + flip_loss_s + color_loss + scale_loss
			# losses = loss + flip_loss_l + flip_loss_s + scale_loss

			optimizer.zero_grad()
			losses.backward()
			optimizer.step()

			if (step) % args.display == 0:
				print(
					'epoch: {},\ttrain step: {}\tLoss: {:.6f}'.format(epoch+1,
					step, losses.item())
				)
				print(
					'\tcls loss: {:.4f};\tflip_loss_l: {:.4f}'
					'\tflip_loss_s: {:.4f};\tcolor_loss: {:.4f};\tscale_loss: {:.4f}'.format(
						loss.item(),
						flip_loss_l.item(),
						flip_loss_s.item(),
						color_loss.item(),
						scale_loss.item()
					)
				)

		epoch_end = time.clock()
		elapsed = epoch_end - epoch_start
		print("Epoch time: ", elapsed)

		# test
		if (epoch+1) % args.snapshot == 0:
			model_file = model_prefix + '_epoch{}.pth'
			print("Saving model to " + model_file.format(epoch+1))
			torch.save(model, model_file.format(epoch+1))

			if args.test:
				model.eval()
				test(model, test_loader, epoch+1)
				model.train()                    ###########测试完后，需要进入train模式，记住###############

	final_model =model_prefix + '_final.pth'
	print("Saving model to " + final_model)
	torch.save(model, final_model)
	model.eval()
	test(model, test_loader, epoch+1)



if __name__ == '__main__':
	main()
