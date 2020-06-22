# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import random
import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import ipdb
st = ipdb.set_trace
import glob
from scipy.misc import imsave
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
			adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--dataset', dest='dataset',
											help='training dataset',
											default='clevr_trainval', type=str)
	parser.add_argument('--suffix', dest='suffix',
											help='training dataset',
											default='M', type=str)
	parser.add_argument('--net', dest='net',
										help='vgg16, res101',
										default='vgg16', type=str)
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--epochs', dest='max_epochs',
											help='number of epochs to train',
											default=1000, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=100, type=int)
	parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
											help='number of iterations to display',
											default=10000, type=int)

	parser.add_argument('--save_dir', dest='save_dir',
											help='directory to save models', default="models",
											type=str)
	parser.add_argument('--nw', dest='num_workers',
											help='number of worker to load data',
											default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',default=True,
											help='whether use CUDA',
											action='store_true')
	parser.add_argument('--aug', dest='aug',default=False,
										help='whether use CUDA')
	parser.add_argument('--ls', dest='large_scale',
											help='whether use large imag scale',
											action='store_true')                      
	parser.add_argument('--mGPUs', dest='mGPUs',
											help='whether use multiple GPUs',
											action='store_true')
	parser.add_argument('--bs', dest='batch_size',
											help='batch_size',
											default=5, type=int)
	parser.add_argument('--cag', dest='class_agnostic',
											help='whether perform class_agnostic bbox regression',
											action='store_true',default=True)
	parser.add_argument('--o', dest='optimizer',
											help='training optimizer',
											default="sgd", type=str)
	parser.add_argument('--lr', dest='lr',
											help='starting learning rate',
											default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
											help='step to do learning rate decay, unit is epoch',
											default=0.0, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
											help='learning rate decay ratio',
											default=0.0, type=float)
	parser.add_argument('--s', dest='session',
											help='training session',
											default=1, type=int)
	parser.add_argument('--r', dest='resume',
											help='resume checkpoint or not',
											default=False, type=bool)
	parser.add_argument('--checksession', dest='checksession',
											help='checksession to load model',
											default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
											help='checkepoch to load model',
											default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
											help='checkpoint to load model',
											default=0, type=int)
	parser.add_argument('--use_tfb', dest='use_tfboard',
											help='whether use tensorboard',
											default= True,
											action='store_true')

	args = parser.parse_args()
	return args


class sampler(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size)
		self.batch_size = batch_size
		self.range = torch.arange(0,batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
		self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1)

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data

def crop_objs(img,boxes,num_boxes,cropped_list):
	for index,img_val in enumerate(img):
		num_val = num_boxes[index]
		box_val = boxes[index][:num_val]
		for box in box_val:
			box  = box.to(torch.int)
			xmin,ymin,xmax,ymax,_ = torch.unbind(box)
			# ymin,xmin,ymax,xmax,_ = torch.unbind(box)
			crp = img_val[:,ymin:ymax,xmin:xmax]
			cropped_list.append(crp)
	return cropped_list


def create_binary_mask(boxes,shape):
		canvas = torch.zeros(shape)
		for box in boxes:
				box = box.to(torch.int)
				xmin,ymin,xmax,ymax,_ = torch.unbind(box)
				canvas[ymin:ymax,xmin:xmax] = 1
		return canvas

def sample_boxes(b_mask,num_objects,shape_val,shape_tensor):
		bboxes = []
		for i in range(num_objects):
				shape_curr = list(shape_val[i])
				ysize,xsize = shape_curr
				intersect = True
				box = None
				while intersect:
						x_min = random.randint(0, (shape_tensor[1]-xsize-1))
						y_min = random.randint(0, (shape_tensor[0]-ysize-1))
						x_max = x_min + xsize
						y_max = y_min + ysize
						int_voxels = torch.sum(b_mask[y_min:y_max,x_min:x_max]).cpu().numpy()
						if int_voxels == 0:
							intersect = False
							box = [x_min,y_min,x_max,y_max,1.0]
							b_mask[y_min:y_max,x_min:x_max] = 1
				bboxes.append(box)
		return np.array(bboxes)

def update_scene_with_object_crops(img,objects,boxes):
		boxes = torch.clamp(boxes,min=0)
		index_to_take = -1
		for index_box,box in enumerate(boxes):
			box = box.to(torch.int)
			index_to_take += 1
			xmin,ymin,xmax,ymax,_ = torch.unbind(box)
			size = [ymax-ymin,xmax-xmin]
			obj_add = objects[index_box].unsqueeze(0)
			try:
				obj_add = torch.nn.functional.interpolate(obj_add,size).squeeze()
			except Exception as e:
				raise Exception('hello')
			img[:,ymin:ymax,xmin:xmax] = obj_add
		return img

def save_rgb(tensor_val,name,boxes=None):
	tensor_val = tensor_val.permute(1,2,0).cpu().numpy()
	if boxes is not None:
		for box in boxes:
			xmin,ymin,xmax,ymax,_ = torch.unbind(box)
			tensor_val = cv2.rectangle(tensor_val, (int(xmin), int(ymin)), (int(xmax), int(ymax)),(151, 0, 255),2)
	imsave(f'{name}.png',tensor_val)


if __name__ == '__main__':
	args = parse_args()
	cropped_objs = []

	print('Called with args:')
	print(args)

	if args.dataset == "pascal_voc":
			args.imdb_name = "voc_2007_trainval"
			args.imdbval_name = "voc_2007_test"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "clevr_trainval":
			args.imdb_name = "clevr_trainval"
			args.imdbval_name = "clevr_test"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "clevrvqa":
			args.imdb_name = f"clevrvqa_train{args.suffix}"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "carla":
			args.imdb_name = f"carla_train{args.suffix}"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']			
	elif args.dataset == "pascal_voc_0712":
			args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
			args.imdbval_name = "voc_2007_test"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "coco":
			args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
			args.imdbval_name = "coco_2014_minival"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	elif args.dataset == "imagenet":
			args.imdb_name = "imagenet_train"
			args.imdbval_name = "imagenet_val"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	elif args.dataset == "vg":
			# train sizes: train, smalltrain, minitrain
			# train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
			args.imdb_name = "vg_150-50-50_minitrain"
			args.imdbval_name = "vg_150-50-50_minival"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

	args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	#torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = args.cuda
	# st()
	imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
	train_size = len(roidb)

	print('{:d} roidb entries'.format(len(roidb)))
	if args.aug:
		output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "_" +args.suffix + "_" + "aug"
	else:
		output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + "_" +args.suffix
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	sampler_batch = sampler(train_size, args.batch_size)

	dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
													 imdb.num_classes, training=True)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
														sampler=sampler_batch, num_workers=args.num_workers)

	# initilize the tensor holder here.
	im_data = torch.FloatTensor(1)
	im_info = torch.FloatTensor(1)
	num_boxes = torch.LongTensor(1)
	gt_boxes = torch.FloatTensor(1)

	# ship to cuda
	if args.cuda:
		im_data = im_data.cuda()
		im_info = im_info.cuda()
		num_boxes = num_boxes.cuda()
		gt_boxes = gt_boxes.cuda()

	# make variable
	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.cuda:
		cfg.CUDA = True

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
	elif args.net == 'res101':
		fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res50':
		fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res152':
		fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
	else:
		print("network is not defined")
		pdb.set_trace()

	fasterRCNN.create_architecture()

	lr = cfg.TRAIN.LEARNING_RATE
	lr = args.lr
	#tr_momentum = cfg.TRAIN.MOMENTUM
	#tr_momentum = args.momentum
	# st()

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
								'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

	if args.cuda:
		fasterRCNN.cuda()


	if args.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params ,weight_decay=1e-5)

	elif args.optimizer == "sgd":
		# optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM, weight_decay=1e-5)

	if args.resume:
		load_name = os.path.join(output_dir,
			'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		args.session = checkpoint['session']
		args.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		lr = optimizer.param_groups[0]['lr']
		if 'pooling_mode' in checkpoint.keys():
			cfg.POOLING_MODE = checkpoint['pooling_mode']
		print("loaded checkpoint %s" % (load_name))

	if args.mGPUs:
		fasterRCNN = nn.DataParallel(fasterRCNN)

	iters_per_epoch = int(train_size / args.batch_size)
	# st()
	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		folder_name = f"logs/{args.imdb_name}"
		try:
			os.mkdir(folder_name)
		except Exception:
			pass
		logger = SummaryWriter(folder_name)

	all_crops = []
	all_shapes = []
	if args.aug:
		collectSamp = True
		while collectSamp:
			data_iter = iter(dataloader)
			for step in range(iters_per_epoch):
				if not collectSamp:
					break
				data = next(data_iter)
				with torch.no_grad():
						im_data.resize_(data[0].size()).copy_(data[0])
						im_info.resize_(data[1].size()).copy_(data[1])
						gt_boxes.resize_(data[2].size()).copy_(data[2])
						num_boxes.resize_(data[3].size()).copy_(data[3])
				cropped_list = crop_objs(im_data,gt_boxes,num_boxes,[])
				# st()
				for ind,crop_val in enumerate(cropped_list):
					# print(crop.shape)
					# st()
					if args.dataset == "clevrvqa":
						shape_val = (torch.tensor(crop_val.shape[1:]).to(torch.float)*random.uniform(0.3,0.6)).to(torch.int)
					else:
						shape_val = (torch.tensor(crop_val.shape[1:]).to(torch.float)*random.uniform(0.8,1.2)).to(torch.int)
					all_shapes.append(shape_val)
					crop_img = crop_val.permute(1,2,0).cpu().numpy()
					all_crops.append(crop_img)
					
					# imsave(f'tmp/cropimg_{len(all_crops)}.png',crop_img)
					# st()
					# print(len(all_crops))
					if len(all_crops) ==int(args.suffix):
						collectSamp = False
						break
		# [save_rgb(obj,f'tmp/all_crops_et{ind}.png')for ind,obj in enumerate(all_crops)]
		# st()
		# all_shapes = [i.shape for i in all_crops]
	import random
	for epoch in range(args.start_epoch, args.max_epochs + 1):
		# setting to train mode
		# st()
		fasterRCNN.train()
		loss_temp = 0
		start = time.time()
		data_iter = iter(dataloader)
		# if epoch % (args.lr_decay_step + 1) == 0:
		# 		adjust_learning_rate(optimizer, args.lr_decay_gamma)
		# 		lr *= args.lr_decay_gamma

		for step in range(iters_per_epoch):
			data = next(data_iter)
			with torch.no_grad():
							im_data.resize_(data[0].size()).copy_(data[0])
							im_info.resize_(data[1].size()).copy_(data[1])
							gt_boxes.resize_(data[2].size()).copy_(data[2])
							num_boxes.resize_(data[3].size()).copy_(data[3])
			all_aug_bboxes = []
			all_aug_img = []
			all_bboxes = []
			all_num_boxes = [] 
			# st()
			if args.aug:
				for i,img_data_e in enumerate(im_data):
					gt_boxes_e =gt_boxes[i]
					num_boxes_e =num_boxes[i]
					gt_boxes_e = gt_boxes_e[:num_boxes_e]
					num_objects = 1
					# num_objects = 4
					num_boxes_upd = num_objects +num_boxes_e
					shapes = [random.choice(all_shapes)[:] for i in range(num_objects)]
					b_mask = create_binary_mask(gt_boxes_e,list(img_data_e.shape[1:]))
					# st()
					aug_bboxes = torch.from_numpy(sample_boxes(b_mask,num_objects,shapes,img_data_e.shape[1:]))
					# st()
					objects_being_replaced_i = [random.choice(range(len(all_crops))) for i in range(num_objects)]
					objects_being_replaced = [all_crops[ind] for ind in objects_being_replaced_i]
					# [save_rgb(obj,f'tmp/rep_{objects_being_replaced_i[ind]}.png')for ind,obj in enumerate(objects_being_replaced)]
					# [imsave(f'tmp/all_crops{ind}.png',obj)for ind,obj in enumerate(objects_being_replaced)]
					objects_being_replaced_cuda = [torch.from_numpy(val).cuda().permute(2,0,1) for val in objects_being_replaced]
					img_data_ec = img_data_e.clone()
					try:
						aug_img = update_scene_with_object_crops(img_data_e,objects_being_replaced_cuda,aug_bboxes)
					except Exception as e:
						st()
					gt_boxes_aug = torch.cat([aug_bboxes.to(torch.float),gt_boxes_e.cpu().to(torch.float)],dim=0).numpy()
					save_rgb(aug_img,f"tmp/whole_{i}",boxes=aug_bboxes)
					save_rgb(img_data_ec,f"tmp/og_whole_{i}")
					
					# for index,gt_box in enumerate(gt_boxes_aug):
					# 	xmin,ymin,xmax,ymax,_ =gt_box.astype(np.int)
					# 	aug_img_crop = aug_img[:,ymin:ymax,xmin:xmax]
					# 	save_rgb(aug_img_crop,f'part_{index}')
					# print('check')
					gt_boxes_aug = np.pad(gt_boxes_aug,[[0,cfg.MAX_NUM_GT_BOXES - gt_boxes_aug.shape[0]],[0,0]])
					all_bboxes.append(gt_boxes_aug)
					all_aug_img.append(aug_img)
					all_aug_bboxes.append(torch.from_numpy(gt_boxes_aug))
					all_num_boxes.append(num_boxes_upd)
				all_aug_bboxes = torch.stack(all_aug_bboxes).cuda()
				all_aug_img = torch.stack(all_aug_img).cuda()
				all_num_boxes = torch.stack(all_num_boxes).cuda()
				# st()
				gt_boxes = all_aug_bboxes
				im_data = all_aug_img
				num_boxes = all_num_boxes

			fasterRCNN.zero_grad()
			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

			loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
					 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			loss_temp += loss.item()
			# backward
			optimizer.zero_grad()
			loss.backward()
			if args.net == "vgg16":
					clip_gradient(fasterRCNN, 10.)
			optimizer.step()
			# st()
			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					loss_temp /= (args.disp_interval + 1)

				if args.mGPUs:
					loss_rpn_cls = rpn_loss_cls.mean().item()
					loss_rpn_box = rpn_loss_box.mean().item()
					loss_rcnn_cls = RCNN_loss_cls.mean().item()
					loss_rcnn_box = RCNN_loss_bbox.mean().item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				else:
					loss_rpn_cls = rpn_loss_cls.item()
					loss_rpn_box = rpn_loss_box.item()
					loss_rcnn_cls = RCNN_loss_cls.item()
					loss_rcnn_box = RCNN_loss_bbox.item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" \
																% (args.session, epoch, step, iters_per_epoch, loss_temp, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" \
											% (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))
				if args.use_tfboard:
					info = {
						'loss': loss_temp,
						'loss_rpn_cls': loss_rpn_cls,
						'loss_rpn_box': loss_rpn_box,
						'loss_rcnn_cls': loss_rcnn_cls,
						'loss_rcnn_box': loss_rcnn_box
					}
					logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * iters_per_epoch + step)

				loss_temp = 0
				start = time.time()
		save_name = os.path.join(output_dir, 'faster_rcnn_{:08d}.pth'.format(epoch))
		all_files = glob.glob(output_dir + "/*")
		all_files.sort()
		files_to_remove = all_files[:-3]
		for file in files_to_remove:
			os.remove(file)
		# st()
		if epoch % 30 ==0:
			save_checkpoint({
				'session': args.session,
				'epoch': epoch + 1,
				'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
				'optimizer': optimizer.state_dict(),
				'pooling_mode': cfg.POOLING_MODE,
				'class_agnostic': args.class_agnostic,
			}, save_name)
		print('save model: {}'.format(save_name))

	if args.use_tfboard:
		logger.close()