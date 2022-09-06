from __future__ import absolute_import, division, print_function
from datetime import datetime
import numpy as np
import math
import time
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import hflip
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import json
from my_utils import *
import torchvision
from utils import *
from kitti_utils import *
from layers import *
import datasets
import networks
from IPython import embed

full_res_shape = (608, 968)
# full_res_shape = (1200, 1600)

corr_loss = CorrelationLoss()

class Trainer:
    def __init__(self, options):
        now = datetime.now()
        # current_time_date = now.strftime("%d%m%Y_%H%M%S")
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda:0")#not using cuda?
        self.num_scales = len(self.opt.scales)#scales = [0,1,2,3]'scales used in the loss'
        self.num_input_frames = len(self.opt.frame_ids)#frames = [0,-1,1]'frame to load'
        self.num_pose_frames = 2 if self.opt.pose_model_input == "pairs" else self.num_input_frames
        #defualt is pose_model_input = 'pairs'

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])
        #able if not using use_stereo or frame_ids !=0
        #use_stereo defualt disable
        #frame_ids defualt =[0,-1,1]

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")
        
        self.models["encoder"] = networks.test_hr_encoder.hrnet18(True)
        self.models["encoder"].num_ch_enc = [ 64, 18, 36, 72, 144 ]
        
        para_sum = sum(p.numel() for p in self.models['encoder'].parameters())
        print('params in encoder',para_sum)
        
        self.models["depth"] = networks.HRDepthDecoder(
            self.models["encoder"].num_ch_enc, self.opt.scales)
        
        # self.models["bg2r"] = networks.BG2RCoeffsNetwork(64, 3)
        # self.models["bg2r"].to(self.device)
        # self.parameters_to_train += list(self.models["bg2r"].parameters())

        # 2nd NN
        if self.opt.use_recons_net:
            self.models["recon"] = networks.WaterTypeRegression(3)
            self.models["recon"].to(self.device)
            self.parameters_to_train += list(self.models["recon"].parameters())
            self.GWLoss = nn.L1Loss()
        
        # self.gwOptimizer = optim.SGD(self.models["recon"].parameters(), lr=1e-4)

        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())
        para_sum = sum(p.numel() for p in self.models['depth'].parameters())
        print('params in depth decdoer',para_sum)

        if self.use_pose_net:#use_pose_net = True
            if self.opt.pose_model_type == "separate_resnet":#defualt=separate_resnet  choice = ['normal or shared']
                
                self.models["pose_encoder"] = networks.ResnetEncoder(
                    self.opt.num_layers,
                    self.opt.weights_init == "pretrained",
                    num_input_images=self.num_pose_frames)#num_input_images=2
                
                self.models["pose"] = networks.PoseDecoder(
                    self.models["pose_encoder"].num_ch_enc,
                    num_input_features=1,
                    num_frames_to_predict_for=2)

            self.models["pose_encoder"].to(self.device)
            self.models["pose"].to(self.device)

            self.parameters_to_train += list(self.models["pose"].parameters())
            self.parameters_to_train += list(self.models["pose_encoder"].parameters())
        if self.opt.predictive_mask:
            #defualt = store_true 'uses a predictive mask like Zhou's'
            assert self.opt.disable_automasking, \
                "When using predictive_mask, please disable automasking with --disable_automasking"

            self.models["predictive_mask"] = networks.DepthDecoder(
                self.models["encoder"].num_ch_enc, self.opt.scales,
                num_output_channels=(len(self.opt.frame_ids) - 1))
            self.models["predictive_mask"].to(self.device)
            self.parameters_to_train += list(self.models["predictive_mask"].parameters())

        
        #self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)#learning_rate=1e-4
        self.model_optimizer = optim.Adam(self.parameters_to_train, 0.5 * self.opt.learning_rate)#learning_rate=1e-4
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)#defualt = 15'step size of the scheduler'

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.log_path)
        print("Training is using:\n  ", self.device)

        # data
        if self.opt.dataset=='kitti':
            datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                            "kitti_odom": datasets.KITTIOdomDataset}
        elif self.opt.dataset=='aqualoc': # aqualoc
            datasets_dict = {"aqualoc": datasets.AQUALOCDataset, # TODO continue here
                            "aqualoc_imu": datasets.AQUALOCDataset}
        elif self.opt.dataset=='sc': # aqualoc
            datasets_dict = {"sc": datasets.SCDataset, # TODO continue here
                            "sc_imu": datasets.SCDataset}               
        elif self.opt.dataset=='uc': # aqualoc
            datasets_dict = {"uc": datasets.UCanyonDataset, # TODO continue here
                            "uc_imu": datasets.UCanyonDataset}               
        elif self.opt.dataset=='flatiron': # aqualoc
            datasets_dict = {"flatiron": datasets.UCanyonDataset, # TODO continue here
                            "uc_imu": datasets.UCanyonDataset}               


        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        #change trainset
        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'
        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs
        
        train_dataset = self.dataset(self.opt.dataset,
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext, opts = self.opt)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(self.opt.dataset,
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext, opts = self.opt)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        if not self.opt.no_ssim:
            self.ssim = SSIM()
            self.ssim.to(self.device)
        self.num_batch = train_dataset.__len__() // self.opt.batch_size

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.opt.scales:
            h = self.opt.height // (2 ** scale)#defualt=[0,1,2,3]'scales used in the loss'
            w = self.opt.width // (2 ** scale)

            self.backproject_depth[scale] = BackprojectDepth(self.opt.batch_size, h, w)#in layers.py
            self.backproject_depth[scale].to(self.device)

            self.project_3d[scale] = Project3D(self.opt.batch_size, h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for k,m in self.models.items():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.init_time = time.time()
        if isinstance(self.opt.load_weights_folder,str):
            try:
                self.epoch_start = int(self.opt.load_weights_folder[-1]) + 1
            except:
                self.epoch_start=1 # weights_last
        else:
            self.epoch_start = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs - self.epoch_start):
            self.epoch = self.epoch_start + self.epoch
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:#number of epochs between each save defualt =1
                self.save_model()
            self.save_model(isLast=True)
        self.total_training_time = time.time() - self.init_time
        print('====>total training time:{}'.format(sec_to_hm_str(self.total_training_time)))

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        print("Threads: " + str(torch.get_num_threads()))
        print("Training")
        self.set_train()
        self.every_epoch_start_time = time.time()
        
        for batch_idx, inputs in enumerate(self.train_loader):
            before_op_time = time.time()
            outputs, losses = self.process_batch(inputs)
            
            # add 2nd NN
            # inputs[('color', 0, 0)].requires_grad=True
            # recon = self.models["recon"](inputs[('color', 0, 0)],outputs[("depth", 0, 0)] )
            # gwloss = self.computeGWLoss(recon)
            # losses["loss"]+=(1e-7)*gwloss

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

      
            # gwloss = self.computeGWLoss(recon)
            # self.gwOptimizer.zero_grad()
            # # torch.autograd.set_detect_anomaly(True)
            # gwloss.backward()
            # self.gwOptimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000#log_fre 's defualt = 250
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                # if "depth_gt" in inputs:
                #     self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()
            self.step += 1
        
        self.model_lr_scheduler.step()
        self.every_epoch_end_time = time.time()
        print("====>training time of this epoch:{}".format(sec_to_hm_str(self.every_epoch_end_time-self.every_epoch_start_time)))
   
    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():#inputs.values() has :12x3x196x640.
            inputs[key] = ipt.to(self.device)#put tensor in gpu memory

        if self.opt.pose_model_type == "shared":
            # If we are using a shared encoder for both depth and pose (as advocated
            # in monodepthv1), then all images are fed separately through the depth encoder.
            all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
            all_features = self.models["encoder"](all_color_aug)#stacked frames processing color together
            all_features = [torch.split(f, self.opt.batch_size) for f in all_features]#? what does inputs mean?

            features = {}
            for i, k in enumerate(self.opt.frame_ids):
                features[k] = [f[i] for f in all_features]

            outputs = self.models["depth"](features[0])
        else:
            # Otherwise, we only feed the image with frame_id 0 through the depth encoder
            features = self.models["encoder"](inputs["color_aug", 0, 0])

            outputs = self.models["depth"](features)

            # bg2rInFeatures = features[1][0]
            # BG_R = self.models["bg2r"](bg2rInFeatures, inputs["color_aug", 0, 0])
            # outputs["BG_R"] = BG_R
        if self.opt.predictive_mask:
            outputs["predictive_mask"] = self.models["predictive_mask"](features)
            #different form 1:*:* depth maps ,it will output 2:*:* mask maps

        if self.use_pose_net:
            outputs.update(self.predict_poses(inputs, features))

        self.generate_images_pred(inputs, outputs)

        if self.opt.use_recons_net:
            inputs[('color', 0, 0)].requires_grad=True
            outputs['recon'], coeffs = self.models["recon"](inputs[('color', 0, 0)],outputs[("depth", 0, 0)] )

        losses = self.compute_losses(inputs, outputs)


        return outputs, losses

    def predict_poses(self, inputs, features):
        """Predict poses between input frames for monocular sequences.
        """
        outputs = {}
        if self.num_pose_frames == 2:
            # In this setting, we compute the pose to each source frame via a
            # separate forward pass through the pose network.

            # select what features the pose network takes as input
            if self.opt.pose_model_type == "shared":
                pose_feats = {f_i: features[f_i] for f_i in self.opt.frame_ids}
            else:
                pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in self.opt.frame_ids}
            #pose_feats is a dict:
            #key:
            """"keys
                0
                -1
                1
            """
            for f_i in self.opt.frame_ids[1:]:
                #frame_ids = [0,-1,1]
                if f_i != "s":
                    # To maintain ordering we always pass frames in temporal order
                    if f_i < 0:
                        pose_inputs = [pose_feats[f_i], pose_feats[0]]#nerboring frames
                    else:
                        pose_inputs = [pose_feats[0], pose_feats[f_i]]

                    if self.opt.pose_model_type == "separate_resnet":
                        pose_inputs = [self.models["pose_encoder"](torch.cat(pose_inputs, 1))]
                    elif self.opt.pose_model_type == "posecnn":
                        pose_inputs = torch.cat(pose_inputs, 1)

                    axisangle, translation = self.models["pose"](pose_inputs)
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    #axisangle and translation are two 2*1*3 matrix
                    #f_i=-1,1
                    # Invert the matrix if the frame id is negative
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

        else:
            # Here we input all frames to the pose net (and predict all poses) together
            if self.opt.pose_model_type in ["separate_resnet", "posecnn"]:
                pose_inputs = torch.cat(
                    [inputs[("color_aug", i, 0)] for i in self.opt.frame_ids if i != "s"], 1)

                if self.opt.pose_model_type == "separate_resnet":
                    pose_inputs = [self.models["pose_encoder"](pose_inputs)]

            elif self.opt.pose_model_type == "shared":
                pose_inputs = [features[i] for i in self.opt.frame_ids if i != "s"]

            axisangle, translation = self.models["pose"](pose_inputs)

            for i, f_i in enumerate(self.opt.frame_ids[1:]):
                if f_i != "s":
                    outputs[("axisangle", 0, f_i)] = axisangle
                    outputs[("translation", 0, f_i)] = translation
                    outputs[("cam_T_cam", 0, f_i)] = transformation_from_parameters(
                        axisangle[:, i], translation[:, i])

        return outputs

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)
            ##
            batch_idx=0
            plt.set_cmap('jet')
            saveDir = os.path.join(self.log_path, 'imgs')
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)
            inputColor = toNumpy(inputs['color', 0, 0]*255, keepDim=True).astype(np.uint8)
            
            outPred = toNumpy(normalize_image(outputs['disp', 0])*255, keepDim=True).astype(np.uint8)
            # if self.opt.batch_size==1:
            #     outPred = np.expand_dims(outPred,0)
            #     inputColor = np.expand_dims(inputColor,0)
            for j in range(self.opt.batch_size):
                currPred = np.squeeze(outPred[j,:])
                currColor = np.squeeze(inputColor[j,:])
        
                # if self.opt.load_depth:
                #     gt_depth = (inputs['depth_gt'])
                #     currGTDepth = toNumpy((normalize_image(gt_depth[j,:])*255)).astype(np.uint8)
                #     plt.imsave(saveDir + "/frame_{:06d}_gtDepth.bmp".format(batch_idx+j), currGTDepth)

                plt.imsave(saveDir + "/frame_{:06d}_color.bmp".format(batch_idx+j), currColor)
                plt.imsave(saveDir + "/frame_{:06d}_pred.bmp".format(batch_idx+j), currPred)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def generate_images_pred(self, inputs, outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        for scale in self.opt.scales:
            disp = outputs[("disp", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                disp = F.interpolate(
                    disp, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            _, depth = disp_to_depth(disp, self.opt.min_depth, self.opt.max_depth)#disp_to_depth function is in layers.py
            
            ## debug
            # # TODO: remove taht after!!
            # gt = inputs[('depth_gt')]
            # gt = F.interpolate(gt, size=(480, 640))
            # depth[gt>0] = gt[gt>0]

            outputs[("depth", 0, scale)] = depth
            # outputs[("depth", 0, scale)] = 0.1*inputs[("depth_gt")].clone()
            for i, frame_id in enumerate(self.opt.frame_ids[1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"]
                else:
                    T = outputs[("cam_T_cam", 0, frame_id)]

                # from the authors of https://arxiv.org/abs/1712.00175
                if self.opt.pose_model_type == "posecnn":

                    axisangle = outputs[("axisangle", 0, frame_id)]
                    translation = outputs[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(
                        axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](
                    depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](
                    cam_points, inputs[("K", source_scale)], T)
                outputs[("sample", frame_id, scale)] = pix_coords

                outputs[("color", frame_id, scale)] = F.grid_sample(
                    inputs[("color", frame_id, source_scale)],
                    outputs[("sample", frame_id, scale)],
                    padding_mode="border")

                if not self.opt.disable_automasking:
                    #doing this
                    outputs[("color_identity", frame_id, scale)] = \
                        inputs[("color", frame_id, source_scale)]

    def compute_reprojection_loss(self, pred, target, mask=None):
        """Computes reprojection loss between a batch of predicted and target images
        """
        # mask3 = mask.expand(mask.shape[0], 3, mask.shape[2], mask.shape[3])
        abs_diff = torch.abs(target - pred)
        l1_loss = abs_diff.mean(1, True)
        alpha = float(self.opt.alpha)
        if self.opt.no_ssim:
            reprojection_loss = l1_loss
        else:
            ssim_loss = self.ssim(pred, target).mean(1, True)
            reprojection_loss = (1-alpha) * ssim_loss + alpha * l1_loss

        # mask = disp
        # if mask is not None:
        #     weights_mask = normalize_image(mask)
        #     reprojection_loss*=weights_mask
        total_rloss = reprojection_loss
        if self.opt.use_lvw:
            self.lv = LocalVariation(k_size=25)
            lv = normalize_image(self.lv(pred, target).mean(1, True))
            if 0: # self.opt.load_weights_folder is not None or self.epoch > 5:
                dweight = normalize_image(mask)
                total_rloss = reprojection_loss*lv*(1-dweight) + reprojection_loss*dweight
                
            else:
                total_rloss = reprojection_loss*lv
                
        # if mask is not None:
        #     reprojection_loss[mask<1e-3]=0
        #     reprojection_loss[mask>15]=0
        if 0:
            rnd = np.random.randint(100)
            saveTensor(target,'local_variation_samples/'+str(rnd)+'_'+'rgb.png')
            saveTensor(pred,'local_variation_samples/'+str(rnd)+'_'+'rgb_pred.png')
            saveTensor(l1_loss, 'local_variation_samples/'+str(rnd)+'_'+'l1.png')
            saveTensor(ssim_loss, 'local_variation_samples/'+str(rnd)+'_'+'ssim_loss.png')
            saveTensor(reprojection_loss, 'local_variation_samples/'+str(rnd)+'_'+'reprojection_loss.png')
            saveTensor(lv, 'local_variation_samples/'+str(rnd)+'_'+'lv.png')
            saveTensor(total_rloss, 'local_variation_samples/'+str(rnd)+'_'+'total_rloss.png')

        return total_rloss

    def computeGWLoss(self, img):
        globalMu = torch.mean(img.view(img.shape[0], -1), dim=1)
        channelMu = torch.mean(img.view(img.shape[0], img.shape[1],-1), dim=2)
        globalMu = globalMu.unsqueeze(1).repeat(1,3)
        gwloss = self.GWLoss(globalMu, channelMu)
        # TODO: try minimize only thr max error
        return gwloss



    def compute_losses(self, inputs, outputs):
        """Compute the reprojection and smoothness losses for a minibatch
        """
        losses = {}
        total_loss = 0

        for scale in self.opt.scales:
            #scales=[0,1,2,3]
            loss = 0
            reprojection_losses = []

            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                source_scale = 0

            disp = outputs[("disp", scale)]
            color = inputs[("color", 0, scale)]
            target = inputs[("color", 0, source_scale)]
            disp_src = outputs[("disp", source_scale)]
            ## add resize for depth 
            depth=None
            if "depth_gt" in inputs:
                s = 2 ** 0
                depth = torch.nn.functional.interpolate(inputs[("depth_gt")], (self.opt.height // s, self.opt.width // s),
                                                mode="nearest")
            # 
                    
            for frame_id in self.opt.frame_ids[1:]:
                pred = outputs[("color", frame_id, scale)]
                reprojection_losses.append(self.compute_reprojection_loss(pred, target, disp_src))
            reprojection_losses = torch.cat(reprojection_losses, 1)
            if not self.opt.disable_automasking:
                #doing this 
                identity_reprojection_losses = []
                for frame_id in self.opt.frame_ids[1:]:
                    pred = inputs[("color", frame_id, source_scale)]
                    
                    identity_reprojection_losses.append(
                        self.compute_reprojection_loss(pred, target, disp_src))

                identity_reprojection_losses = torch.cat(identity_reprojection_losses, 1)
                if self.opt.avg_reprojection:
                    identity_reprojection_loss = identity_reprojection_losses.mean(1, keepdim=True)
                else:
                    # save both images, and do min all at once below
                    identity_reprojection_loss = identity_reprojection_losses

            elif self.opt.predictive_mask:
                mask = outputs["predictive_mask"]["disp", scale]
                if not self.opt.v1_multiscale:
                    mask = F.interpolate(
                        mask, [self.opt.height, self.opt.width],
                        mode="bilinear", align_corners=False)

                reprojection_losses *= mask
                #reprojection_losses.size() =12X2X192X640 

                # add a loss pushing mask to 1 (using nn.BCELoss for stability)
                weighting_loss = 0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cuda()) if torch.cuda.is_available() else   0.2 * nn.BCELoss()(mask, torch.ones(mask.shape).cpu())
                loss += weighting_loss.mean()

            if self.opt.avg_reprojection:
                reprojection_loss = reprojection_losses.mean(1, keepdim=True)
            else:
                #doing_this
                reprojection_loss = reprojection_losses

            if not self.opt.disable_automasking:
                #doing_this
                # add random numbers to break ties
                    #identity_reprojection_loss.shape).cuda() * 0.00001
                if torch.cuda.is_available():
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).to(self.device) * 0.00001
                else:
                    identity_reprojection_loss += torch.randn(identity_reprojection_loss.shape).cpu() * 0.00001
                combined = torch.cat((identity_reprojection_loss, reprojection_loss), dim=1)
            else:
                combined = reprojection_loss

            if combined.shape[1] == 1:
                to_optimise = combined
            else:
                #doing this
                to_optimise, idxs = torch.min(combined, dim=1)
            if not self.opt.disable_automasking:
                outputs["identity_selection/{}".format(scale)] = (
                # outputs["identity_selection/{}".format(0)] = (
                    idxs > identity_reprojection_loss.shape[1] - 1).float()

            loss += to_optimise.mean()
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, color)

            loss += self.opt.disparity_smoothness * smooth_loss / (2 ** scale)#defualt=1e-3 something with get_smooth_loss function
            total_loss += loss
            losses["loss/{}".format(scale)] = loss
        
        total_loss /= self.num_scales

        # correlation loss
        if self.opt.use_corrLoss:
            corrLoss = corr_loss(inputs[("color", 0, 0)], inputs[("color", 0, 0)], outputs[('depth', 0, 0)])
            total_loss += (1e-5*corrLoss)
            losses["loss/corrLoss"] = corrLoss

        # BG_R loss
        # bgrLoss = compute_bg_r_loss(outputs[('BG_R')], outputs[('depth', 0, 0)])
        # total_loss += (1e-7*bgrLoss)

        if self.opt.use_recons_net:
        # GW loss
            w=0
            if self.epoch>5:
                w=1e-4
            if self.epoch>25:
                w=1
            gwloss = self.computeGWLoss(outputs['recon'])
            total_loss+=(w)*gwloss
            losses["loss/gwloss"] = gwloss

        ## debug A
        if 0:
            img = toNumpy(inputs[("color", 0, 0)])
            depth = toNumpy(outputs[('depth', 0, 0)])
            A = estimateA(img, depth)
            TM = np.zeros_like(img)
            for t in range(3):
                # TM[:,:,t] =  np.exp(-beta_rgb[t]*depth)
                TM[:,:,t] =  water_types_Nrer_rgb["3C"][t]**depth
            S = A*(1-TM)
            J = (img - A) / TM + A
    
        if 0:
            dataname='sc'
            frameNum = inputs["frameNum"].numpy()[0]
            saveTensor(inputs[('color', -1, 0)], dataname + '_lossDebug/' + dataname + str(frameNum)+'_minusonecolor.png')
            saveTensor(inputs[('color', 1, 0)], dataname+'_lossDebug/'+ dataname +str(frameNum)+'_plusonecolor.png')
            saveTensor(inputs[('color', 0, 0)], dataname + '_lossDebug/'+ dataname +str(frameNum)+'_color.png')
            saveTensor(outputs[('color', 1, 0)], dataname+ '_lossDebug/'+ dataname +str(frameNum)+'_plusonecolorpred.png')
            saveTensor(outputs[('color', -1, 0)], dataname+'_lossDebug/'+ dataname +str(frameNum)+'_minusonecolorpred.png')
            saveTensor((inputs[('color', 0, 0)] - outputs[('color', -1, 0)]), dataname+'_lossDebug/'+ dataname +str(frameNum)+'_diffminus.png')
            saveTensor((inputs[('color', 0, 0)] - outputs[('color', 1, 0)]), dataname+'_lossDebug/'+ dataname +str(frameNum)+'_diffplus.png')
            saveTensor((outputs[('depth', 0, 0)]), dataname+'_lossDebug/'+ dataname +str(frameNum)+'_depth.png')
        losses["loss"] = total_loss 
        return losses

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so i#s only used to give an indication of validation performance


        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, full_res_shape, mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        # crop_mask[:, :, 153:371, 44:1197] = 1
        # mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch_idx {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses, batch_idx=0):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            s=0
            # for s in self.opt.scales:
            for frame_id in self.opt.frame_ids:
                writer.add_image(
                    "color_{}_{}/{}".format(frame_id, s, j),
                    inputs[("color", frame_id, s)][j].data, self.step)
                if s == 0 and frame_id != 0:
                    writer.add_image(
                        "color_pred_{}_{}/{}".format(frame_id, s, j),
                        outputs[("color", frame_id, s)][j].data, self.step)

            writer.add_image(
                "disp_{}/{}".format(s, j),
                normalize_image(outputs[("disp", s)][j]), self.step)

            # if self.opt.predictive_mask:
            #     for f_idx, frame_id in enumerate(self.opt.frame_ids[1:]):
            #         writer.add_image(
            #             "predictive_mask_{}_{}/{}".format(frame_id, s, j),
            #             outputs["predictive_mask"][("disp", s)][j, f_idx][None, ...],
            #             self.step)

            # elif not self.opt.disable_automasking:
            #     writer.add_image(
            #         "automask_{}/{}".format(s, j),
            #         outputs["identity_selection/{}".format(s)][j][None, ...], self.step)

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self, isLast=False):
        """Save model weights to disk
        """
        if isLast:
            name='last'
        else:
            name=self.epoch
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(name))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path, map_location=torch.device(self.device))
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path, map_location=torch.device(self.device))
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
