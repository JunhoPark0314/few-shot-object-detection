from detectron2.utils.events import get_event_storage
from torch import nn
import torch
from fsdet.layers.dynamic_convolutions import AttentionLayerWoGP
from torch.cuda import get_arch_list

def build_latent_encoder(cfg, rpn_shape):
	rpn_features = cfg.MODEL.RPN.IN_FEATURES
	rpn_shape = [rpn_shape[f] for f in rpn_features]
	roi_shape = cfg.MODEL.ROI_BOX_HEAD.FC_DIM

	return LatentEncoder(cfg, rpn_shape, roi_shape)

class LatentEncoder(nn.Module):
	def __init__(self, cfg, rpn_shape, roi_shape):
		super().__init__()
		
		rpn_in_channels = [s.channels for s in rpn_shape]
		assert len(set(rpn_in_channels)) == 1, "Each level must have the same channel!"
		rpn_in_channels = rpn_in_channels[0]
		conv_dims = cfg.MODEL.RPN.CONV_DIMS
		reduce = cfg.MODEL.ENCODER.REDUCE
		rpn_out_channels = rpn_in_channels if conv_dims[0] == -1 else conv_dims[0]

		self.use_deltas = cfg.MODEL.ENCODER.USE_DELTAS
		self.use_scale = cfg.MODEL.ENCODER.USE_DELTAS
		self.alpha_dim = cfg.MODEL.ENCODER.ALPHA_DIM
		self.nok = cfg.MODEL.ENCODER.NOK

		self.rpn_feature = nn.Linear(rpn_in_channels, rpn_out_channels * 2)
		self.rpn_deltas = nn.Linear(4, rpn_out_channels * 2)
		self.rpn_scale = nn.Linear(1, rpn_out_channels * 2)

		self.roi_feature = nn.Linear(roi_shape, roi_shape*2)
		self.roi_deltas = nn.Linear(4, roi_shape*2)
		self.roi_scale = nn.Linear(1, roi_shape*2)

		self.rpn_shape = rpn_out_channels
		self.roi_shape = roi_shape
		self.max_iter = cfg.SOLVER.MAX_ITER
		self.temperature = cfg.MODEL.ENCODER.TEMPERATURE

		self.rpn_cls_alphas = AttentionLayerWoGP(c_dim=rpn_out_channels, hidden_dim=max(1, rpn_out_channels // reduce), nof_kernels=self.nok)
		self.rpn_bbox_alphas = AttentionLayerWoGP(c_dim=rpn_out_channels, hidden_dim=max(1, rpn_out_channels // reduce), nof_kernels=self.nok)
		self.roi_cls_alphas = AttentionLayerWoGP(c_dim=roi_shape, hidden_dim=max(1, roi_shape // reduce), nof_kernels=self.nok)
		self.roi_bbox_alphas = AttentionLayerWoGP(c_dim=roi_shape, hidden_dim=max(1, roi_shape // reduce), nof_kernels=self.nok)
		self.cont_loss = True
	
	
	def forward(self, support_feature, device):
		support_gt_class = None
		curr_iter = get_event_storage().iter
		progress = (self.max_iter - curr_iter) / self.max_iter
		temperature = max(1, self.temperature * progress)
		loss = {}

		if support_feature is not None:
			support_gt_class = support_feature["roi"]["class"].long().unique(sorted=True)

			rpn_gt_class = support_feature["proposal"]["class"].long()
			rpn_feature = self.rpn_feature(support_feature["proposal"]["feature"])
			rpn_deltas = self.rpn_deltas(support_feature["proposal"]["deltas"])
			rpn_scale = self.rpn_scale(support_feature["proposal"]["scale"].view(-1, 1))
			rpn_cls_weight, rpn_bbox_weight = torch.chunk(rpn_feature + rpn_deltas + rpn_scale, 2, dim=-1)
			rpn_cls_alphas = self.rpn_cls_alphas(rpn_cls_weight, temperature)
			rpn_bbox_alphas = self.rpn_bbox_alphas(rpn_bbox_weight, temperature)

			roi_gt_class = support_feature["roi"]["class"].long()
			roi_feature = self.roi_feature(support_feature["roi"]["feature"])
			roi_deltas = self.roi_deltas(support_feature["roi"]["deltas"])
			roi_scale = self.roi_scale(support_feature["roi"]["scale"].view(-1, 1))
			roi_cls_weight, roi_bbox_weight = torch.chunk(roi_feature + roi_deltas + roi_scale, 2, dim=-1)
			roi_cls_alphas = self.roi_cls_alphas(roi_cls_weight, temperature)
			roi_bbox_alphas = self.roi_cls_alphas(roi_bbox_weight, temperature)

			if self.cont_loss:
				def contrastive(x, temp):
					eps = 1e-7
					return (x / temp).exp() / ((x / temp).exp()+eps).sum(dim=1)
				
				def cont_loss(x, gt, temp):
					return -(contrastive(x, temp).log() * contrastive(gt, temp)).sum(dim=1).mean()
					
				roi_cls_graph_gt = (roi_gt_class.view(1, -1) == roi_gt_class.view(-1, 1)).float()
				rpn_cls_graph_gt = (rpn_gt_class.view(1, -1) == rpn_gt_class.view(-1, 1)).float()
				roi_bbox_graph_gt = torch.mm(support_feature["roi"]["deltas"], support_feature["roi"]["deltas"].T)
				rpn_bbox_graph_gt = torch.mm(support_feature["proposal"]["deltas"], support_feature["proposal"]["deltas"].T)

				roi_cls_graph = torch.mm(roi_cls_alphas, roi_cls_alphas.T)
				rpn_cls_graph = torch.mm(rpn_cls_alphas, rpn_cls_alphas.T)
				roi_bbox_graph = torch.mm(roi_bbox_alphas, roi_bbox_alphas.T)
				rpn_bbox_graph = torch.mm(rpn_bbox_alphas, rpn_bbox_alphas.T)

				roi_cls_loss = cont_loss(roi_cls_graph, roi_cls_graph_gt, 0.5)
				rpn_cls_loss = cont_loss(rpn_cls_graph, rpn_cls_graph_gt, 0.5)
				roi_bbox_loss = cont_loss(roi_bbox_graph, roi_bbox_graph_gt, 0.5)
				rpn_bbox_loss = cont_loss(rpn_bbox_graph, rpn_bbox_graph_gt, 0.5)

				loss = {
					"roi_cls_grpah_loss": roi_cls_loss,
					"rpn_cls_grpah_loss": rpn_cls_loss,
					"roi_bbox_grpah_loss": roi_bbox_loss,
					"rpn_bbox_grpah_loss": rpn_bbox_loss,
				}

			per_class_alphas = []
			for cid in support_gt_class:
				idx = support_feature["roi"]["class"] == cid
				per_class_alphas.append(roi_cls_alphas[idx].mean(dim=0).view(1, -1))

			rpn_cls_alphas = rpn_cls_alphas.mean(dim=0)
			rpn_bbox_alphas = rpn_bbox_alphas.mean(dim=0)
			roi_cls_alphas = torch.cat(per_class_alphas)
			roi_bbox_alphas = roi_bbox_alphas.mean(dim=0).view(1, -1)

			storage = get_event_storage()
			storage.put_scalar(
				"temperature", temperature
			)

		else:
			rpn_cls_alphas = torch.ones(self.nok, device=device) / self.nok
			rpn_bbox_alphas = torch.ones(self.nok, device=device) / self.nok
			roi_cls_alphas = torch.ones((1,self.nok), device=device) / self.nok
			roi_bbox_alphas = torch.ones((1,self.nok), device=device) / self.nok
		
		latent_vector = {
			"proposal": [rpn_cls_alphas, rpn_bbox_alphas],
			"roi": [roi_cls_alphas, roi_bbox_alphas]
		}
		return latent_vector, support_gt_class, loss