from torch import nn
import torch

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
		rpn_out_channels = rpn_in_channels if conv_dims[0] == -1 else conv_dims[0]

		self.rpn_feature = nn.Linear(rpn_in_channels, rpn_out_channels * 2)
		self.rpn_deltas = nn.Linear(4, rpn_out_channels * 2)
		self.rpn_scale = nn.Linear(1, rpn_out_channels * 2)

		self.roi_feature = nn.Linear(roi_shape, roi_shape*2)
		self.roi_deltas = nn.Linear(4, roi_shape*2)
		self.roi_scale = nn.Linear(1, roi_shape*2)

		self.rpn_shape = rpn_out_channels
		self.roi_shape = roi_shape
	
	def forward(self, support_feature, device):
		support_gt_class = None

		if support_feature is not None:
			rpn_feature = self.rpn_feature(support_feature["proposal"]["feature"])
			rpn_deltas = self.rpn_deltas(support_feature["proposal"]["deltas"])
			rpn_scale = self.rpn_scale(support_feature["proposal"]["scale"].view(-1, 1))
			rpn_weight = (rpn_feature + rpn_deltas + rpn_scale).mean(dim=0)

			support_gt_class = support_feature["roi"]["class"].long().unique(sorted=True)
			roi_feature = self.roi_feature(support_feature["roi"]["feature"])
			roi_deltas = self.roi_deltas(support_feature["roi"]["deltas"])
			roi_scale = self.roi_scale(support_feature["roi"]["scale"].view(-1, 1))

			per_class_weight = []
			for cid in support_gt_class:
				idx = support_feature["roi"]["class"] == cid
				per_class_weight.append((roi_feature[idx] + roi_deltas[idx] + roi_scale[idx]).mean(dim=0)[None,:self.roi_shape])

			roi_cls_weight = torch.cat(per_class_weight)
			roi_bbox_weight = (roi_feature + roi_deltas + roi_scale).mean(dim=0)[None,self.roi_shape:]
		else:
			rpn_weight = torch.zeros(self.rpn_shape * 2, device=device)
			roi_cls_weight = torch.zeros((1, self.roi_shape), device=device)
			roi_bbox_weight = torch.zeros((1, self.roi_shape), device=device)
		
		latent_vector = {
			"proposal": torch.chunk(rpn_weight, 2, dim=-1),
			"roi": [roi_cls_weight, roi_bbox_weight]
		}

		return latent_vector, support_gt_class