from collections import defaultdict
from torch import nn
import torch

def build_meta_memory(cfg, cls_mapping, keepclass, memory_dim):
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    return Memory(cfg, cls_mapping, keepclass, memory_dim)

class Memory(nn.Module):
    def __init__(self, cfg, cls_mapping, keepclass, memory_dim):
        super().__init__()
        self.register_buffer("proposal_feature_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE, memory_dim), persistent=False)
        self.register_buffer("proposal_delta_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE, 4), persistent=False)
        self.register_buffer("proposal_scale_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE), persistent=False)

        fc_dim  = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        self.register_buffer("roi_feature_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE, fc_dim), persistent=False)
        self.register_buffer("roi_delta_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE, 4), persistent=False)
        self.register_buffer("roi_scale_memory", torch.zeros(len(keepclass), cfg.MEMORY.NUM_INSTANCE), persistent=False)

        self.roi_ptr = defaultdict(int)
        self.prop_ptr = defaultdict(int)

        self.roi_max = defaultdict(int)
        self.prop_max = defaultdict(int)

        cls_idx = [cls_mapping[x] for x in keepclass]
        self.mapping = {k-1:v for k, v in zip(cls_idx, range(len(cls_idx)))}
        self.reverse_mapping = {v:k for k,v in self.mapping.items()}

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.to(self.device)
    
    def forward(self, feature_dict=None, gt_class=None, sample_rate=0.1):
        assert ~((feature_dict is not None) and (gt_class is not None))
        if feature_dict is None and gt_class is None:
            # initialize memory 
            self.roi_ptr = defaultdict(int)
            self.prop_ptr = defaultdict(int)
        elif feature_dict is not None:
            def insert_queue(queue, q_id, value):
                if q_id + len(value) > len(queue):
                    insert_queue(queue, q_id, value[:len(queue) - q_id])
                    value = value[len(queue) - q_id:]
                    q_id = 0
                queue[q_id:q_id + len(value)] = value
                return int(q_id + len(value))

            # TODO_P: update memory with feature dict
            class_id = feature_dict["proposal"]["class"].unique()
            per_class_idx = class_id.view(-1, 1) == feature_dict["proposal"]["class"]
            for i, cid in enumerate(class_id):
                cid = self.mapping[int(cid)]
                cid_idx = per_class_idx[i]
                len_idx = cid_idx.sum()
                per_class_feature = feature_dict["proposal"]["feature"][cid_idx]
                per_class_deltas = feature_dict["proposal"]["deltas"][cid_idx]
                per_class_scale = feature_dict["proposal"]["scale"][cid_idx]

                m_idx = self.prop_ptr[cid]
                insert_queue(self.proposal_feature_memory[cid], m_idx, per_class_feature)
                insert_queue(self.proposal_delta_memory[cid], m_idx, per_class_deltas)
                new_m_idx = insert_queue(self.proposal_scale_memory[cid], m_idx, per_class_scale)
                self.prop_ptr[cid] = new_m_idx
                if self.prop_max[cid] < new_m_idx:
                    self.prop_max[cid] = new_m_idx
                elif self.prop_max[cid] > new_m_idx:
                    self.prop_max[cid] = len(self.proposal_feature_memory[cid])

            class_id = feature_dict["roi"]["class"].unique()
            per_class_idx = class_id.view(-1, 1) == feature_dict["roi"]["class"]
            for i, cid in enumerate(class_id):
                cid = self.mapping[int(cid)]
                cid_idx = per_class_idx[i]
                len_idx = cid_idx.sum()
                per_class_feature = feature_dict["roi"]["feature"][cid_idx]
                per_class_deltas = feature_dict["roi"]["deltas"][cid_idx]
                per_class_scale = feature_dict["roi"]["scale"][cid_idx]

                m_idx = self.roi_ptr[cid]

                insert_queue(self.roi_feature_memory[cid], m_idx, per_class_feature)
                insert_queue(self.roi_delta_memory[cid], m_idx, per_class_deltas)
                new_m_idx = insert_queue(self.roi_scale_memory[cid], m_idx, per_class_scale)
                self.roi_ptr[cid] = new_m_idx

                self.roi_ptr[cid] = new_m_idx
                if self.roi_max[cid] < new_m_idx:
                    self.roi_max[cid] = new_m_idx
                elif self.roi_max[cid] > new_m_idx:
                    self.roi_max[cid] = len(self.roi_feature_memory[cid])

        elif gt_class is not None:
            # TODO_P: sample feature from memory
            # gt_class = torch.cat([x["instances"].gt_classes for x in class_dict]).long().unique(sorted=True)
            prop_feature = []
            prop_deltas = []
            prop_scale = []
            prop_class = []

            roi_feature = []
            roi_deltas = []
            roi_scale = []
            roi_class = []

            for cid in gt_class:
                cid_map = self.mapping[int(cid)]
                prop_len = self.prop_max[cid_map]
                roi_len = self.roi_max[cid_map]

                prop_idx = torch.randperm(prop_len)[:max(int(prop_len*sample_rate), 10)]
                roi_idx = torch.randperm(roi_len)[:max(int(roi_len*sample_rate), 10)]

                prop_feature.append(self.proposal_feature_memory[cid_map][prop_idx])
                prop_deltas.append(self.proposal_delta_memory[cid_map][prop_idx])
                prop_scale.append(self.proposal_scale_memory[cid_map][prop_idx])
                prop_class.append(torch.ones_like(self.proposal_scale_memory[cid_map][prop_idx]) * (cid))

                roi_feature.append(self.roi_feature_memory[cid_map][roi_idx])
                roi_deltas.append(self.roi_delta_memory[cid_map][roi_idx])
                roi_scale.append(self.roi_scale_memory[cid_map][roi_idx])
                roi_class.append(torch.ones_like(self.roi_scale_memory[cid_map][roi_idx]) * (cid))

            prop_feature = torch.cat(prop_feature)
            prop_deltas = torch.cat(prop_deltas)
            prop_scale = torch.cat(prop_scale)
            prop_class = torch.cat(prop_class)
            roi_feature = torch.cat(roi_feature)
            roi_deltas = torch.cat(roi_deltas)
            roi_scale = torch.cat(roi_scale)
            roi_class = torch.cat(roi_class)

            feature_dict = {
                "proposal":{
                    "feature": prop_feature,
                    "deltas": prop_deltas,
                    "scale": prop_scale,
                    "class": prop_class
                },
                "roi":{
                    "feature": roi_feature,
                    "deltas": roi_deltas,
                    "scale": roi_scale,
                    "class": roi_class
                }
            }

            return feature_dict
