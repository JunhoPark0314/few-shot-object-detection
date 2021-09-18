from collections import defaultdict
from torch import nn
import torch

def build_meta_memory(cfg, cls_id_list, memory_dim):
    """
    Built the whole model, defined by `cfg.MODEL.META_ARCHITECTURE`.
    """
    return Memory(cfg, cls_id_list, memory_dim)

class Memory(nn.Module):
    def __init__(self, cfg, cls_id_list, memory_dim):
        super().__init__()
        self.register_buffer("proposal_feature_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE, memory_dim), persistent=False)
        self.register_buffer("proposal_delta_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE, 4), persistent=False)
        self.register_buffer("proposal_scale_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE), persistent=False)

        roi_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.register_buffer("roi_feature_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE, memory_dim, roi_size, roi_size), persistent=False)
        self.register_buffer("roi_delta_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE, 4), persistent=False)
        self.register_buffer("roi_scale_memory", torch.zeros(len(cls_id_list), cfg.MEMORY.NUM_INSTANCE), persistent=False)

        self.roi_ptr = defaultdict(int)
        self.prop_ptr = defaultdict(int)
        self.id_map = {k:i for i, k in enumerate(cls_id_list)}
    
    def forward(self, feature_dict=None, class_dict=None):
        assert ~((feature_dict is not None) and (class_dict is not None))
        if feature_dict is None and class_dict is None:
            # initialize memory 
            self.roi_ptr = defaultdict(int)
            self.prop_ptr = defaultdict(int)
        elif feature_dict is not None:
            def insert_queue(queue, q_id, value):
                if q_id + len(value) > len(queue):
                    insert_queue(queue, q_id, value[:len(queue) - q_id])
                    q_id = 0
                    value = value[len(queue) - q_id:]
                queue[q_id:q_id + len(value)] = value
                return q_id + len(value)

            # TODO_P: update memory with feature dict
            class_id = feature_dict["proposal"]["class"].unique()
            per_class_idx = class_id.view(-1, 1) == feature_dict["proposal"]["class"]
            for i, cid in enumerate(class_id):
                cid = cid - 1
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

            class_id = feature_dict["roi"]["class"].unique()
            per_class_idx = class_id.view(-1, 1) == feature_dict["roi"]["class"]
            for i, cid in enumerate(class_id):
                cid = cid - 1
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

            return 1

        elif class_dict is not None:
            # TODO_P: sample feature from memory
            pass
