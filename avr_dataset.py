import json
import pandas as pd
import xml.etree.ElementTree as ET
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from utils import parse_panels, bbox_to_xyxy, parse_rules

# Stage 1


class AVRStage1Dataset(Dataset):
    def __init__(
            self,
            dataset_dir: str,
            split: str = 'train',
            transform = None,
            target_transform = None):
        assert split in ['train', 'val', 'test']

        self.split = split
        self.configurations = [
            'center_single',
            'distribute_four',
            'distribute_nine',
            'in_center_single_out_center_single',
            'in_distribute_four_out_center_single',
            'left_center_single_right_center_single',
            'up_center_single_down_center_single'
        ]
        self.dataset_path = Path(dataset_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.all_file_stems = list(fn.stem for fn in (self.dataset_path / Path(self.configurations[0])).glob(f'*_{self.split}.npz'))
        self.all_file_paths = [Path(self.dataset_path, config, base_fn) for config, base_fn in product(self.configurations, self.all_file_stems)]

        self.id2type = ['none', 'triangle', 'square', 'pentagon', 'hexagon', 'circle']
        self.id2size = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.id2color = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]

    def get_panel_by_id(self, idx: int):
        file_idx, panel_idx = int(idx / 16), int(idx % 16)
        return self.all_file_paths[file_idx], panel_idx

    def __len__(self):
        return len(self.all_file_paths) * 16

    def __getitem__(self, idx):
        file_path, panel_idx = self.get_panel_by_id(idx)

        npz = np.load(file_path.with_suffix('.npz'))
        image = torch.as_tensor(npz['image'][panel_idx, :, :], dtype=torch.float) / 255
        image = torch.unsqueeze(image, dim=0)

        targets = {}
        boxes = []
        types = []
        sizes = []
        colors = []
        xml = ET.parse(file_path.with_suffix('.xml'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        panel_info = panel_info_list[panel_idx]
        all_entities = []
        for component in panel_info['components']:
            all_entities += component['entities']
        for entity in all_entities:
            boxes.append(bbox_to_xyxy(json.loads(entity['real_bbox'])))
            types.append(int(entity['Type']))
            sizes.append(int(entity['Size']))
            colors.append(int(entity['Color']))

        targets['boxes'] = torch.as_tensor(boxes, dtype=torch.float)
        targets['types'] = torch.as_tensor(types, dtype=torch.int64)
        targets['labels'] = torch.as_tensor(types, dtype=torch.int64)
        targets['sizes'] = torch.as_tensor(sizes, dtype=torch.int64)
        targets['colors'] = torch.as_tensor(colors, dtype=torch.int64)
        targets['image_id'] = torch.tensor(idx)
        # targets['image'] = image

        return image, targets


def collate_fn(batch):
    return tuple(zip(*batch))


# Stage 2


def get_stage2_dataset(dataset_dir: str, split: str):
    configurations = [
        'distribute_nine',
        'center_single',
        'distribute_four',
        'distribute_nine',
        'in_center_single_out_center_single',
        'in_distribute_four_out_center_single',
        'left_center_single_right_center_single',
        'up_center_single_down_center_single'
    ]
    id2type = ['none', 'triangle', 'square', 'pentagon', 'hexagon', 'circle']
    id2size = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    id2color = [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
    id2slot = [(0.5, 0.5, 0.33, 0.33), (0.42, 0.42, 0.15, 0.15), (0.25, 0.75, 0.5, 0.5), (0.5, 0.75, 0.5, 0.5),
               (0.5, 0.5, 1, 1), (0.75, 0.5, 0.5, 0.5), (0.58, 0.58, 0.15, 0.15), (0.83, 0.83, 0.33, 0.33),
               (0.83, 0.16, 0.33, 0.33), (0.42, 0.58, 0.15, 0.15), (0.83, 0.5, 0.33, 0.33), (0.16, 0.5, 0.33, 0.33),
               (0.75, 0.25, 0.5, 0.5), (0.5, 0.83, 0.33, 0.33), (0.58, 0.42, 0.15, 0.15), (0.5, 0.25, 0.5, 0.5),
               (0.16, 0.83, 0.33, 0.33), (0.16, 0.16, 0.33, 0.33), (0.5, 0.16, 0.33, 0.33), (0.25, 0.5, 0.5, 0.5),
               (0.75, 0.75, 0.5, 0.5), (0.25, 0.25, 0.5, 0.5)]
    slot2id = {slot: idx for idx, slot in enumerate(id2slot)}

    dataset_path = Path(dataset_dir)
    all_file_stems = list(fn.stem for fn in (dataset_path / Path(configurations[0])).glob(f'*_{split}.npz'))
    all_file_paths = [Path(dataset_path, config, base_fn) for config, base_fn in
                      product(configurations, all_file_stems)]

    full_data_dict = []
    full_data_pd = []

    for file_path in all_file_paths:
        xml = ET.parse(file_path.with_suffix('.xml'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        rules = parse_rules(xml_root)

        context_panels = panel_info_list[:6]

        for panel_idx, panel in enumerate(context_panels):
            for component in panel['components']:
                component_idx = component['component']['id']
                comp_slots_data = {}
                for entity in component['entities']:
                    entity_size = eval(entity['Size'])
                    entity_type = eval(entity['Type'])
                    entity_color = eval(entity['Color'])
                    entity_slot_id = slot2id[tuple(eval(entity['bbox']))]
                    comp_slots_data.update(
                        {entity_slot_id: {'size': entity_size, 'type': entity_type, 'color': entity_color}})

                data = {'file': file_path, 'panel': panel_idx, 'component': component_idx}
                full_data_dict.append({**data, 'slots': comp_slots_data})
                data_pd = data.copy()

                for slot_idx in range(0, len(id2slot)):
                    if slot_idx in comp_slots_data:
                        data_pd.update({f'slot{slot_idx}_color': comp_slots_data[slot_idx]['color'],
                                        f'slot{slot_idx}_size': comp_slots_data[slot_idx]['size'],
                                        f'slot{slot_idx}_type': comp_slots_data[slot_idx]['type']})
                    else:
                        data_pd.update(
                            {f'slot{slot_idx}_color': -1, f'slot{slot_idx}_size': -1, f'slot{slot_idx}_type': -1})
                full_data_pd.append(data_pd)
    return pd.DataFrame(full_data_pd)