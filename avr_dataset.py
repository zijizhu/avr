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

configurations = [
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


def panel_dict_to_df(indices: list | range, panel_dict: dict, file_path: str):
    full_panel_data = []
    for panel_idx, panel in zip(indices, panel_dict):
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

            data = {'file': str(file_path), 'panel': int(panel_idx), 'component': int(component_idx)}
            data_pd = data.copy()

            for slot_idx in range(0, len(id2slot)):
                if slot_idx in comp_slots_data:
                    data_pd.update({f'slot{slot_idx}_color': comp_slots_data[slot_idx]['color'],
                                    f'slot{slot_idx}_size': comp_slots_data[slot_idx]['size'],
                                    f'slot{slot_idx}_type': comp_slots_data[slot_idx]['type']})
                else:
                    data_pd.update(
                        {f'slot{slot_idx}_color': -1, f'slot{slot_idx}_size': -1, f'slot{slot_idx}_type': -1})
            full_panel_data.append(data_pd)
    return pd.DataFrame(full_panel_data)


def extract_stage2_ground_truth(dataset_dir: str, split: str):
    dataset_path = Path(dataset_dir)
    all_file_stems = list(fn.stem for fn in (dataset_path / Path(configurations[0])).glob(f'*_{split}.npz'))
    all_file_paths = [Path(dataset_path, config, base_fn) for config, base_fn in
                      product(configurations, all_file_stems)]

    all_panel_df = []
    full_rule_data = []

    for file_path in all_file_paths:
        xml = ET.parse(file_path.with_suffix('.xml'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        component_rules = parse_rules(xml_root)
        context_panels = panel_info_list[:6]

        # Get rules (labels)
        for component in component_rules:
            rule_data = {'file_path': str(file_path), 'component': int(component['component_id'])}
            for rule in component['rules']:
                if (rule['attr'] == 'Number/Position') or (rule['attr'] == 'Number') or (rule['attr'] == 'Position'):
                    rule_data['number'] = rule['name']
                    rule_data['position'] = rule['name']
                elif rule['attr'] == 'Type':
                    rule_data['type'] = rule['name']
                elif rule['attr'] == 'Size':
                    rule_data['size'] = rule['name']
                elif rule['attr'] == 'Color':
                    rule_data['color'] = rule['name']
            full_rule_data.append(rule_data)

        # Get discrete panel representations (features)
        panel_df = panel_dict_to_df(range(len(context_panels)), context_panels, str(file_path))
        all_panel_df.append(panel_df)

    return pd.concat(all_panel_df).reset_index(drop=True), pd.DataFrame(full_rule_data)


def prepare_stage2_dataset(panels_df: pd.DataFrame, rules_df: pd.DataFrame, merge_row=True):
    panels_df_copy = panels_df.copy()

    if merge_row:
        panels_df_copy['row'] = np.where(panels_df_copy['panel'] < 3, 0, 1)
        panels_df_copy['panel'] = np.where(panels_df_copy['panel'] < 3,
                                           panels_df_copy['panel'],
                                           panels_df_copy['panel'] - 3)
        reshaped_indices = ['file', 'component', 'row', 'panel']
    else:

        reshaped_indices = ['file', 'component', 'panel']

    reshaped_panels_df = panels_df_copy.set_index(reshaped_indices).unstack(level=-1)
    reshaped_panels_df.columns.names = ['slot_attr', 'panel']
    reshaped_panels_df.columns = reshaped_panels_df.columns.swaplevel(0, 1)
    reshaped_panels_df = reshaped_panels_df.sort_index(axis=1, level=0)

    index_tuples = []
    for panel_idx, slot_idx, attr in list(product(range(0, 3) if merge_row else range(0, 6),
                                                  range(0, 22),
                                                  ['color', 'size', 'type'])):
        index_tuples.append((panel_idx, f'slot{slot_idx}_{attr}'))
    # index_tuples = list(product(range(0, 6), range(0, 22), ['color', 'size', 'type']))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['panel', 'slot_attr'])
    reshaped_panels_df = pd.DataFrame(reshaped_panels_df, columns=multi_index)

    reshaped_panels_df.columns = reshaped_panels_df.columns.map(lambda x: 'panel' + '_'.join(list(map(str, x))))

    rules_df = rules_df.rename(columns={'file_path': 'file'})
    rules_df = rules_df.set_index(['file', 'component'])

    final_df = reshaped_panels_df.join(rules_df)

    return final_df


class AVRStage2Dataset(Dataset):
    def __init__(self, dataset_dir, split):
        super().__init__()
        panels_df, rules_df = extract_stage2_ground_truth(dataset_dir, split)
        self.final_df = prepare_stage2_dataset(panels_df, rules_df, merge_row=False)
        self.final_df = self.final_df.reset_index()
        self.info_cols = self.final_df.columns.tolist()[:2]
        self.feature_cols = self.final_df.columns.tolist()[2:-5]
        self.label_cols = self.final_df.columns.tolist()[-5:]
        self.label2id = {'Constant': 0, 'Distribute_Three': 1, 'Progression': 2, 'Arithmetic': 3}

    def __len__(self):
        return len(self.final_df)

    def __getitem__(self, idx):
        data = self.final_df.iloc[idx]

        info = data[self.info_cols].to_dict()

        panels = torch.split(torch.tensor(data[self.feature_cols].values.astype(np.int64)), 22 * 3)
        reshaped_panels = list(torch.stack(torch.split(p, 3)) for p in panels)
        features = torch.stack(reshaped_panels)

        labels = data[self.label_cols].map(self.label2id).to_dict()
        for key, val, in labels.items():
            labels[key] = torch.tensor(val)

        return {
            'info': info,
            'features': features,
            'labels': labels
        }


# Stage 3

def extract_stage3_ground_truth(dataset_dir: str, split: str):
    dataset_path = Path(dataset_dir)
    all_file_stems = list(fn.stem for fn in (dataset_path / Path(configurations[0])).glob(f'*_{split}.npz'))
    all_file_paths = [Path(dataset_path, config, base_fn) for config, base_fn in
                      product(configurations, all_file_stems)]

    all_panel_df = []
    full_rule_data = []
    full_target_data = []

    for file_path in all_file_paths:
        xml = ET.parse(file_path.with_suffix('.xml'))
        npz = np.load(file_path.with_suffix('.npz'))
        xml_root = xml.getroot()
        panel_info_list = parse_panels(xml_root)
        component_rules = parse_rules(xml_root)
        context_panels = panel_info_list[6:]
        
        full_target_data.append({'file': str(file_path), 'target': npz['target'].item()})

        # Get rules (labels)
        for component in component_rules:
            cid = int(component['component_id'])
            rule_data = {'file_path': str(file_path)}
            for rule in component['rules']:
                if (rule['attr'] == 'Number/Position') or (rule['attr'] == 'Number') or (rule['attr'] == 'Position'):
                    rule_data[f'component{cid}_number'] = rule['name']
                    rule_data[f'component{cid}_position'] = rule['name']
                elif rule['attr'] == 'Type':
                    rule_data[f'component{cid}_type'] = rule['name']
                elif rule['attr'] == 'Size':
                    rule_data[f'component{cid}_size'] = rule['name']
                elif rule['attr'] == 'Color':
                    rule_data[f'component{cid}_color'] = rule['name']
            full_rule_data.append(rule_data)

        # Get discrete panel representations (features)
        panel_df = panel_dict_to_df(range(6, 16), context_panels, str(file_path))
        all_panel_df.append(panel_df)

    return (pd.concat(all_panel_df).reset_index(drop=True),
            pd.DataFrame(full_rule_data),
            pd.DataFrame(full_target_data))


def prepare_stage3_dataset(panels_df: pd.DataFrame, rules_df: pd.DataFrame, target_df: pd.DataFrame):
    panels_df_copy = panels_df.copy()
    reshaped_indices = ['file', 'component', 'panel']

    reshaped_panels_df = panels_df_copy.set_index(reshaped_indices).unstack(level=-1)
    reshaped_panels_df.columns.names = ['slot_attr', 'panel']
    reshaped_panels_df.columns = reshaped_panels_df.columns.swaplevel(0, 1)
    reshaped_panels_df = reshaped_panels_df.sort_index(axis=1, level=0)

    index_tuples = []
    for panel_idx, slot_idx, attr in list(product(range(6, 16),
                                                  range(0, 22),
                                                  ['color', 'size', 'type'])):
        index_tuples.append((panel_idx, f'slot{slot_idx}_{attr}'))
    multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['panel', 'slot_attr'])
    reshaped_panels_df = pd.DataFrame(reshaped_panels_df, columns=multi_index)

    reshaped_panels_df.columns = reshaped_panels_df.columns.map(lambda x: 'panel' + '_'.join(list(map(str, x))))
    reshaped_panels_df = reshaped_panels_df.groupby('file').max()
    
    # return reshaped_panels_df
    rules_df = rules_df.rename(columns={'file_path': 'file'})
    rules_df = rules_df.set_index(['file'])

    final_df = reshaped_panels_df.join(rules_df).join(target_df.set_index(['file']))

    return final_df


class AVRStage3Dataset(Dataset):
    def __init__(self, dataset_dir, split):
        super().__init__()
        panels_df, rules_df, targets_df = extract_stage3_ground_truth(dataset_dir, split)
        self.final_df = prepare_stage3_dataset(panels_df, rules_df, targets_df)
        self.final_df = self.final_df.reset_index()
        self.info_col = self.final_df.columns.tolist()[0]
        self.panel_cols = self.final_df.columns.tolist()[1:-11]
        self.rule_cols = self.final_df.columns.tolist()[-11:-1]
        self.target_col = self.final_df.columns.tolist()[-1]
        self.rule2id = {'Constant': 0, 'Distribute_Three': 1, 'Progression': 2, 'Arithmetic': 3, -1: -1}
    
    def __len__(self):
        return len(self.final_df)
    
    def __getitem__(self, idx):
        data = self.final_df.iloc[idx]

        info = data[self.info_col]

        panels = torch.split(torch.tensor(data[self.panel_cols].values.astype(np.int64)), 22 * 3)
        reshaped_panels = list(torch.stack(torch.split(p, 3)) for p in panels)
        panel_features = torch.stack(reshaped_panels)
        
        rules = data[self.rule_cols].replace(np.nan, -1).map(self.rule2id).to_dict()
        for key, val, in rules.items():
            rules[key] = torch.tensor(val)

        return {
            'info': info,
            'panels': panel_features,
            'rules': rules,
            'target': torch.tensor(data[self.target_col])
        }
