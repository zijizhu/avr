import cv2
import json
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.gridspec as gridspec
from xml.etree.ElementTree import Element


def parse_panels(root: Element):
    """
    Parse the panels object in the XML tree format into a dictionary.

    Parameters
    ----------
    root: Element
        The root of the XML tree

    Returns
    -------
    list
        A dictionary of panel information
    """
    panel_list = []
    all_panel_elements = root[0]
    for panel_el in all_panel_elements:
        panel_info = {}
        # Structure
        structure_el = panel_el[0]
        panel_info['structure'] = structure_el.attrib['name']
        # Components
        component_list = []
        for component_el in structure_el:
            comp_info = {'component': component_el.attrib}
            # Layout of the component
            layout_el = component_el[0]
            comp_info['layout'] = layout_el.attrib
            # Entities in the component
            entity_list = []
            for entity_el in layout_el:
                entity_list.append(entity_el.attrib)
            comp_info['entities'] = entity_list
            # Add component to component list
            component_list.append(comp_info)
        # Add panel to panel list
        panel_info['components'] = component_list
        panel_list.append(panel_info)
    return panel_list


def parse_rules(root: ET.Element):
    all_rule_groups = root[1]
    all_rules = []
    for rule_group_el in all_rule_groups:
        component_rules = {'component_id': rule_group_el.attrib['id'],
                           'rules': []}
        for rule_el in rule_group_el:
            component_rules['rules'].append(rule_el.attrib)
        all_rules.append(component_rules)
    return all_rules


def bbox_to_xyxy(bbox: list[float], image_size: int = 160):
    [center_y_ratio, center_x_ratio, width_ratio, height_ratio] = bbox
    width_px, height_px = width_ratio * image_size, height_ratio * image_size
    center_x_px, center_y_px = center_x_ratio * image_size, center_y_ratio * image_size
    top_left_x_px = int(center_x_px - width_px / 2)
    top_left_y_px = int(center_y_px - height_px / 2)
    bottom_right_x_px = int(center_x_px + width_px / 2)
    bottom_right_y_px = int(center_y_px + height_px / 2)
    return [top_left_x_px, top_left_y_px, bottom_right_x_px, bottom_right_y_px]


def plot_example(panels: np.ndarray, fig: plt.Figure | None = None):
    if not fig:
        fig = plt.figure(figsize=(8, 10))
    outer = gridspec.GridSpec(2, 1, height_ratios=[3, 2], wspace=0.2, hspace=0.2)

    # Context Panels
    inner = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=outer[0], wspace=0.1, hspace=0.1)
    for i in range(0, 8):
        ax = plt.Subplot(fig, inner[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(cv2.cvtColor(panels[i], cv2.COLOR_BGR2RGB))
        fig.add_subplot(ax)

    # Candidate Panels
    inner = gridspec.GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], wspace=0.1, hspace=0.1)
    for i in range(0, 8):
        ax = plt.Subplot(fig, inner[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(cv2.cvtColor(panels[8 + i], cv2.COLOR_BGR2RGB))
        fig.add_subplot(ax)
    return fig


'''
example_data = np.load('./distribute_nine/RAVEN_0_train.npz')
example_label_tree = ET.parse('./distribute_nine/RAVEN_0_train.xml')
example_label_root = example_label_tree.getroot()

panel_image = example_data['image'][0]
entities = panel_data[0]['components'][0]['entities']
for e in entities:
    bbox = json.loads(e['real_bbox'])
    top_left, bottom_right = to_cv2_bbox(bbox)
    img_with_bbox = cv2.rectangle(panel_image, top_left, bottom_right, (0,255,0), 3)
    
plt.imshow(cv2.cvtColor(panel_image2, cv2.COLOR_BGR2RGB))
plt.show()
'''