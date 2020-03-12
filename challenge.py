from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime

import numpy as np
from tqdm import tqdm
from matplotlib.ticker import NullLocator

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


# Object confidence threshold
CONF_THRES = 0.8

# IoU thresshold for non-maximum suppression
NMS_THRES = 0.4

# IoU threshold required to qualify as detected
IOU_THRE = 0.5

# Size of each image dimension
IMG_SIZE = 416


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_class_names(data_config):
    data_config = parse_data_config(data_config)
    class_names = load_classes(data_config['names'])
    return class_names


def load_model(weights_path, model_def, data_config):

    # Initiate model
    model = Darknet(model_def).to(device)
    if weights_path.endswith('.weights'):
        # Load darknet weights
        model.load_darknet_weights(weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(weights_path))

    model.eval()

    return model


def detect_imgs(model, imgs, conf_thres=CONF_THRES, nms_thres=NMS_THRES):
    
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float().to(device)
    else:
        imgs = imgs.type(Tensor)

    imgs = Variable(imgs, requires_grad=False)
    
    with torch.no_grad():
        detections = model(imgs)
        detections = non_max_suppression(detections, conf_thres=conf_thres, nms_thres=nms_thres)
        
    return detections


def get_dataloader(path, path_type='folder', img_size=IMG_SIZE, batch_size=8):
    assert path_type in ('folder', 'list')

    if path_type == 'folder':
        dataset = ImageFolder(path, img_size=img_size)
    elif path_type == 'list':
        dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
        
    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn
    )
    
    return dataloader


def parse_detections(detections, class_names):
    detection_data = []
    
    if detections is not None:
    
        for detection in detections:
            detection_np = detection.numpy()
            
            detection_dict = {'box': detection_np[:4],
                              # 'x1': detection_np[0],
                              # 'y1': detection_np[1],
                              # 'x2': detection_np[2],
                              # 'y2': detection_np[3],
                              'score': detection_np[4],
                               'label_id': int(detection_np[-1])}

            detection_dict['label'] = class_names[detection_dict['label_id']]
            # detection_dict['width'] = detection_dict['x2'] - detection_dict['x1']
            # detection_dict['hight'] = detection_dict['y2'] - detection_dict['y1']

            
            detection_data.append(detection_dict)

    return detection_data
    

def detect_folder(model, class_names, path, img_size=IMG_SIZE, batch_size=8, conf_thres=CONF_THRES, nms_thres=0.7,
                progress=tqdm):
    imgs = []
    img_detections = []
    
    dataloader = get_dataloader(path, 'folder', img_size, batch_size)
    
    for batch_i, (img_paths, input_imgs) in enumerate(progress(dataloader)):

        detections = detect_imgs(model, input_imgs)
        imgs.extend(img_paths)
        
        img_detections.extend(parse_detections(detections_as_tensors, class_names)
                              for detections_as_tensors in detections)

    return imgs, img_detections


def evaluate(model, path, iou_thres=IOU_THRE, conf_thres=CONF_THRES, nms_thres=NMS_THRES, img_size=IMG_SIZE, batch_size=8,
              progress=tqdm):

    dataloader = get_dataloader(path, 'list', img_size, batch_size)

    
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    
    for batch_i, (_, imgs, targets) in enumerate(progress(dataloader)):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        detections = detect_imgs(model, imgs, conf_thres, nms_thres)
        # imgs = Variable(imgs.type(Tensor), requires_grad=False)

        # with torch.no_grad():
        #     outputs = model(imgs)
        #     outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)
                
        sample_metrics += get_batch_statistics(detections, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def draw_img(path):
    img = np.array(Image.open(path))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    return ax


def draw_detections(path, detections, img_size=IMG_SIZE, output_path=None):

    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Iterate through images and save plot of detections
    if True:
        img_i = 0
    #for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = np.array(Image.open(path))
        
        ax = draw_img(path)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            # detections = rescale_boxes(detections, img_size, img.shape[:2])
            # unique_labels = detections[:, -1].cpu().unique()
            unique_labels = np.unique([detection_data['label_id'] for detection_data in detections])
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            # for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            for detection_data in detections:

                # Rescale boxes to original image
                x1, y1, x2, y2 = rescale_boxes(detection_data['box'][None, :].copy(),
                                               img_size, img.shape[:2])[0]
                
                box_w = x2 - x1
                box_h = y2 - y1
                
                color = bbox_colors[(np.where(unique_labels
                                              == detection_data['label_id']
                                             )[0][0])]

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                ax.text(
                    x1,
                    y1,
                    s=detection_data['label'],
                    color='white',
                    verticalalignment='top',
                    bbox={'color': color, 'pad': 0},
                )

        if (output_path is not None
            and not output_path.endswith(('.png', '.jpg', '.jpeg'))):
            
            output_path += 'png'
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
