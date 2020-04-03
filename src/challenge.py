import os
import sys
import time
import datetime
import itertools as it
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pylab as plt
from matplotlib.ticker import NullLocator
import PIL

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

from src.models import *
from src.utils.utils import *
from src.utils.datasets import *
from src.utils.parse_config import *
from src.augmentation import automold as am
from src.augmentation import helpers as hp


# Object confidence threshold
CONF_THRES = 0.8

# IoU thresshold for non-maximum suppression
NMS_THRES = 0.4

# IoU threshold required to qualify as detected
IOU_THRE = 0.5

# Size of each image dimension
IMG_SIZE = 416


LABEL_FOLDER_PATH = './data/custom/labels'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


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
        
    if isinstance(imgs, np.ndarray):
        imgs = torch.from_numpy(imgs).float().to(device)
    else:
        imgs = imgs.type(Tensor)

    imgs = Variable(imgs, requires_grad=False)
    
    with torch.no_grad():
        detections = model(imgs)
        detections = non_max_suppression(detections, conf_thres=conf_thres, nms_thres=nms_thres)
        
    return detections


def get_dataloader(path, path_type='folder', img_size=IMG_SIZE, batch_size=8,
                   with_targets=True, label_folder_path=None):
    assert path_type in ('folder', 'list', 'paths')

    if path_type == 'folder':
        dataset = ImageFolder(path, img_size=img_size)
    elif path_type == 'list':
        dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False,
                              label_folder_path=label_folder_path)
    elif path_type == 'paths':
        dataset = ImagePaths(path, img_size=img_size)

    collate_fn = dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=collate_fn
    )
    
    if not with_targets and path_type == 'list':
        dataloader = (item for *item, _ in dataloader)
        
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
    

def detect_multi(model, class_names, path, path_type='list', img_size=IMG_SIZE, batch_size=8, conf_thres=CONF_THRES, nms_thres=0.7, label_folder_path=LABEL_FOLDER_PATH, progress=tqdm):
    imgs = []
    img_detections = []
    
    dataloader = get_dataloader(path, path_type, img_size, batch_size,
                                with_targets=False, label_folder_path=label_folder_path)

    for batch_i, (img_paths, input_imgs) in enumerate(progress(dataloader)):

        detections = detect_imgs(model, input_imgs)
        imgs.extend(img_paths)
        
        img_detections.extend(parse_detections(detections_as_tensors, class_names)
                              for detections_as_tensors in detections)

    return imgs, img_detections


def evaluate(model, paths, class_names, label_folder_path=None,
              iou_thres=0.5, conf_thres=0.8, nms_thres=0.4,
              img_size=416, batch_size=1,
              progress=iter, report=False):
    
    assert batch_size == 1, 'Batch size should be equal to 1.'
   
    dataloader = get_dataloader(paths, 'list',
                                img_size, batch_size,
                                label_folder_path=label_folder_path)

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    detections = []
    
    evaluations = []
    
    for batch_i, (batch_paths, imgs, targets) in enumerate(progress(dataloader)):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        detections = detect_imgs(model, imgs, conf_thres, nms_thres)
        statistics = get_batch_statistics(detections, targets, iou_threshold=iou_thres)

        sample_metrics += statistics        

        for img_in_batch_i, (path, metrics, img_detections) in enumerate(it.zip_longest(batch_paths, statistics, detections,
                                                                        fillvalue=[])):
                
            target_mask = (targets[:, 0] == img_in_batch_i)
            target_labels = targets[target_mask, 1]
            
            evaluations.append({'path': path,
                                'tp': list(metrics[0]) if metrics else [],
                                'pred_confs': list(metrics[1].numpy()) if metrics else [],
                                'pred_labels': [class_names[int(label_index)]
                                          for label_index in metrics[2]] if metrics else [],
                                'true_labels': [class_names[int(label_index)]
                                          for label_index in target_labels],
                                'detections': parse_detections(img_detections, class_names),})

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels,
                                                       progress=progress)

    mAP = AP.mean()
    
    evaluations_df = pd.DataFrame(evaluations)

    # Only for the detection of boxes, not classes
    evaluations_df['precision'] = evaluations_df['tp'].apply(lambda tp: np.mean(tp))
    evaluations_df['recall'] = (evaluations_df['tp'].apply(sum)
                                / evaluations_df['true_labels'].apply(len))

    cols = list(evaluations_df.columns)
    cols.append(cols.pop(cols.index('detections')))
    evaluations_df = evaluations_df[cols]


    if report:
        print("Average Precisions:")
        for i, c in enumerate(ap_class):
            print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]}")

        print(f"mAP: {mAP}")

    return precision, recall, AP, f1, ap_class, mAP, evaluations_df


def draw_img(path, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)

    img = np.array(Image.open(path))
    ax.imshow(img)
    plt.tight_layout()
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    return ax


def draw_detections(path, detections, img_size=IMG_SIZE, output_path=None, ax=None):
    # Bounding-box colors
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    # Iterate through images and save plot of detections
    if True:
        img_i = 0
    #for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        # Create plot
        img = np.array(Image.open(path))
        
        ax = draw_img(path, ax)

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
                # linewidth=2
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                ax.text(
                    x1,
                    y1-20,
                    s=detection_data['label'],
                    fontsize=10, #new
                    color='white',
                    verticalalignment='top',
                    bbox={'color': color, 'pad': 0},
                )

        if (output_path is not None
            and not output_path.endswith(('.png', '.jpg', '.jpeg'))):
            
            output_path += 'png'
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0.0)
    return ax


def draw_evaluation(evaluation, title_prefix='', ax=None):
    ax = draw_detections(evaluation['path'], evaluation['detections'], ax=ax)
    ax.set_title(f'{title_prefix}Recall={evaluation.recall}\nPrecision={evaluation.precision}')
    ax.axis('off')
    return ax


def draw_multiple_detections(evaluations_df):
    size = int(np.ceil(np.sqrt(len(evaluations_df))))
    f, axarr = plt.subplots(size, size, figsize=(10, 10))
    axes = axarr.flatten()
    
    for (index, row), ax in zip(evaluations_df.iterrows(), axes):
        draw_evaluation(row, f'Index={index}\n', ax=ax)
    
    plt.tight_layout()
    
    return axes


def apply_augmentation(paths_file, aug_func, **aug_params):
    
    paths_file = Path(paths_file)

    func_name = aug_func.__name__
    func_params_desc = '-'.join(map(str, aug_params.values())).replace('.', '_')
    
    if func_params_desc:
        func_params_desc = '_' + func_params_desc
    idetifier = f'{paths_file.stem}--aug-{func_name}{func_params_desc}'
    
    output_path = paths_file.parent  / idetifier
    output_path.mkdir(exist_ok=True)
    
    img_paths = paths_file.read_text().splitlines()

    
    output_img_paths = []
    
    for img_path in img_paths:
    
        img = np.array(PIL.Image.open(img_path))

        img = aug_func(img, **aug_params)

        # Changed to PNG because JPEG is lossy
        filename = Path(img_path).stem + '.png'
        output_img_path = str(output_path / filename)
        output_img_paths.append(output_img_path)
        
        PIL.Image.fromarray(img).save(output_img_path)

    output_paths_file = paths_file.parent / idetifier
    output_paths_file = output_paths_file.with_suffix('.txt')
    output_paths_file.write_text('\n'.join(output_img_paths))
    
    return str(output_paths_file)
