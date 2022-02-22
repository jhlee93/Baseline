import cv2
import numpy as np
import json

def data_load(annotation_path, image_dir):
    '''
    Input:
        COCO instance segmentation format
    Output:
        Image array, Mask array / Shape=(n, h, w, c=1)
    '''
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annots = json.load(f)
        
    image_info = annots['images']

    images, masks = [], []
    for i in image_info:
        file_name = i['file_name']
        image_id = int(i['id'])
        
        # Get segmentations / img
        segs = [x['segmentation'][0] for x in annots['annotations'] if int(x['image_id']) == image_id]
        segs = [np.array(x).reshape(-1,2).astype(int) for x in segs]
        
        # poly to mask
        img = cv2.imread(f'{image_dir}/{file_name}', 0)
        height, width = img.shape
        for sid in range(len(segs)):
            segs[sid][:,0] = np.clip(segs[sid][:,0], 0, width-1)
            segs[sid][:,1] = np.clip(segs[sid][:,1], 0, height-1)
        
        mask = np.zeros((height, width)).astype(np.uint8)
        mask = cv2.fillPoly(mask, segs, 1)
        
        mask = np.expand_dims(mask, axis=(0, -1))
        img = np.expand_dims(img, axis=(0, -1))
        
        images.append(img)
        masks.append(mask)
    
    images = np.concatenate(images)
    masks = np.concatenate(masks)
    
    
    # Normalication
    images = images.astype('float32') / 255.0
    
    return images, masks