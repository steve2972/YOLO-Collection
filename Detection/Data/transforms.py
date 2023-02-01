import torch 
import torchvision.transforms.functional as F
import random

from PIL import Image
from typing import List
from Detection.Utils.utils import compute_iou

from torchvision.transforms.transforms import (
    PILToTensor, ConvertImageDtype,
    Compose, Normalize,
    RandomAutocontrast, RandomAdjustSharpness
)
import torchvision.transforms.transforms as T

#------------------------Image-only transforms------------------------#

def distort(image):
    ''' Distort brightness, contrast, saturation
    Args:
        image: A PIL image    
    Returns:
        (PIL Image)
    '''
    if type(image) != Image.Image:
        image = F.to_pil_image(image)
    new_image = image
    distortions = [F.adjust_brightness,
                  F.adjust_contrast,
                  F.adjust_saturation]
    
    random.shuffle(distortions)
    
    for function in distortions:
        if random.random() < 0.5:
            adjust_factor = random.uniform(0.5, 1.5)
            new_image = function(new_image, adjust_factor)
            
    return new_image

def lighting_noise(image):
    '''Color channel swap in image

    Args:
        image: A PIL image
    Returns:
        (PIL.Image) New image with swap channel (Probability = 0.5)
    '''
    if type(image) != Image.Image:
        image = F.to_pil_image(image)
    new_image = image
    if random.random() < 0.5:
        perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
                 (1, 2, 0), (2, 0, 1), (2, 1, 0))
        swap = perms[random.randint(0, len(perms)- 1)]
        new_image = F.to_tensor(new_image)
        new_image = new_image[swap, :, :]
        new_image = F.to_pil_image(new_image)
    return new_image

#------------------------Image+Bbox transforms------------------------#

def resize(
        image:Image, 
        boxes:torch.Tensor, 
        dims:List[int]=(448, 448), 
        return_percent_coords:bool=True):
    ''' Resizes image and bounding box coordinates

    Args:
        image (PIL Image): A PIL image
        boxes (Tensor): a tensor of bbox dimensions (n_objects, 4)
        
    Returns:
        (PIL Image, Tensor): new image, new boxes or percent coordinates
    '''
    if type(image) != Image.Image:
        image = F.to_pil_image(image)
    new_image= F.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes

def expand_filler(image, boxes, filler):
    '''
        Perform a zooming out operation by placing the 
        image in a larger canvas of filler material. Helps to learn to detect 
        smaller objects.
        image: input image, a tensor of dimensions (3, original_h, original_w)
        boxes: bounding boxes, a tensor of dimensions (n_objects, 4)
        filler: RBG values of the filler material, a list like [R, G, B]
        
        Out: new_image (A Tensor), new_boxes
    '''
    if type(image) == Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale*original_h)
    new_w = int(scale*original_w)
    
    #Create an image with the filler
    filler = torch.FloatTensor(filler) #(3)
    new_image = torch.ones((3, new_h, new_w), dtype= torch.float) * filler.unsqueeze(1).unsqueeze(1)
    
    # Place the original image at random coordinates 
    #in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    
    new_image[:, top:bottom, left:right] = image
    
    #Adjust bounding box
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(0)
    
    return new_image, new_boxes

def random_crop(image, boxes, labels, difficulties):
    '''
        Performs a random crop. Helps to learn to detect larger and partial object
        image: A tensor of dimensions (3, original_h, original_w)
        boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)
        labels: labels of object, a tensor of dimensions (n_objects)
        difficulties: difficulties of detect object, a tensor of dimensions (n_objects)
        
        Out: cropped image (Tensor), new boxes, new labels, new difficulties
    '''
    if type(image) == Image.Image:
        image = F.to_tensor(image)
    original_h = image.size(1)
    original_w = image.size(2)
    
    while True:
        mode = random.choice([0.1, 0.3, 0.5, 0.9, None])
        
        if mode is None:
            return image, boxes, labels, difficulties
        
        new_image = image
        new_boxes = boxes
        new_difficulties = difficulties
        new_labels = labels
        for _ in range(50):
            # Crop dimensions: [0.3, 1] of original dimensions
            new_h = random.uniform(0.3*original_h, original_h)
            new_w = random.uniform(0.3*original_w, original_w)
            
            # Aspect ratio constraint b/t .5 & 2
            if new_h/new_w < 0.5 or new_h/new_w > 2:
                continue
            
            #Crop coordinate
            left = random.uniform(0, original_w - new_w)
            right = left + new_w
            top = random.uniform(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([int(left), int(top), int(right), int(bottom)])
            
            # Calculate IoU  between the crop and the bounding boxes
            overlap = compute_iou(crop.unsqueeze(0), boxes) #(1, n_objects)
            overlap = overlap.squeeze(0)
            # If not a single bounding box has a IoU of greater than the minimum, try again
            if overlap.max().item() < mode:
                continue
            
            #Crop
            new_image = image[:, int(top):int(bottom), int(left):int(right)] #(3, new_h, new_w)
            
            #Center of bounding boxes
            center_bb = (boxes[:, :2] + boxes[:, 2:])/2.0
            
            #Find bounding box has been had center in crop
            center_in_crop = (center_bb[:, 0] >left) * (center_bb[:, 0] < right
                             ) *(center_bb[:, 1] > top) * (center_bb[:, 1] < bottom)    #(n_objects)
            
            if not center_in_crop.any():
                continue
            
            #take matching bounding box
            new_boxes = boxes[center_in_crop, :]
            
            #take matching labels
            new_labels = labels[center_in_crop]
            
            #take matching difficulities
            new_difficulties = difficulties[center_in_crop]
            
            #Use the box left and top corner or the crop's
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])
            
            #adjust to crop
            new_boxes[:, :2] -= crop[:2]
            
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:],crop[2:])
            
            #adjust to crop
            new_boxes[:, 2:] -= crop[:2]
            
            return new_image, new_boxes, new_labels, new_difficulties
    
        return new_image, new_boxes, new_labels, new_difficulties 

def random_flip(image, boxes):
    '''
        Flip image horizontally.
        image: a PIL image
        boxes: Bounding boxes, a tensor of dimensions (n_objects, 4)
        
        Out: flipped image (A PIL image), new boxes
    '''
    if type(image) != Image.Image:
        image = F.to_pil_image(image)
    if random.random() > 0.5:
        return image, boxes
    new_image = F.hflip(image)
    
    #flip boxes 
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    return new_image, new_boxes

#------------------------Combine transforms------------------------#

def transform(image, boxes, labels, difficulties, split, dims=(448,448), difficult=False):
    ''' Apply transformation

    Args:
        image: A PIL image
        boxes: bounding boxe, a tensor of dimensions (n_objects, 4)
        labels: labels of object a tensor of dimensions (n_object)
        difficulties: difficulties of object detect, a tensor of dimensions (n_object)
        split: one of "TRAIN", "VAL"
        
    Returns:
        transformed images, transformed bounding boxes, transformed labels,
        transformed difficulties
    '''
    if type(image) != Image.Image:
        image = F.to_pil_image(image)
    split = split.upper()
    if split not in {"TRAIN", "VAL"}:
        raise NameError(f"Param split in transform not one of (Train, VAL). Received {split}")
    
    #mean and std from ImageNet [RGB]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    new_image = image
    new_boxes = boxes
    new_labels = labels
    new_difficulties = difficulties
    #Skip transform for VALing
    if split == "TRAIN":
        if difficult:
            #Apply distort image

            new_image = distort(new_image)
            
            #Apply lighting noise
            new_image = lighting_noise(new_image)
            #Expand image
            if random.random() < 0.5:
                new_image, new_boxes = expand_filler(new_image, boxes, mean)
            
            #Random crop
            new_image, new_boxes, new_labels, new_difficulties = random_crop(new_image, 
                                                                            new_boxes, 
                                                                            new_labels, new_difficulties)
            
            #Flip image
            new_image, new_boxes = random_flip(new_image, new_boxes)

        # Random Augment
        augment = Compose([
            RandomAutocontrast(),
            RandomAdjustSharpness(0.5),
            T.RandomApply([
                T.ColorJitter(),
                T.GaussianBlur((7,7)),

            ]),
            T.RandomPosterize(bits=2),
            T.RandomEqualize()
        ])
        new_image = augment(new_image)

    
        
    new_image, new_boxes = resize(new_image, new_boxes, dims=dims, return_percent_coords=False)

    norm_transform = Compose([
        PILToTensor(),
        ConvertImageDtype(torch.float),
        Normalize(mean=mean, std=std)
    ])

    new_image = norm_transform(new_image)
        
    return new_image, new_boxes, new_labels, new_difficulties