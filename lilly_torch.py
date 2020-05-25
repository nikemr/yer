import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn
model_transfer = models.vgg16(pretrained=True)

from PIL import Image
import torchvision.transforms as transforms

# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ORIGINAL FUCTION --------------------------------------------------------------------------------------
# def process_image(img_path):
#     ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
#         returns an Numpy array
#     '''
#     pil_im = Image.open(img_path, 'r')

#     pil_im.thumbnail((240,240))
#     #pil_im=pil_im.crop((16,16,240,240))


#     #ish(np.asarray(pil_im))
#     np_image = np.array(pil_im)
    
#     #couldn't make it the other way
#     transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),
#                                                                (0.229, 0.224, 0.225))])
#     np_image = transformations(np_image).float()    
#     return np_image

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    pil_im = Image.open(img_path, 'r')

    pil_im.thumbnail((240,240))
    #pil_im=pil_im.crop((16,16,240,240))


    #ish(np.asarray(pil_im))
    np_image = np.array(pil_im)
    
    #couldn't make it the other way
    transformations = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))])
    np_image = transformations(np_image).float()    
    return np_image

def VGG16_predict(img_path):
    '''
    Use pre-trained VGG-16 model to obtain index corresponding to 
    predicted ImageNet class for image at specified path
    
    Args:
        img_path: path to an image
        
    Returns:
        Index corresponding to VGG-16 model's prediction
    '''
    
    ## TODO: Complete the function.
    ## Load and pre-process an image from the given img_path
    ## Return the *index* of the predicted class for that image
    
    if use_cuda:
        model_transfer.cuda()
    
    model_transfer.eval()    
    
    with torch.no_grad():
       
        image = process_image(img_path)
        if use_cuda:
            image = image.type(torch.FloatTensor).cuda()   
        #I do not understand why we shold do this why not 3 dimensions are not sufficent.
        #print(image.shape)
        image = image.unsqueeze(0)
        #print(image.shape)
        
        output = model_transfer.forward(image)
        
        #ps = torch.exp(logps)
        _,cls_idx=torch.max(output, 1)
        #probs_tensor, classes_tensor = ps.topk(1,dim=1)
       
        
              
            
    #return probs_tensor, classes_tensor    
    #return cls_idx.item()
    return output

# print(model_transfer)

for param in model_transfer.features.parameters():
    param.requires_grad = False
    
#number thrusters on the Lilly   
thruster=2

# new layers automatically have requires_grad = True
second_layer= nn.Linear(4096, 2048)
model_transfer.classifier[3] = second_layer

last_layer = nn.Linear(2048, thruster)
model_transfer.classifier[6] = last_layer
print(model_transfer)
# check if CUDA is available
use_cuda = torch.cuda.is_available()

# move model to GPU if CUDA is available
if use_cuda:
    model_transfer = model_transfer.cuda()

# predict_that=VGG16_predict("MYBUFFER.jpg")
# print(predict_that)    