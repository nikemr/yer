
import torch.nn as nn
import torch
import numpy as np
import efficientnet_pytorch 
from PIL import Image
import torchvision.transforms as transforms
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class EfficientNet_b0(nn.Module):
    def __init__(self):
        super(EfficientNet_b0, self).__init__()
        self.model = efficientnet_pytorch.EfficientNet.from_pretrained('efficientnet-b0')
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(1280 , 512),
            nn.ReLU(),
            #nn.BatchNorm1d(512),
            #nn.Dropout(0.2),
            nn.Linear(512 , 256),
            nn.ReLU(),
            nn.Linear(256 , 4)
        )
        
    def forward(self, inputs):
        x = self.model.extract_features(inputs)

        # Pooling and final linear layer
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.model._dropout(x)
        x = self.classifier_layer(x)
        return x


def predict(np_im):
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
        model.cuda()
    
    model.eval()    
    
    with torch.no_grad():
       
        image = process_image(np_im)
        if use_cuda:
            image = image.type(torch.FloatTensor).cuda()   
        #I do not understand why we shold do this why not 3 dimensions are not sufficent.
        #print(image.shape)
        image = image.unsqueeze(0)
        #print(image.shape)
        
        output = model.forward(image)
        output =output.cpu().detach().numpy()
        #ps = torch.exp(logps)
        #_,cls_idx=torch.max(output, 1)
        #probs_tensor, classes_tensor = ps.topk(1,dim=1)

    #return probs_tensor, classes_tensor    
    #return cls_idx.item()
    
    return output

    
def process_image(np_im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #pil_im = Image.open(img_path, 'r')

    #pil_im.thumbnail((240,240))
    #pil_im=pil_im.crop((16,16,240,240))


    #ish(np.asarray(pil_im))
    #np_image = np.array(pil_im)
    
    #couldn't make it the other way
    np_im=np_im.reshape(224,224,3)
    transformations = transforms.Compose([transforms.ToTensor()])
    torch_image = transformations(np_im).float()    
    return torch_image

use_cuda = torch.cuda.is_available()
print("Cuda Available")
model= EfficientNet_b0()
#print(model)
# model.model seems weird but check sandbox/EffcientNet.ipynb
for param in model.model.parameters():
    param.requires_grad = False



if use_cuda:
    model = model.cuda()




