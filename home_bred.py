import torch
import scipy
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # convolutional layer (sees 240x240x3 image tensor)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # convolutional layer (sees 120x120x16 tensor)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # convolutional layer (sees 60x60x32 tensor)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 30 * 30 -> 500)
        self.fc1 = nn.Linear(64 * 30 * 30, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 4)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 30 * 30)        
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))        
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x

model= Net()

for param in model.modules():
    # print(param)
    param.requires_grad = False


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
    np_im=np_im.reshape(240,240,3)
    transformations = transforms.Compose([transforms.ToTensor()])
    torch_image = transformations(np_im).float()    
    return torch_image

def home_bred_predict(np_im):
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




use_cuda = torch.cuda.is_available()


if use_cuda:
    model = model.cuda()
 
 