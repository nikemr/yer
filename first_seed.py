import torch.nn as nn
import torch
import numpy as np
import torchvision.transforms as transforms
import pretrainedmodels

class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model =  pretrainedmodels.__dict__['resnet18'](pretrained='imagenet')
        
        self.classifier_layer = nn.Sequential(
            nn.Linear(512 , 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256 , 128),
            nn.ReLU(),
            nn.Linear(128 , 4)
        )
        
    def forward(self, x):
        batch_size ,_,_,_ = x.shape     #taking out batch_size from input image
        x = self.model.features(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)     # then reshaping the batch_size
        x = self.classifier_layer(x)
        return x

def model_loader():
    model_list=[]
    use_cuda = torch.cuda.is_available()
    
    for i in range(5):
        model_list.append(Resnet18())
        model_list[i].load_state_dict(torch.load(f'resnet_models/{i}.pt'))
        
    for model in model_list:
        for param in model.model.parameters():
            param.requires_grad = False
        model.eval()
        if use_cuda:
            model = model.cuda()
            
    return model_list

def predict(np_im,model):
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
    
    # if use_cuda:
    #     model.cuda()
    
    # model.eval()    
    
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

    
    return output

    
def process_image(np_im):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    np_im=np_im.reshape(224,224,3)
    transformations = transforms.Compose([transforms.ToTensor()])
    torch_image = transformations(np_im).float()    
    return torch_image

use_cuda = torch.cuda.is_available()