import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import json


def get_dataloaders(data_dir):
    
    #define training and validation data directories
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    image_transforms = {"train":train_transforms, "validation":validation_transforms}


    #  Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transforms)

    image_datasets = {"train":train_dataset, "validation":validation_dataset}

    #  Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=True)
    
    data_loaders = {"train":trainloader, "validation":validationloader}
    
    return data_loaders, train_dataset.class_to_idx


def get_model(arch):
    if arch == "vgg11":
        model = models.vgg11(pretrained=True)
    if arch == "vgg13":
        model = models.vgg13(pretrained=True)
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    if arch == "vgg19":
        model = models.vgg19(pretrained=True)
        
    return model


def change_classifier(model, hidden_units):
    
    #Turn of gradients for feature network
    for param in model.parameters():
        param.requires_grad = False
    
    #define new classifier with given number of hidden units
    classifier = nn.Sequential(OrderedDict([
                    ("fc1", nn.Linear(25088, hidden_units)),
                    ("relu", nn.ReLU()),
                    ("dropout", nn.Dropout(p=0.4)),
                    ("fc2", nn.Linear(hidden_units, 102)),
                    ("output", nn.LogSoftmax(dim=1))
    ]))
    
    #change the model classifier
    model.classifier = classifier
    
    return model



###### TRAIN THE MODEL ######

def train(model, dataloaders, learning_rate, epochs, gpu):
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=learning_rate)
    model.to(device)

    training_losses, validation_losses = [], []

    for e in range(epochs):
        train_loss = 0
        for images, labels in dataloaders["train"]:
            #move input and label tensors to gpu
            images, labels = images.to(device), labels.to(device)

            #reset gradients
            optimizer.zero_grad()

            log_ps = model.forward(images) #forward pass
            loss = criterion(log_ps, labels) #calculate loss
            loss.backward() #backward pass
            optimizer.step() #update gradients

            train_loss += loss.item()

        else:
            with torch.no_grad():
                model.eval()
                validation_loss = 0
                accuracy = 0
                for images, labels in dataloaders["validation"]:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images) #forward pass
                    validation_loss += criterion(log_ps, labels) #calculate loss

                    ps = torch.exp(log_ps) #get the probabilities
                    top_p, top_class = ps.topk(1, dim=1) #get the models predicted class for each example
                    equals = top_class == labels.view(*top_class.shape) #find how many predictions matched label
                    accuracy += torch.mean(equals.type(torch.FloatTensor)) #use previous step to get accuracy of network

                model.train()

        training_losses.append(train_loss/len(dataloaders["train"]))
        validation_losses.append(validation_loss/len(dataloaders["validation"]))

        print('Epoch: {}/{}'.format(e+1, epochs),
              'Train Loss: {:.3f}..'.format(train_loss/len(dataloaders["train"])),
              'Validation Loss: {:.3f}..'.format(validation_loss/len(dataloaders["validation"])),
              'Accuracy: {:.3f}'.format(accuracy/len(dataloaders["validation"])))

        
              
def save_checkpoint(model, destination, class_to_idx, arch, hidden_units):
    checkpoint = {"model classifier state_dict": model.classifier.state_dict(),
                  "class to idx": class_to_idx,
                  "arch": arch,
                  "hidden units": hidden_units}
    torch.save(checkpoint, destination)

        
        
def load_model(filepath, gpu):
    if torch.cuda.is_available():
        map_loc = lambda storage, loc: storage
    else:
        map_loc = 'cpu'
        
    checkpoint = torch.load(filepath, map_location=map_loc)
    arch  = checkpoint["arch"]
    classifier = checkpoint["model classifier state_dict"]
    class_to_idx = checkpoint["class to idx"]
    hidden_units = checkpoint["hidden units"]
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    
    
    model = get_model(arch)
    model = change_classifier(model, hidden_units)
    model.classifier.load_state_dict(classifier)
    model.class_to_idx = class_to_idx
    model.to(device)
    
    
    return model, device
   
    
    
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    with Image.open(image) as im:
        
        #define transforms
        image_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])
        
        #apply transforms
        norm_im = image_transforms(im)
        
        return norm_im
    
    
def predict(image_path, model, device, topk, category_names):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #get mapping from idx to class
    idx_to_class = {idx:data_class for data_class, idx in model.class_to_idx.items()} 
    
    img = process_image(image_path)
    img = img.view(1, *img.shape) #size = batch, channels, width, height
    img = img.to(device)
    
    ps = torch.exp(model.forward(img)) #get probs
    top_p, top_idx = ps.topk(topk) #get top k predictions
    top_p = top_p.detach()
    
    #translate from idx given by model to class numbers
    top_classes = [idx_to_class[idx] for idx in top_idx.cpu().numpy().squeeze()]
    
    #get mapping from categories to names of flowers
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    class_names = [cat_to_name[class_num] for class_num in top_classes]
    
    return top_p, class_names
