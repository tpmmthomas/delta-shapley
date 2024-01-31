import torch
import torch.nn.functional as F
import numpy as np
from torch import optim

def train(return_model, num_class, dataloader, params):
    """
    Training loop for NNs
    Args:
        model:
        optimizer:
        dataloader:
        params:
    Return model
    """
    print("Creating model")
    model = return_model(params,num_class)
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0)
    lr_lambda = lambda epoch: params.learning_rate / (epoch + 1)
    #scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print("Training")
    for epoch in np.arange(params.epochs):
        for (images, targets) in dataloader:
            model.zero_grad()
            optimizer.zero_grad()
            if params.device != 'cpu':
                images = images.to(params.device)
                targets = targets.to(params.device)
            output = model(images)
            loss = F.cross_entropy(output, targets, reduction='mean')
            loss.backward()
            optimizer.step()
            #scheduler.step()
            output = None
            loss = None
    print("Completed Training")
    return model