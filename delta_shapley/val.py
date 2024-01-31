import torch
import torch.nn.functional as F
import numpy as np
from torch import optim


def evaluate_model(model, testloader, params):
    """
    Computes the validation loss of a model
    // Can be changed to accuracy or something else
    """
    print("Start evaluation")
    len_test = 0
    optimizer = optim.SGD(model.parameters(), lr=params.learning_rate, momentum=0)
    with torch.no_grad():
        losses = []
        for (images, targets) in testloader:
            model.zero_grad()
            optimizer.zero_grad()
            if params.device != 'cpu':
                images = images.cuda()
                targets = targets.cuda()
            output = model(images)
            loss = F.cross_entropy(output, targets, reduction='sum')
            losses.append(loss.clone().detach())
            output = None
            loss = None
            len_test += 1
        overall_test_loss = sum(losses) / len_test
    print("Complete evaluation")
    return overall_test_loss.item()