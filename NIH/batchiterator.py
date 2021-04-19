
import torch
from utils import *
import numpy as np
#from evaluation import *




PRINTFREQ=10


def BatchIterator(model, phase,
        Data_loader,
        criterion,
        optimizer,
        device,
        epoch):


    # --------------------  Initial paprameterd
    grad_clip = 0.5  # clip gradients at an absolute value of

    print_freq = 1000
    running_loss = 0.0

    for i, data in enumerate(Data_loader):
        top1 = AverageMeter_str('Acc@1', ':6.2f')
        progress = ProgressMeter(len(Data_loader),
                                top1, prefix="Epoch: [{}]".format(epoch))


        imgs, labels, _ = data

        batch_size = imgs.shape[0]
        imgs = imgs.to(device)
        labels = labels.to(device)

        if phase == "train":
            optimizer.zero_grad()
            model.train()
            outputs = model(imgs)
        else:

            model.eval()
            with torch.no_grad():
                outputs = model(imgs)

        print("first: ",outputs.shape)
        zero_append = torch.FloatTensor([[0] for i in range(32)])
        zero_append = zero_append.to(device)
        outputs=torch.cat([outputs, zero_append], dim=1)
        print("second: ", outputs.shape)
        loss = criterion(outputs, labels)

        if phase == 'train':

            loss.backward()
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)
            optimizer.step()  # update weights
            if i % PRINTFREQ == 0:
                progress.print(i)
        running_loss += loss * batch_size
        if (i % 200 == 0):
            print(str(i * batch_size))



    print('=> Acc@1 {top1.avg:.3f}'
          .format(top1=top1))
    return running_loss
