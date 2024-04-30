import torch
import torch.nn.functional as F
import torch.nn as nn
#time
import time

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def train(config, model, train_loader, valid_loader):
    
    num_epochs = config.num_epochs
    learning_rate = config.learning_rate 
    
    torch.cuda.empty_cache()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()


    train_loss = []
    valid_loss = []
    start_time = time.time()
    size_rates = [0.75, 1, 1.25]
    

    
    for epoch in range(num_epochs):
        model.train()
        train_epoch_loss = []
        valid_epoch_loss = []
            
        for img, mask in train_loader:
            img = img.cuda()
            mask = mask.cuda()
            for rate in size_rates:
                # augmentation
                if rate != 1:
                    img_in = F.interpolate(img, scale_factor=rate, mode='bilinear', align_corners=True)
                    mask_in = F.interpolate(mask, scale_factor=rate, mode='nearest')
                pred_mask = model(img_in)
                loss = criterion(pred_mask, mask_in)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if rate == 1:
                    train_epoch_loss.append(loss.item())
                    
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {np.mean(train_epoch_loss)}, elapsed time: {time.time() - start_time}")
        # save model
        if epoch % 5 == 0 or epoch == (num_epochs - 1) :
            torch.save(model.state_dict(), f"./checkpoint/{config.model}_{epoch}.pth")
        train_loss.append(np.mean(train_epoch_loss))
        
        # validation
        model.eval()
        with torch.no_grad():
            for img, mask in valid_loader:
                img = img.cuda()
                mask = mask.cuda()

                pred_mask = model(img)
                loss = criterion(pred_mask, mask)
            valid_epoch_loss.append(loss.item())
            print(f"Epoch [{epoch}/{num_epochs}], Validation Loss: {np.mean(valid_epoch_loss)}")
        valid_loss.append(np.mean(valid_epoch_loss))
            
    return model, train_loss, valid_loss


