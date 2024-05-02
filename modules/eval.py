

from .metrics import cal_fm, cal_mae, cal_dice, cal_ber, cal_acc, cal_iou, cal_sm, cal_em, cal_wfm

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def test(model, test_loader,len_dataset):
    # test, 그리고 metric 계산.
    #  loss, accuracy, dice score 계산
    model.eval()
    criterion = nn.BCELoss()
    
    test_loss = []
    
    # get data size test_loader. not iteration
    
    cal_fm_ = cal_fm(len_dataset)
    cal_mae_ = cal_mae()
    cal_dice_ = cal_dice()
    cal_ber_ = cal_ber()
    cal_acc_ = cal_acc()
    cal_iou_ = cal_iou()
    cal_sm_ = cal_sm()
    cal_em_ = cal_em()
    cal_wfm_ = cal_wfm()
    
    
    
    with torch.no_grad():
        for img, mask in test_loader:
            img = img.cuda()
            mask = mask.cuda()

            pred_mask = model(img)
            loss = criterion(pred_mask, mask)
            test_loss.append(loss.item())

            pred_mask[pred_mask > 0.5] = 1
            pred_mask[pred_mask <= 0.5] = 0
            
            
            # metric 계산
            for i in range(len(pred_mask)):
                pred_res = pred_mask[i][0].cpu().numpy()
                gt_res = mask[i][0].cpu().numpy()
                
                cal_fm_.update(pred_res, gt_res)
                cal_mae_.update(pred_res, gt_res)
                cal_dice_.update(pred_res, gt_res)
                cal_ber_.update(pred_res, gt_res)
                cal_acc_.update(pred_res, gt_res)
                cal_iou_.update(pred_res, gt_res)
                cal_sm_.update(pred_res, gt_res)
                cal_em_.update(pred_res, gt_res)
                cal_wfm_.update(pred_res, gt_res)
                
            
            

    # print(f"Dice Score: {np.mean(dice_score)}")
    # print(f"IoU Score: {np.mean(iou_score)}")
    # print(f"Enhanced Alignment Score: {np.mean(f_e_score)}")
    # print(f"MAE Score: {np.mean(mae_score)}")
    # print(f"Structure Measure: {np.mean(structure_measure_iist)}")
    # print(f"Weighted F-beta Score: {np.mean(weighted_dice)}")
    
    # print("----")
    # print(f"Frequency-tuned maxFm: {cal_fm_.show()[0]}")
    # print(f"Frequency-tuned meanFm: {cal_fm_.show()[1]}")
    # print(f"Precision: {cal_fm_.show()[2]}")
    # print(f"Recall: {cal_fm_.show()[3]}")
    print(f"Test Loss: {np.mean(test_loss)}")    
    print(f"Dice Score: {cal_dice_.show()}")
    print(f"IoU Score: {cal_iou_.show()}")    
    print(f"Weighted F-beta Score: {cal_wfm_.show()}")    
    print(f"Structure Measure: {cal_sm_.show()}")    
    print(f"Balanced Error Rate: {cal_ber_.show()}")
    print(f"Enhanced Alignment Score: {cal_em_.show()}")
    print(f"Mean Absolute Error: {cal_mae_.show()}")    
    print(f"Accuracy: {cal_acc_.show()}")
    print("----")
        
    #return test_loss, dice_score, iou_score, f_e_score, mae_score , structure_measure_iist, weighted_dice


