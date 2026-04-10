

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def auroc(dist_id, dist_ood, return_tpr_fpr=False):
    start = dist_id.min()
    end = dist_id.max()
    
    start = np.min(np.array([start, dist_ood.min()]))
    end = np.max(np.array([end, dist_ood.max()]))
    
    thresholds = np.arange(start, end, (end - start)/1000)
    thresholds = thresholds[1:thresholds.shape[0]-1]
    
    tpr_fpr_vals = []
    
    for threshold in thresholds:
        if np.mean(dist_ood)>np.mean(dist_id):
            ID_tpr = np.sum(dist_id<threshold)/len(dist_id)
            OOD_fpr = np.sum(dist_ood<threshold)/len(dist_ood)
            
        else:
            ID_tpr = np.sum(dist_id>threshold)/len(dist_id)
            OOD_fpr = np.sum(dist_ood>threshold)/len(dist_ood)
        
        # print(f"Threshold : {threshold}\t, OOD TPR : {OOD_tpr}\t ID_FPR : {ID_fpr}")
        
        tpr_fpr_vals.append((ID_tpr, OOD_fpr))
    
    
    AUROC = 0
    tpr_fpr_vals = tpr_fpr_vals[::-1]
    for i in range(len(tpr_fpr_vals)-1):
        AUROC += np.abs(1.0 * ( tpr_fpr_vals[i][0]) * (tpr_fpr_vals[i+1][1] - tpr_fpr_vals[i][1]))
        
    if return_tpr_fpr:
        return AUROC, tpr_fpr_vals
    else:
        return AUROC

def fpr_x(dist_id, dist_ood, tpr_val=95):
    if tpr_val>1:
        tpr_val = tpr_val/100
    
    if np.mean(dist_ood)>np.mean(dist_id):
        
        threshold = np.quantile(dist_id, tpr_val)
        
        ID_tpr = np.sum(dist_id<threshold)/len(dist_id)
        OOD_fpr = np.sum(dist_ood<threshold)/len(dist_ood)
        
    else:
        
        threshold = np.quantile(dist_id, 1.0 - tpr_val)
        
        __ = np.sum(dist_id>threshold)/len(dist_id)
        OOD_fpr = np.sum(dist_ood>threshold)/len(dist_ood)
        
    return OOD_fpr

def aupr(dist_id, dist_ood, return_pr_vals=False):
    start = dist_id.min()
    end = dist_id.max()
    
    
    start = np.min(np.array([start, dist_ood.min()]))
    end = np.max(np.array([end, dist_ood.max()]))
    
    # temp = np.concatenate((dist_id.ravel(), dist_ood.ravel()))
    # thresh_q = np.arange(0.01, 1.0, 0.01)
    thresholds = np.arange(start, end, (end - start)/1000)
    thresholds = thresholds[1:thresholds.shape[0]-1]
    
    pr_vals = []
    
    for threshold in thresholds:
        if np.mean(dist_ood)>np.mean(dist_id):
            precision_val = np.sum(dist_id<threshold)/(np.sum(dist_id<threshold) + np.sum(dist_ood<threshold))
            recall_val = np.sum(dist_id<threshold) / len(dist_id)
            
        else:
            recall_val = np.sum(dist_id>threshold)/len(dist_id)
            precision_val = np.sum(dist_id>threshold)/(np.sum(dist_id>threshold) + np.sum(dist_ood>threshold))
        
        # print(f"Threshold : {threshold}\t, OOD TPR : {OOD_tpr}\t ID_FPR : {ID_fpr}")
        
        pr_vals.append((precision_val, recall_val))
        
    AUPR = 0
    pr_vals = pr_vals[::-1]
    for i in range(len(pr_vals)-1):
        AUPR += np.abs(1.0 * ( pr_vals[i][0]) * (pr_vals[i+1][1] - pr_vals[i][1]))
        
    if return_pr_vals:
        return AUPR, pr_vals
    else:
        return AUPR
    
    
    