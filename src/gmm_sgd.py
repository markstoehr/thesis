from __future__ import division
import numpy as np
from _gmm_sgd import lsvm_inference

def gmm_likelihoods(X,y,means,inv_covs, weights,class_ids,class_id_to_component_start_idx, beta,step, mean_gradients, cov_gradients):
    """
    """
    sq_data_mean_diff = (2*X[:,np.newaxis,:] * means 
                            - X[:,np.newaxis,:]**2 - means**2) 
    likes = .5 * (sq_data_mean_diff * inv_covs - np.log(inv_covs)).sum(-1) + weights
    max_likes, max_like_components = lsvm_inference(likes,y,class_ids,class_id_to_components_start_idx)
    hinge_data_idx = np.where(max_likes[:,1] - max_likes[:,0] + beta > 0)[0]

    n_components = means.shape[0]
    true_component_counts = np.bincount(max_like_components[:,0][hinge_data_idx],minlength=n_components)
    false_component_counts = np.bincount(max_like_components[:,1][hinge_data_idx],minlength=n_components)
    
    weight_loss_gradients = false_component_counts - true_component_counts
    weights -= step*weight_loss_gradients
    
    data_mean_diff = (X[:,np.newaxis,:] - means) * inv_covs
    
    
    
    
    

