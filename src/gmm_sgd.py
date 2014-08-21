from __future__ import division
import numpy as np
from _gmm_sgd import lsvm_inference, compute_mean_gradient

def gmm_estimation(data_file,means,log_covs,weights,class_ids, class_id_to_component_start_idx,beta,step_alpha=100,mini_batch_size=1000,init_alpha=1.0,n_iterations=1000,random_seed=0):
    """
    """
    np.random.seed(randome_seed)
    n_data,n_features = data_file['train_data'].shape
    n_components, n_features2 = means.shape
    assert n_features == n_features2
   
    batch_starts_ends = np.zeros((n_iterations,2),dtype=np.int64)
    n_batch_starts = n_data // mini_batch_size
    if n_data % mini_batch_size == 0:
        n_batch_starts -= 1

    t = np.arange(n_iterations,dtype=np.int32) % n_batch_starts
    np.random.shuffle(t)
    batch_starts_ends[:,0] = t
    batch_starts_ends[:,1] = np.minimum(batch_starts_ends[:,0] + mini_batch_size,
                                        n_data)
    

    mean_gradients = np.zeros(means.shape,dtype=np.float32)
    log_cov_gradients = np.zeros(log_covs.shape,dtype=np.float32)
    
    
    for iteration_id, (batch_start, batch_end) in enumerate(batch_starts_ends):
        X = data_file['train_data'][batch_start: batch_end]
        y = data_file['train_labels'][batch_start: batch_end]
        
        hinge_loss, weight_loss_gradients, mean_gradients, log_cov_gradients = gmm_gradients(X,y,means,np.exp(-log_covs), weights,class_ids,class_id_to_component_start_idx, beta, mean_gradients=mean_gradients, log_cov_gradients=log_cov_gradients)


        step_size = 1.0/(step_alpha*(init_alpha + iteration_id))
        gmm_gradient_step(weights, means, log_covs, weight_loss_gradients, mean_gradients, log_cov_gradients, step_size)
        print "Iteration {0}: hinge_loss={1}".format(iteration_id,hinge_loss)

    return means, log_covs, weights
    

def gmm_hinge_loss(X,y,means,inv_covs, weights,class_ids,class_id_to_component_start_idx, beta):
    """
    """
    data_mean_diff = X[:,np.newaxis,:] - means
    sq_data_mean_diff = data_mean_diff**2
    likes = .5 * (sq_data_mean_diff * inv_covs - np.log(inv_covs)).sum(-1) + weights
    max_likes, max_like_components = lsvm_inference(likes,y,class_ids,class_id_to_components_start_idx)
    return np.minimum(max_likes[:,1] - max_likes[:,0] + beta,0).sum()
    

def gmm_gradients(X,y,means,inv_covs, weights,class_ids,class_id_to_component_start_idx, beta, mean_gradients=None, log_cov_gradients=None):
    """
    """
    data_mean_diff = X[:,np.newaxis,:] - means
    sq_data_mean_diff = data_mean_diff**2
    likes = .5 * (sq_data_mean_diff * inv_covs - np.log(inv_covs)).sum(-1) + weights
    max_likes, max_like_components = lsvm_inference(likes,y,class_ids,class_id_to_components_start_idx)
    hinge_data_idx = (max_likes[:,1] - max_likes[:,0] + beta > 0).astype(np.int32)
    hinge_loss = np.minimum(max_likes[:,1] - max_likes[:,0] + beta,0).sum()

    n_components = means.shape[0]
    n_features = means.shape[1]
    true_component_counts = np.bincount(max_like_components[:,0][hinge_data_idx],minlength=n_components)
    false_component_counts = np.bincount(max_like_components[:,1][hinge_data_idx],minlength=n_components)
    
    weight_loss_gradients = false_component_counts - true_component_counts
    
    data_mean_diff *= inv_covs
    if mean_gradients is None:
        mean_gradients = np.zeros((n_components,n_features),dtype=np.float32)
    else:
        mean_gradients[:] = 0.
    compute_mean_gradient(hinge_data_idx, mean_gradients, data_mean_diff,max_like_components)
    
    sq_data_mean_diff *= inv_covs
    sq_data_mean_diff -= 1.0
    sq_data_mean_diff /= 2.0
    if log_cov_gradients is None:
        log_cov_gradients = np.zeros((n_components,n_features),dtype=np.float32)
    else:
        log_cov_gradients[:] = 0.
    compute_mean_gradient(hinge_data_idx, log_cov_gradients,
                          sq_data_mean_diff,max_like_components)
    
    return hinge_loss, weight_loss_gradients, mean_gradients, log_cov_gradients

    
def gmm_gradient_step(weights, means, log_covs, weight_loss_gradients, mean_gradients, log_cov_gradients, step_size):
    weights -= weight_loss_gradient * step_size
    means -= mean_gradients* step_size
    log_covs -= log_cov_graidents * step_size
    

    
    
    
    
    
    

