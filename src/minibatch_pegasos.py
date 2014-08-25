from __future__ import division
import numpy as np
import h5py
from _gmm_sgd import lsvm_inference, compute_mean_gradient, update_confusion_matrix, compute_weights_gradient


def pegasos_estimation(data_file,weights,constants,class_ids,class_id_to_component_start_idx,
                       beta,lambda_param,step_alpha=100,mini_batch_size=1000,
                       init_alpha=1,n_iterations=1000,random_seed=0):
    """
    data is n_data x n_features
    weights is n_features x n_components

    this is to do fast score computation
    """
    np.random.seed(random_seed)
    n_data,n_features = data_file['train_data'].shape
    n_features2, components = weights.shape
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
    
    

    weights_gradient = np.zeros(weights.T.shape,dtype=np.float32)
    constants_gradient = np.zeros(constants.shape,dtype=np.float32)
    
    import pdb; pdb.set_trace()
    for iteration_id, (batch_start, batch_end) in enumerate(batch_starts_ends):
        X = data_file['train_data'][batch_start: batch_end]
        y = data_file['train_labels'][batch_start: batch_end]

        # log_covs = np.maximum(log_covs,np.log(.1))

        try:
            hloss, weights_gradient, constants_gradient = hinge_gradient(X,y,weights,constants,class_ids,class_id_to_component_start_idx,beta,lambda_param,
                   weights_gradient, constants_gradient)
        except:
            import pdb; pdb.set_trace()

        print weights[:5,:5]
        step_size = 1.0/(step_alpha*(init_alpha + iteration_id))
        hinge_gradient_step(weights,weights_gradient,constants, constants_gradient, step_size)
        print weights[:5,:5]
        print "Iteration {0}: hinge_loss={1}".format(iteration_id,hloss)

    return weights, constants


def pegasos_confusion_matrix(data_file,dset,weights,constants,class_ids, class_id_to_component_start_idx,mini_batch_size=1000):
    """
    how many 0-1 loss is accumulated for each type of mistake
    confusion matrix is each row corresponds to the underlying true class
    and the columns correspond to the class it was classified as
    so if an example p is of underlying class i and was classified
    as j then
    conf_mat[i,j] += 1
    """
    dset_data = '{0}_data'.format(dset)
    dset_labels = '{0}_labels'.format(dset)
    n_data,n_features = data_file[dset_data].shape
    n_features2, components = weights.shape
    assert n_features == n_features2
   
    n_classes = class_ids.max() + 1

    conf_mat = np.zeros((n_classes,n_classes),dtype=np.float32)

    n_batches = (n_data -1) // mini_batch_size + 1
    
    # only count mistakes -- no margin
    beta=0
    
    for batch_id in xrange(n_batches):
        batch_start = batch_id * mini_batch_size
        batch_end = min(batch_start + mini_batch_size,
                        n_data)
        print "batch_start={0}, batch_end={1}".format(batch_start,batch_end)
        X = data_file[dset_data][batch_start: batch_end]
        y = data_file[dset_labels][batch_start: batch_end]
        
        cur_loss, max_likes, max_like_components = hinge_loss(X,y,weights,constants,class_ids,class_id_to_component_start_idx,beta, return_likes_components=True)

        update_confusion_matrix(conf_mat,
                                class_ids,
                                max_likes,
                                max_like_components)
        
    return conf_mat



def hinge_loss(X,y,weights,constants,class_ids,class_id_to_component_start_idx,beta, return_likes_components=False, hinge_sum=True):
    scores = np.dot(X,weights) + constants
    max_likes, max_like_components = lsvm_inference(scores,y,class_ids,class_id_to_component_start_idx)

    if hinge_sum:
        if return_likes_components:
            return np.maximum(max_likes[:,1] - max_likes[:,0] + beta,0).sum(), max_likes, max_like_components
        else:
            return np.maximum(max_likes[:,1] - max_likes[:,0] + beta,0).sum()
    else:
        if return_likes_components:
            return np.maximum(max_likes[:,1] - max_likes[:,0] + beta,0), max_likes, max_like_components
        else:
            return np.maximum(max_likes[:,1] - max_likes[:,0] + beta,0)

def hinge_gradient(X,y,weights,constants,class_ids,class_id_to_component_start_idx,beta,lambda_param,
                   weights_gradient, constants_gradient):
    """
    compute the loss and the hinge loss
    """
    hloss, max_likes, max_like_components = hinge_loss(X,y,weights,constants,class_ids,
                                                            class_id_to_component_start_idx,beta, return_likes_components=True,hinge_sum=False)
    
    hinge_data_idx = (hloss > 0).astype(np.int32)

    weights_gradient[:] = lambda_param*weights.T
    compute_weights_gradient(hinge_data_idx, 
                             X,
                             weights_gradient,
                             max_like_components)
    
    n_components = weights.shape[1]
    n_data = X.shape[0]
    add_constant_ones = np.bincount(max_like_components[:,1][hinge_data_idx],minlength=n_components)
    negative_constant_ones = np.bincount(max_like_components[:,0][hinge_data_idx],minlength=n_components)
    constants_gradient[:] = add_constant_ones/n_data - negative_constant_ones/n_data + lambda_param * constants
    return hloss.sum(), weights_gradient, constants_gradient


def hinge_gradient_step(weights,weights_gradient,constants, constants_gradient, step_size):
    weights -=  step_size*weights_gradient.T
    constants -= step_size*constants
    
    
    
    
