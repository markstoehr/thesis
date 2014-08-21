# cython: profile=False, boundscheck=False, wraparound=False, cdivision=True, 

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Lars Buitinck <larsmans@gmail.com>
#
# Licence: BSD 3 clause

from libc.math cimport sqrt
import numpy as np
cimport numpy as np
cimport cython

ctypedef np.float64_t DOUBLE_t
ctypedef np.float32_t SINGLE_t
ctypedef np.int32_t INT_t
SINGLE = np.float32
DOUBLE = np.float64


np.import_array()

cdef extern from "cblas.h":
    double ddot "cblas_ddot"(int, double *, int, double *, int) nogil
    void dscal "cblas_dscal"(int, double, double *, int) nogil
    void saxpy "cblas_saxpy"(int, float,float *,
                 int, float *, int) nogil


def lsvm_inference(np.ndarray[SINGLE_t,ndim=2] likes, 
                   np.ndarray[INT_t,ndim=1] y, 
                   np.ndarray[INT_t,ndim=1] class_ids,
                   np.ndarray[INT_t,ndim=1] class_id_to_component_start_idx):
    """
    
    """
    cdef:
        unsigned int data_id, component_id, max_true_component_id, max_false_component_id
        unsigned int n_data = y.shape[0]
        unsigned int n_components = class_ids.shape[0]
        DOUBLE_t max_true_score, max_false_score
        np.ndarray[SINGLE_t, ndim=2] max_likes = np.empty( 
            (n_data,2),dtype=np.float32)
        np.ndarray[INT_t,ndim=2] max_like_components = np.empty(
            (n_data,2),dtype=np.int32)

    for data_id in range(n_data):
        max_true_component_id = class_id_to_component_start_idx[y[data_id]]
        if y[data_id] == 0:
            max_false_component_id = class_id_to_component_start_idx[1]
        else:
            max_false_component_id = class_id_to_component_start_idx[0]
            
        max_true_score = likes[data_id,max_true_component_id]
        max_false_score = likes[data_id, max_false_component_id]
        
        for component_id in range(n_components):
            if class_ids[component_id] == y[data_id]:
                if likes[data_id,component_id] > max_true_score:
                    max_true_score = likes[data_id,component_id]
                    max_true_component_id = component_id
            else:
                if likes[data_id,component_id] > max_false_score:
                    max_false_score = likes[data_id,component_id]
                    max_false_component_id = component_id

        max_likes[data_id,0] = max_true_score
        max_likes[data_id,1] = max_false_score

        max_like_components[data_id,0] = max_true_component_id
        max_like_components[data_id,1] = max_false_component_id

    return max_likes, max_like_components

def test_saxpy_subtract(np.ndarray[SINGLE_t,ndim=2] mean_gradients,
                        np.ndarray[SINGLE_t,ndim=2] X,
                        np.ndarray[INT_t,ndim=1] max_like_components):
    cdef:
        unsigned int data_id
        unsigned int n_data = X.shape[0]
        unsigned int n_features = X.shape[1]
        unsigned int n_components = mean_gradients.shape[0]
        float *mean_grad_ptr = <float *>mean_gradients.data
        float *X_ptr = <float *>X.data

    for data_id in range(n_data):
        saxpy(n_features, -1.0, X_ptr,1,
                  <float *>(mean_grad_ptr + max_like_components[data_id]*n_features),1)

        if data_id < n_data -1:
            X_ptr += n_features
    
def compute_mean_gradient(np.ndarray[INT_t,ndim=1] hinge_data_idx, 
                          np.ndarray[SINGLE_t,ndim=2] mean_gradients,
                          np.ndarray[SINGLE_t,ndim=3] data_mean_diff,
                          np.ndarray[INT_t,ndim=2] max_like_components):
    """
    """
    cdef:
        unsigned int data_id, true_component_id, false_component_id
        unsigned int n_data = data_mean_diff.shape[0]
        unsigned int n_features = data_mean_diff.shape[2]
        unsigned int n_components = mean_gradients.shape[0]
        unsigned int n_component_features = n_features * n_components
        float *mean_grad_ptr = <float *>mean_gradients.data
        float *data_diff_ptr = <float *>data_mean_diff.data

    for data_id in range(n_data):
        if hinge_data_idx[data_id] > 0: # check whether there is a margin violation
            saxpy(n_features, -1.0, <float *>(data_diff_ptr + max_like_components[data_id,0]*n_features) ,1,
                  <float *>(mean_grad_ptr + max_like_components[data_id,0]*n_features),1)
            saxpy(n_features, 1.0, <float *>(data_diff_ptr + max_like_components[data_id,1]*n_features),1,
                  <float *>(mean_grad_ptr + max_like_components[data_id,1]*n_features),1)

        if data_id < n_data -1:
            data_diff_ptr += n_component_features
    
