from __future__ import division
import numpy as np
import argparse
import h5py


class DataIndices(object):
    def __init__(self):
        self.within_class_indices = np.array([],dtype=np.int32)
        self.class_indices = np.array([],dtype=np.int32)
        self.cur_n_data = 0
        self.cur_allocated_data = 0

    def add_indices(self,class_id,n_data):
        if n_data + self.cur_n_data > self.cur_allocated_data:
            new_cur_allocated_data = max(2*self.cur_allocated_data,
                                         n_data+self.cur_n_data)
            new_within_class_indices = np.empty(new_cur_allocated_data,dtype=np.int32)
            new_class_indices = np.empty(new_cur_allocated_data,dtype=np.int32)
            new_within_class_indices[:self.cur_n_data] = self.within_class_indices[:self.cur_n_data]
            new_class_indices[:self.cur_n_data] = self.class_indices[:self.cur_n_data]
            self.within_class_indices = new_within_class_indices
            self.class_indices = new_class_indices
            self.cur_allocated_data = new_cur_allocated_data


        self.class_indices[self.cur_n_data:self.cur_n_data + n_data] = class_id
        self.within_class_indices[self.cur_n_data:self.cur_n_data + n_data] = np.arange(n_data,dtype=np.int32)
        self.cur_n_data += n_data


    def collapse_allocation(self):
        self.cur_allocated_data = self.cur_n_data
        self.class_indices = self.class_indices[:self.cur_allocated_data]
        self.within_class_indices = self.within_class_indices[:self.cur_allocated_data]

    def permute_indices(self):
        self.perm = np.random.permutation(self.cur_n_data)
        self.class_indices[:self.cur_n_data] = self.class_indices[self.perm]
        self.within_class_indices[:self.cur_n_data] = self.within_class_indices[self.perm]
                                  

parser = argparse.ArgumentParser("""Process phones and load in data
""")
parser.add_argument('--phns',type=str,nargs='+',help='list of phones to process')
parser.add_argument('--fl_prefix',type=str)
parser.add_argument('--fl_suffix',type=str)
parser.add_argument('--hdf5_file',type=str,help='file to save to')
parser.add_argument('--class_indices_path',type=str,help='file to save the file indices to')
parser.add_argument('--dset',type=str,help='dataset we are working on')
args = parser.parse_args()

# get the length of the file
# read in the file in random order and blocks
# save to the HDF5 format so that I can implement the stochastic gradient descent

phn_data_indices = DataIndices()

for phn_id, phn in enumerate(args.phns):
    print phn_id, phn
    fpath = '{0}/{1}_{2}'.format(args.fl_prefix,phn,args.fl_suffix)
    X = np.loadtxt('{0}_{1}.dat'.format(fpath,args.dset)).T
    n_data, n_features = X.shape
    phn_data_indices.add_indices(phn_id,n_data)

phn_data_indices.collapse_allocation()
f = h5py.File(args.hdf5_file,'w')
n_data = phn_data_indices.cur_n_data
chunksize = 1000
n_chunks = n_data // chunksize + (n_data % chunksize > 0)
n_rows=n_chunks*chunksize

dset = f.create_dataset("{0}_data".format(args.dset),
    (n_data,
     n_features),dtype=np.float32)

dset_labels = f.create_dataset("{0}_labels".format(args.dset),
    (n_data,
     ),dtype=np.int32)

np.random.seed(0)
phn_data_indices.permute_indices()
perm = np.random.permutation(n_data)

for phn_id, phn in enumerate(args.phns):
    print phn_id, phn
    fpath = '{0}/{1}_{2}'.format(args.fl_prefix,phn,args.fl_suffix)
    X = np.loadtxt('{0}_{1}.dat'.format(fpath,args.dset)).T
    phn_indices = np.where(phn_data_indices.class_indices == phn_id)[0]
    if len(phn_indices) != len(X):
        import pdb; pdb.set_trace()

    for example_idx_id, example_idx in enumerate(phn_indices):
        dset[example_idx,:] = X[example_idx_id,:]
        dset_labels[example_idx] = phn_id
    

f.flush()
np.save(args.class_indices_path,phn_data_indices.class_indices)
                        

