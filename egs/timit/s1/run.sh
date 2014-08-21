dir=`pwd`
conf=$dir/conf
local=$dir/local
exp=$dir/exp
data=$dir/../../../Data/timit_melspec

mkdir -p $conf $local $exp

# convert the phones to a 48 map
phns=`awk '{ print $2 }' $conf/phones.60-48-39.map | sort | uniq | egrep  "^[a-z]+$"`


python $local/prep_classification_data.py --phns $phns --fl_prefix $data --fl_suffix melspec_features_no_deltas --hdf5_file $exp/ellis_melspec_no_deltas_train.hdf5 --class_indices_path $exp/ellis_melspec_class_indices_train.npy --dset train

python $local/prep_classification_data.py --phns $phns --fl_prefix $data --fl_suffix melspec_features_no_deltas --hdf5_file $exp/ellis_melspec_no_deltas_dev.hdf5 --class_indices_path $exp/ellis_melspec_class_indices_dev.npy --dset dev

# move models onto this computer
models_directory=/var/tmp/stoehr/egs/timit/s4/exp
models_tag=melspec_GMM_no_deltas
for phn in $phns ; do
    echo $phn
    scp stoehr@sharpshined.cs.uchicago.edu:${models_directory}/${phn}_${models_tag}_5C_* $exp
done


python $local/train_gaussian_sgd_models.py 
