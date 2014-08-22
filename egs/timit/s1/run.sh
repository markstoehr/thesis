dir=`pwd`
conf=$dir/conf
local=$dir/local
exp=$dir/exp
data=$dir/../../../Data/timit_melspec
results=$dir/results

mkdir -p $conf $local $exp $results


# convert the phones to a 48 map
phns=`awk '{ print $2 }' $conf/phones.60-48-39.map | sort | uniq | egrep  "^[a-z]+$"`
awk '{  if ($2 != "") print $2 " " $3 }' $conf/phones.60-48-39.map | sort | uniq > $conf/phn_map39


python $local/prep_classification_data.py --phns $phns --fl_prefix $data --fl_suffix melspec_features_no_deltas --hdf5_file $exp/ellis_melspec_no_deltas_train.hdf5 --class_indices_path $exp/ellis_melspec_class_indices_train.npy --dset train

python $local/prep_classification_data.py --phns $phns --fl_prefix $data --fl_suffix melspec_features_no_deltas --hdf5_file $exp/ellis_melspec_no_deltas_dev.hdf5 --class_indices_path $exp/ellis_melspec_class_indices_dev.npy --dset dev

# move models onto this computer
models_directory=/var/tmp/stoehr/egs/timit/s4/exp
models_tag=melspec_GMM_no_deltas
rm -f $exp/phn_model_paths
for phn in $phns ; do
    echo $phn
    scp stoehr@sharpshined.cs.uchicago.edu:${models_directory}/${phn}_${models_tag}_5C_* $exp
    echo $phn $exp/${phn}_${models_tag}_5C >> $exp/phn_model_paths
done

python $local/convert_mix_to_gaussian_sgd_model.py --phns $phns \
    --phn_model_paths $exp/phn_model_paths \
    --save_path $exp \
    --save_tag 5C_melspec_untrained

python $local/test_gaussian_sgd_model.py --phn_map $conf/phn_map39 \
    --hdf5_data $exp/ellis_melspec_no_deltas_dev.hdf5 \
    --mini_batch_size 1000 \
    --model_path $exp \
    --model_tag 5C_melspec_untrained \
    --save_path $results \
    --save_tag 5C_melspec_untrained_dev \
    --dset dev



python $local/train_gaussian_sgd_model.py --phns $phns \
    --phn_model_paths $exp/phn_model_paths \
    --hdf5_data $exp/ellis_melspec_no_deltas_train.hdf5 \
    --beta 1.0 \
    --step_alpha 100 \
    --mini_batch_size 1000 \
    --init_alpha 1 \
    --n_iterations 1000 \
    --random_seed 0 \
    --save_path $exp \
    --save_tag beta1_mb1000_sa100_niter1k_melspec

python $local/train_gaussian_sgd_model.py --phns $phns \
    --phn_model_paths $exp/phn_model_paths \
    --hdf5_data $exp/ellis_melspec_no_deltas_train.hdf5 \
    --beta 1.0 \
    --step_alpha 100 \
    --mini_batch_size 2000 \
    --init_alpha 1 \
    --n_iterations 1000 \
    --random_seed 0 \
    --save_path $exp \
    --save_tag beta1_mb2000_sa100_niter1k_melspec

python $local/test_gaussian_sgd_model.py --phn_map $conf/phn_map39 \
    --hdf5_data $exp/ellis_melspec_no_deltas_dev.hdf5 \
    --mini_batch_size 1000 \
    --model_path $exp \
    --model_tag beta1_mb1000_sa100_niter1k_melspec \
    --save_path $results \
    --save_tag beta1_mb1000_sa100_niter1k_melspec \
    --dset dev

python $local/test_gaussian_sgd_model.py --phn_map $conf/phn_map39 \
    --hdf5_data $exp/ellis_melspec_no_deltas_train.hdf5 \
    --mini_batch_size 1000 \
    --model_path $exp \
    --model_tag beta1_mb1000_sa100_niter1k_melspec \
    --save_path $results \
    --save_tag beta1_mb1000_sa100_niter1k_melspec_train \
    --dset train

