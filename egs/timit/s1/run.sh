dir=`pwd`
conf=$dir/conf
local=$dir/local
exp=$dir/exp

mkdir -p $conf $local $exp

# convert the phones to a 48 map


$local/prep_classification_data.py --phns 
