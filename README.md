# retrotemp


Neural network for predicting template relevance a la Segler and Waller's Neural Symbolic paper. 


### Dependencies if you want to use the final model
- RDKit (most versions should be fine)
- numpy

### Dpendencies if you want to retrain on your own data
- RDKit (most versions should be fine)
- tensorflow (r0.12.0)
- h5py
- numpy

### About
Learn to predict template relevance. 

1. Grab reaction precedents from templates stored in MongoDB
    ```python scripts/get_reaxys_data.py```
1. Calculate fingerprints and store in .h5 file
    ```python scripts/make_data_file.py data/reaxys_limit1000000000_reaxys_v2_transforms_retro_v9_10_5.txt 2048```
1. Train model
    ```python retrotemp/nntrain_fingerprint.py -t data/reaxys_limit1000000000_reaxys_v2_transforms_retro_v9_10_5.txt -o 163723 -m models/6d3M_Reaxys_10_5 --fp_len 2048 ```
1. Find best validation performance
    ```
    regex="model\.(.*)\.meta"
    for f in `ls -tr models/6d3M_Reaxys_10_5/*.meta`
    do
        if [[ $f =~ $regex ]]
        then
            ckpt="${BASH_REMATCH[1]}"
            echo $ckpt
            python retrotemp/nntrain_fingerprint.py  -o 163723 -m models/6d3M_Reaxys_10_5 --fp_len 2048 -c "$ckpt" -t data/reaxys_limit1000000000_reaxys_v2_transforms_retro_v9_10_5.txt --test valid
        fi
    done
    ```
1. Retrain on whole dataset (?) for same number of epochs
    ```python retrotemp/nntrain_fingerprint.py -t data/reaxys_limit1000000000_reaxys_v2_transforms_retro_v9_10_5.txt -o 163723 -m models/6d3M_Reaxys_10_5 --fp_len 2048 --fixed_epochs_train_all 15```


1. Run standalone tensorflow version to dump to numpy arrays
