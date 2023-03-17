conda activate chemprop
python interpret.py --data_path bbb_test_full.csv --checkpoint_path /home/ymyung/projects/deeppk/3_Results/All_Random_300EPOCH/bbb/fold_0/model_0/model.pt --property_id 1 --no_features_scaling --features_generator rdkit_2d_normalized

