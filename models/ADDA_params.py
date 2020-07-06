"""Params for ADDA."""

tIndex = "tn"

# params for dataset and data loader
dataset_mean_value = 0.5
dataset_std_value = 0.5
dataset_mean = (dataset_mean_value, dataset_mean_value, dataset_mean_value)
dataset_std = (dataset_std_value, dataset_std_value, dataset_std_value)
#20
#batch_size = 100

# params for source dataset
src_encoder_restore = "source-encoder-final.pt"
src_classifier_restore = "source-classifier-final.pt"
src_regressor_restore = "source-regressor-final.pt"
src_model_trained = True

e_input_dims = 510
e_hidden_dims = 200
e_output_dims = 60

r_input_dims = 60

# params for target dataset
tgt_encoder_restore = "target-encoder-final.pt"
tgt_model_trained = True

# params for setting up models
model_root = "ADDA-snapshots"
d_input_dims = 60
d_hidden_dims = 100
d_output_dims = 2
d_model_restore = "critic-final.pt"


# params for training network
num_gpu = 1
#100
num_epochs_pre = 5
log_step_pre = 20
eval_step_pre = 20
save_step_pre = 100
#2000
num_epochs = 5
log_step = 100
save_step = 100
manual_seed = 10

# params for optimizing models
d_learning_rate = 1e-2
c_learning_rate = 1e-2
g_learning_rate = 1e-3
beta1 = 0.5
beta2 = 0.9
