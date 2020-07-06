"""Discriminator model for ADDA."""
import os
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time

from . import ADDA_params as params
from .adda_models.utils import *

from .adda_models.MPL import *
from .adda_models.ConvLSTM import *

from . import ModelUtils

#################

def train_tgt_mlp(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=params.g_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    # optimizer_critic = optim.SGD(critic.parameters(),
    #                               lr=params.d_learning_rate)
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((datas_src, _), (datas_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            datas_src = make_variable(datas_src)
            datas_tgt = make_variable(datas_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(datas_src)
            feat_tgt = tgt_encoder(datas_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())
            #print("pred_concat.shape:{}".format(pred_concat.shape))

            # prepare real and fake label
            label_src = make_variable(torch.ones(feat_src.size(0)).long())
            label_tgt = make_variable(torch.zeros(feat_tgt.size(0)).long())
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(datas_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = make_variable(torch.ones(feat_tgt.size(0)).long())

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################

            #print("step:{} params.log_step:{}".format(step,params.log_step))

            if ((step + 1) % params.log_step == 0):
                print("train_tgt_mlp Epoch [{}/{}] Step [{}/{}]:"
                        "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                        .format(epoch + 1,
                                params.num_epochs,
                                step + 1,
                                len_data_loader,
                                loss_critic.item(),
                                loss_tgt.item(),
                                acc.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % params.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                params.model_root,
                params.tIndex+"critic-{}.pt".format(epoch + 1)))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                params.model_root,
                params.tIndex+"target-encoder-{}.pt".format(epoch + 1)))

    torch.save(critic.state_dict(), os.path.join(
        params.model_root,
        params.tIndex+params.d_model_restore))
    torch.save(tgt_encoder.state_dict(), os.path.join(
        params.model_root,
        params.tIndex+params.tgt_encoder_restore))

    print("=== Training encoder for target domain === END!!!")
    return tgt_encoder

#####################################


def train_src_mlp(encoder, classifier, data_loader):
    """Train classifier for source domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    encoder.train()
    classifier.train()

    # setup criterion and optimizer
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=params.c_learning_rate,
        betas=(params.beta1, params.beta2))
    criterion = nn.MSELoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (datas, labels) in enumerate(data_loader):
            # make images and labels variable
            datas = make_variable(datas)
            labels = make_variable(labels[:,np.newaxis])

            #print("train_src_mlp labels.shape:{}".format(labels.shape))

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(datas))
            #print("[1]labels size{}".format(labels.size()))
            loss = criterion(preds, labels)

            # optimize source classifier
            loss.backward()
            optimizer.step()

            # print step info
            if ((step + 1) % params.log_step_pre == 0):
                print("Epoch [{}/{}] Step [{}/{}]: loss={}"
                      .format(epoch + 1,
                              params.num_epochs_pre,
                              step + 1,
                              len(data_loader),
                              loss.item()))

        # eval model on test set
        if ((epoch + 1) % params.eval_step_pre == 0):
            eval_src_mlp(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, params.tIndex+"source-encoder-{}.pt".format(epoch + 1),params)
            save_model(
                classifier, params.tIndex+"source-classifier-{}.pt".format(epoch + 1),params)

    # # save final model
    save_model(encoder, params.tIndex+params.src_encoder_restore,params)
    save_model(classifier, params.tIndex+params.src_classifier_restore,params)

    return encoder, classifier


def eval_src_mlp(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

   # init loss and accuracy
    loss = 0
    mae_loss = 0

    # set loss function
    criterion = nn.MSELoss()
    mae_crit1 = nn.L1Loss()

    # evaluate network
    for (datas, labels) in data_loader:
        datas = make_variable(datas, volatile=True)
        labels = make_variable(labels[:,np.newaxis])
        #print("eval_src_mlp labels.shape:{}".format(labels.shape))

        preds = classifier(encoder(datas))
        #print("[2]labels size{}".format(labels.size()))
        loss += criterion(preds, labels).item()
        mae_loss += mae_crit1(preds, labels).item()

    loss /= len(data_loader)
    loss = loss**(0.5)
    mae_loss /= len(data_loader)

    print("RMSE = {}, MAE = {}".format(loss, mae_loss))


#########

def eval_tgt_mlp(encoder, classifier, tgt_data_loader,y_train):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    mae_loss = 0
    rmse_loss = 0
    mape_loss = 0
    mase  = 0

    # set loss function
    criterion = nn.MSELoss()
    mae_crit1 = nn.L1Loss()
    rmse_criterion = RMSELoss()

    # evaluate network
    for (datas, labels) in tgt_data_loader:
        datas = make_variable(datas, volatile=True)
        labels = make_variable(labels[:,np.newaxis])

        #print("datas.shape:"+str(datas.shape))
        #print("labels.shape:"+str(labels.shape))

        preds = classifier(encoder(datas))

        loss += criterion(preds, labels).item()
        mae = mae_crit1(preds, labels).item()
        mae_loss += mae
        
        #print("mae:{}".format(mae))

        rmse_loss += rmse_criterion(preds, labels).item()
        mape_loss += MAPELoss(preds, labels)

        y_true = labels.cpu().detach().numpy()
        y_pred = preds.cpu().detach().numpy()
        
        mase += ModelUtils.mean_absolut_scaled_error(y_train,y_true,y_pred)


    loss /= len(tgt_data_loader)
    #mae_loss = loss
    loss = loss**(0.5)

    print("mae_loss:{}".format(mae_loss))

    mae_loss /= len(tgt_data_loader)
    rmse_loss /= len(tgt_data_loader)
    mape_loss /= len(tgt_data_loader)
    mase /= len(tgt_data_loader)

    print("RMSE = {}, MAE = {} MAPE = {} MASE = {}".format(rmse_loss, mae_loss, mape_loss, mase))
    return rmse_loss, mae_loss, mape_loss, mase

def updateParams(parameters):
    if "dataset_mean_value" in parameters: 
        params.dataset_mean_value = parameters["dataset_mean_value"]
        params.dataset_mean = (params.dataset_mean_value, params.dataset_mean_value, params.dataset_mean_value)
        
    if "dataset_std_value" in parameters: 
        params.dataset_std_value = parameters["dataset_std_value"]
        params.dataset_std = (params.dataset_std_value, params.dataset_std_value, params.dataset_std_value)

    if "src_encoder_restore" in parameters: params.src_encoder_restore = parameters["src_encoder_restore"]
    if "src_regressor_restore" in parameters: params.src_regressor_restore = parameters["src_regressor_restore"]
    if "tgt_encoder_restore" in parameters: params.tgt_encoder_restore = parameters["tgt_encoder_restore"]
    if "d_model_restore" in parameters: params.d_model_restore = parameters["d_model_restore"]
    if "src_classifier_restore" in parameters: params.src_classifier_restore = parameters["src_classifier_restore"]

    if "src_model_trained" in parameters: params.src_model_trained = parameters["src_model_trained"]
    if "e_input_dims" in parameters: params.e_input_dims = parameters["e_input_dims"]
    if "e_hidden_dims" in parameters: params.e_hidden_dims = parameters["e_hidden_dims"]
    if "e_output_dims" in parameters: params.e_output_dims = parameters["e_output_dims"]
    if "r_input_dims" in parameters: params.r_input_dims = parameters["r_input_dims"]
    if "tgt_model_trained" in parameters: params.tgt_model_trained = parameters["tgt_model_trained"]
    if "d_input_dims" in parameters: params.d_input_dims = parameters["d_input_dims"]
    if "d_hidden_dims" in parameters: params.d_hidden_dims = parameters["d_hidden_dims"]
    if "d_output_dims" in parameters: params.d_output_dims = parameters["d_output_dims"]
    if "num_gpu" in parameters: params.num_gpu = parameters["num_gpu"]
    if "num_epochs_pre" in parameters: params.num_epochs_pre = parameters["num_epochs_pre"]
    if "log_step_pre" in parameters: params.log_step_pre = parameters["log_step_pre"]
    if "eval_step_pre" in parameters: params.eval_step_pre = parameters["eval_step_pre"]
    if "save_step_pre" in parameters: params.save_step_pre = parameters["save_step_pre"]
    if "num_epochs" in parameters: params.num_epochs = parameters["num_epochs"]
    if "log_step" in parameters: params.log_step = parameters["log_step"]
    if "save_step" in parameters: params.save_step = parameters["save_step"]
    if "manual_seed" in parameters: params.manual_seed = parameters["manual_seed"]
    if "d_learning_rate" in parameters: params.d_learning_rate = parameters["d_learning_rate"]
    if "c_learning_rate" in parameters: params.c_learning_rate = parameters["c_learning_rate"]
    if "g_learning_rate" in parameters: params.g_learning_rate = parameters["g_learning_rate"]
    if "beta1" in parameters: params.beta1 = parameters["beta1"]
    if "beta2" in parameters: params.beta2 = parameters["beta2"]

def train(data):
    print("ADDA Training [" + data["col"] + "] start!")
    
    start = time.time()
    
    params.tIndex = data["col"]+"_"
    params.model_root = data["filepath"]
    params.baseUnit = data["baseUnit"]
    params.featureNum = data["featureNum"]
    
    if data["parameters"] is not None:
        parameters = data["parameters"]
        print(data["parameters"])

        if "dataset_mean_value" in parameters: 
            params.dataset_mean_value = parameters["dataset_mean_value"]
            params.dataset_mean = (params.dataset_mean_value, params.dataset_mean_value, params.dataset_mean_value)
            
        if "dataset_std_value" in parameters: 
            params.dataset_std_value = parameters["dataset_std_value"]
            params.dataset_std = (params.dataset_std_value, params.dataset_std_value, params.dataset_std_value)

        if "src_encoder_restore" in parameters: params.src_encoder_restore = parameters["src_encoder_restore"]
        if "src_regressor_restore" in parameters: params.src_regressor_restore = parameters["src_regressor_restore"]
        if "tgt_encoder_restore" in parameters: params.tgt_encoder_restore = parameters["tgt_encoder_restore"]
        if "d_model_restore" in parameters: params.d_model_restore = parameters["d_model_restore"]
        if "src_classifier_restore" in parameters: params.src_classifier_restore = parameters["src_classifier_restore"]

        if "src_model_trained" in parameters: params.src_model_trained = parameters["src_model_trained"]
        if "e_input_dims" in parameters: params.e_input_dims = parameters["e_input_dims"]
        if "e_hidden_dims" in parameters: params.e_hidden_dims = parameters["e_hidden_dims"]
        if "e_output_dims" in parameters: params.e_output_dims = parameters["e_output_dims"]
        if "r_input_dims" in parameters: 
            params.r_input_dims = parameters["r_input_dims"]
            print("updated r_input_dims:{}".format(params.r_input_dims))

        if "tgt_model_trained" in parameters: params.tgt_model_trained = parameters["tgt_model_trained"]
        if "d_input_dims" in parameters: params.d_input_dims = parameters["d_input_dims"]
        if "d_hidden_dims" in parameters: params.d_hidden_dims = parameters["d_hidden_dims"]
        if "d_output_dims" in parameters: params.d_output_dims = parameters["d_output_dims"]
        if "num_gpu" in parameters: params.num_gpu = parameters["num_gpu"]
        if "num_epochs_pre" in parameters: params.num_epochs_pre = parameters["num_epochs_pre"]
        if "log_step_pre" in parameters: params.log_step_pre = parameters["log_step_pre"]
        if "eval_step_pre" in parameters: params.eval_step_pre = parameters["eval_step_pre"]
        if "save_step_pre" in parameters: params.save_step_pre = parameters["save_step_pre"]
        if "num_epochs" in parameters: params.num_epochs = parameters["num_epochs"]
        if "log_step" in parameters: params.log_step = parameters["log_step"]
        if "save_step" in parameters: params.save_step = parameters["save_step"]
        if "manual_seed" in parameters: params.manual_seed = parameters["manual_seed"]
        if "d_learning_rate" in parameters: params.d_learning_rate = parameters["d_learning_rate"]
        if "c_learning_rate" in parameters: params.c_learning_rate = parameters["c_learning_rate"]
        if "g_learning_rate" in parameters: params.g_learning_rate = parameters["g_learning_rate"]
        if "beta1" in parameters: params.beta1 = parameters["beta1"]
        if "beta2" in parameters: params.beta2 = parameters["beta2"]

        print("r_input_dims:{}".format(params.r_input_dims))
        print("d_input_dims:{}".format(params.d_input_dims))
        
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset

    src_data_loader = data["dataloaders"]["srcTrain"]
    tgt_data_loader = data["dataloaders"]["tarTrain"]
    src_data_loader_eval = data["dataloaders"]["srcTest"]
    tgt_data_loader_eval = data["dataloaders"]["tarTest"]

    y_train = data["y_train"]

    encoder = data["parameters"]["encoder"]

    # load models

    src_encoder = None
    if encoder == "MLP":
        src_encoder = init_model(net=LeNetEncoderMLP(input_dims=params.e_input_dims,
                                                hidden_dims=params.e_hidden_dims,
                                                output_dims=params.e_output_dims),
                                restore=params.src_encoder_restore)
        tgt_encoder = init_model(net=LeNetEncoderMLP(input_dims=params.e_input_dims,
                                            hidden_dims=params.e_hidden_dims,
                                            output_dims=params.e_output_dims),
                                restore=params.tgt_encoder_restore)
    elif encoder == "convLSTM":
        src_encoder = init_model(net=LeNetEncoderConvLSTM(output_dims=params.e_output_dims,
                                                            baseUnit = params.baseUnit ,
                                                            featureNum = params.featureNum),
                                restore=params.src_encoder_restore)
        tgt_encoder = init_model(net=LeNetEncoderConvLSTM(output_dims=params.e_output_dims,
                                                            baseUnit = params.baseUnit ,
                                                            featureNum = params.featureNum),
                                restore=params.tgt_encoder_restore)
    else:
        return

    src_regressor = init_model(net=LeNetRegressorMLP(input_dims=params.r_input_dims),
                                restore=params.src_regressor_restore)


    
    critic = init_model(DiscriminatorMLP(input_dims=params.d_input_dims,
                                      hidden_dims=params.d_hidden_dims,
                                      output_dims=params.d_output_dims),
                        restore=params.d_model_restore)

    # train source model
    print("=== Training regressor for source domain ===")
    print(">>> Source Encoder <<<")
    print(src_encoder)
    print(">>> Source regressor <<<")
    print(src_regressor)

    if not (src_encoder.restored and src_regressor.restored and
            params.src_model_trained):
        src_encoder, src_regressor = train_src_mlp(
            src_encoder, src_regressor, src_data_loader)

    # eval source model
    print("=== Evaluating regressor for source domain ===")
    eval_src_mlp(src_encoder, src_regressor, src_data_loader_eval)

    # train target encoder by GAN
    print("=== Training encoder for target domain ===")
    print(">>> Target Encoder <<<")
    print(tgt_encoder)
    print(">>> Critic <<<")
    print(critic)

    # init weights of target encoder with those of source encoder
    if not tgt_encoder.restored:
        tgt_encoder.load_state_dict(src_encoder.state_dict())

    if not (tgt_encoder.restored and critic.restored and
            params.tgt_model_trained):
        tgt_encoder = train_tgt_mlp(src_encoder, tgt_encoder, critic,
                                src_data_loader, tgt_data_loader)

    end = time.time()
    # eval target encoder on test set of target dataset
    print("=== [{}] Evaluating regressor for encoded target domain ===".format(data["col"])+", spent: %.2fs" % (end - start))
    print(">>> source only <<<")
    rmse,mae,mape,mase = eval_tgt_mlp(src_encoder, src_regressor, tgt_data_loader_eval,y_train)
    print(">>> domain adaption <<<")
    t_rmse,t_mae,t_mape,t_mase = eval_tgt_mlp(tgt_encoder, src_regressor, tgt_data_loader_eval,y_train)
    
    return {data["col"]:{"mae_src":mae,"rmse_src":rmse,"mase_src":mase,"mae_tar":t_mae,"rmse_tar":t_rmse,"mase_tar":t_mase}}
