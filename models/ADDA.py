"""Discriminator model for ADDA."""
import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import time

from . import ADDA_params as params

def make_variable(tensor, volatile=False):
    """Convert Tensor to Variable."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return Variable(tensor, volatile=volatile)

def make_cuda(tensor):
    """Use CUDA if it's available."""
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def denormalize(x, std, mean):
    """Invert normalization, and then convert array into image."""
    out = x * std + mean
    return out.clamp(0, 1)


def init_weights(layer):
    """Init weights for layers w.r.t. the original paper."""
    layer_name = layer.__class__.__name__
    if layer_name.find("Conv") != -1:
        layer.weight.data.normal_(0.0, 0.02)
    elif layer_name.find("BatchNorm") != -1:
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.fill_(0)


def init_random_seed(manual_seed):
    """Init random seed."""
    seed = None
    if manual_seed is None:
        seed = random.randint(1, 10000)
    else:
        seed = manual_seed
    print("use random seed: {}".format(seed))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def init_model(net, restore):
    """Init models with cuda and weights."""
    # init weights of model
    net.apply(init_weights)

    # restore model weights
    if restore is not None and os.path.exists(restore):
        net.load_state_dict(torch.load(restore))
        net.restored = True
        print("Restore model from: {}".format(os.path.abspath(restore)))

    # check if cuda is available
    if torch.cuda.is_available():
        cudnn.benchmark = True
        net.cuda()

    return net


def save_model(net, filename):
    """Save trained model."""
    if not os.path.exists(params.model_root):
        os.makedirs(params.model_root)
    torch.save(net.state_dict(),
               os.path.join(params.model_root, filename))
    print("save pretrained model to: {}".format(os.path.join(params.model_root,
                                                             filename)))

class Discriminator(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(Discriminator, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out


class DiscriminatorMLP(nn.Module):
    """Discriminator model for source domain."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init discriminator."""
        super(DiscriminatorMLP, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
            nn.LogSoftmax()
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out
    
"""LeNet model for ADDA."""

class LeNetEncoder(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetEncoder, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            # 1st conv layer
            # input [1 x 28 x 28]
            # output [20 x 12 x 12]
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            # 2nd conv layer
            # input [20 x 12 x 12]
            # output [50 x 4 x 4]
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(50 * 4 * 4, 500)

    def forward(self, input):
        """Forward the LeNet."""
        conv_out = self.encoder(input)
        feat = self.fc1(conv_out.view(-1, 50 * 4 * 4))
        return feat


class LeNetClassifier(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self):
        """Init LeNet encoder."""
        super(LeNetClassifier, self).__init__()
        self.fc2 = nn.Linear(500, 10)

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc2(out)
        return out


class LeNetEncoderMLP(nn.Module):
    """LeNet encoder model for ADDA."""

    def __init__(self, input_dims, hidden_dims, output_dims):
        """Init LeNet encoder."""
        super(LeNetEncoderMLP, self).__init__()

        self.restored = False

        self.encoder = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, input):
        """Forward the LeNet."""
        feat = self.encoder(input.view(input.size(0), -1))
        return feat


class LeNetRegressorMLP(nn.Module):
    """LeNet classifier model for ADDA."""

    def __init__(self, input_dims):
        """Init LeNet encoder."""
        super(LeNetRegressorMLP, self).__init__()
        self.fc2 = nn.Linear(input_dims, 1)
        self.restored = False

    def forward(self, feat):
        """Forward the LeNet classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = F.sigmoid(self.fc2(out))
        return out
    
#################

def train_tgt(src_encoder, tgt_encoder, critic,
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
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs):
        # zip source and target data pair
        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = make_variable(images_src)
            images_tgt = make_variable(images_tgt)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

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
            feat_tgt = tgt_encoder(images_tgt)

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
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
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
    return tgt_encoder


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
                               lr=params.c_learning_rate,
                               betas=(params.beta1, params.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=params.d_learning_rate,
                                  betas=(params.beta1, params.beta2))
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
            if ((step + 1) % params.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
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
    return tgt_encoder

#####################################

def train_src(encoder, classifier, data_loader):
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
    criterion = nn.CrossEntropyLoss()

    ####################
    # 2. train network #
    ####################

    for epoch in range(params.num_epochs_pre):
        for step, (images, labels) in enumerate(data_loader):
            # make images and labels variable
            images = make_variable(images)
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(images))
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
            eval_src(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, params.tIndex+"source-encoder-{}.pt".format(epoch + 1))
            save_model(classifier, params.tIndex+"source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, params.tIndex+"source-encoder-final.pt")
    save_model(classifier, params.tIndex+"source-classifier-final.pt")

    return encoder, classifier


def eval_src(encoder, classifier, data_loader):
    """Evaluate classifier for source domain."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels)

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


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
            labels = make_variable(labels.squeeze_())

            # zero gradients for optimizer
            optimizer.zero_grad()

            # compute loss for critic
            preds = classifier(encoder(datas))
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
            eval_src(encoder, classifier, data_loader)

        # save model parameters
        if ((epoch + 1) % params.save_step_pre == 0):
            save_model(encoder, params.tIndex+"source-encoder-{}.pt".format(epoch + 1))
            save_model(
                classifier, params.tIndex+"source-classifier-{}.pt".format(epoch + 1))

    # # save final model
    save_model(encoder, params.tIndex+params.src_encoder_restore)
    save_model(classifier, params.tIndex+params.src_classifier_restore)

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
        labels = make_variable(labels)

        preds = classifier(encoder(datas))
        loss += criterion(preds, labels).item()
        mae_loss += mae_crit1(preds, labels).item()

    loss /= len(data_loader)
    loss = loss**(0.5)
    mae_loss /= len(data_loader)

    print("RMSE = {}, MAE = {}".format(loss, mae_loss))


##########

def eval_tgt(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    encoder.eval()
    classifier.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = make_variable(images, volatile=True)
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(images))
        loss += criterion(preds, labels).item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).cpu().sum()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def eval_tgt_mlp(encoder, classifier, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
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
        labels = make_variable(labels).squeeze_()

        preds = classifier(encoder(datas))
        loss += criterion(preds, labels).item()
        mae_loss += mae_crit1(preds, labels).item()

    loss /= len(data_loader)
    mae_loss = loss
    loss = loss**(0.5)
    #mae_loss /= len(data_loader)

    print("RMSE = {}, MAE = {}".format(loss, mae_loss))
    return loss, mae_loss

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
    if "beta1" in parameters: params.beta1 = parameters["beta1"]
    if "beta2" in parameters: params.beta2 = parameters["beta2"]

def train(data):
    print("ADDA Training [" + data["col"] + "] start!")
    
    start = time.time()
    
    params.tIndex = data["col"]+"_"
    params.model_root = data["filepath"]
    
    if data["parameters"] is not None:
        updateParams(data["parameters"] )
        
    # init random seed
    init_random_seed(params.manual_seed)

    # load dataset

    src_data_loader = data["dataloaders"]["srcTrain"]
    tgt_data_loader = data["dataloaders"]["tarTrain"]
    src_data_loader_eval = data["dataloaders"]["srcTest"]
    tgt_data_loader_eval = data["dataloaders"]["tarTest"]

    # load models
    src_encoder = init_model(net=LeNetEncoderMLP(input_dims=params.e_input_dims,
                                            hidden_dims=params.e_hidden_dims,
                                            output_dims=params.e_output_dims),
                             restore=params.src_encoder_restore)
    src_regressor = init_model(net=LeNetRegressorMLP(input_dims=params.r_input_dims),
                                restore=params.src_regressor_restore)
    tgt_encoder = init_model(net=LeNetEncoderMLP(input_dims=params.e_input_dims,
                                            hidden_dims=params.e_hidden_dims,
                                            output_dims=params.e_output_dims),
                             restore=params.tgt_encoder_restore)
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
    rmse,mae = eval_tgt_mlp(src_encoder, src_regressor, tgt_data_loader_eval)
    print(">>> domain adaption <<<")
    t_rmse,t_mae = eval_tgt_mlp(tgt_encoder, src_regressor, tgt_data_loader_eval)
    
    return {data["col"]:{"mae_src":mae,"rmse_src":rmse,"mae_tar":t_mae,"rmse_tar":t_rmse}}
