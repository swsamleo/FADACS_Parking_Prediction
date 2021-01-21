class Callback(object):
    """
    Abstract base class used to build new callbacks.
    """

    def __init__(self):
        pass

    def set_params(self, params):
        self.params = params

    def set_trainer(self, model):
        self.trainer = model

    def on_epoch_begin(self, epoch, loss):
        pass

    def on_epoch_end(self, epoch, loss):
        pass

    def on_batch_begin(self, batch, loss):
        pass

    def on_batch_end(self, batch, loss):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass

class EarlyStopping(Callback):
    """
    Early Stopping to terminate training early under certain conditions
    """

    def __init__(self, min_delta=0, patience=10):
        """
        EarlyStopping callback to exit the training loop if training or
        validation loss does not improve by a certain amount for a certain
        number of epochs
        Arguments
        ---------
        monitor : string in {'val_loss', 'loss'}
            whether to monitor train or val loss
        min_delta : float
            minimum change in monitored value to qualify as improvement.
            This number should be positive.
        patience : integer
            number of epochs to wait for improvment before terminating.
            the counter be reset after each improvment
        """
        self.min_delta = min_delta
        self.patience = patience
        self.wait = 0
        self.best_loss = 1e-15
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.stop_flag = False
        super(EarlyStopping, self).__init__()

    def on_train_begin(self):
        self.wait = 0
        self.best_loss = 1e15

    def on_epoch_end(self, epoch, loss):
        current_loss = loss
        if current_loss is None:
            pass
        else:
            if (current_loss - self.best_loss) < -self.min_delta:
                self.best_loss = current_loss
                self.best_epoch = epoch
                self.wait = 1
            else:
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.stop_flag = True
                self.wait += 1

    def on_train_end(self):
        if self.stopped_epoch > 0:
            print('\nTerminated Training for Early Stopping at Epoch %04i' % 
                (self.stopped_epoch))