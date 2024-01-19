import copy
from matplotlib import pyplot as plt

from utils import *


class Evaluator:
    def __init__(self, metric, patience_max):
        self.metric = metric

        self.training_loss = []
        self.val_loss = []
        self.test_loss = []

        self.training_best = np.inf
        self.val_best = np.inf if self.metric in ["mse", "mpe"] else -np.inf

        self.best_val_model = None

        self.patience_max = patience_max
        self.patience_counter = 0

    def record_training(self, loss):
        self.training_loss.append(loss)
        if loss < self.training_best:
            self.training_best = loss

    def record_val(self, performance, state_dict):
        self.patience_counter += 1
        self.val_loss.append(performance)

        if self.metric in ["mse","mpe","interval_width"]:
            if performance < self.val_best:
                self.val_best = performance
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0

        elif self.metric in ["ndcg", "coverage"]:
            if performance[0] > self.val_best:
                self.val_best = performance[0]
                self.best_val_model = copy.deepcopy(state_dict)
                self.patience_counter = 0
        else:
            raise Exception("invalid metric")

        if self.patience_counter >= self.patience_max:
            return True
        return False

    def record_test(self, performance):
        self.test_loss.append(performance)

    def get_best_model(self):
        return self.best_val_model

    def epoch_log(self, epoch):
        if len(self.training_loss) > 0:
            print("epoch:{}, tr loss:{:.5}".format(epoch,self.training_loss[epoch]))

        if len(self.val_loss)>0:
            print("epoch:{}, val performance:{} ".format(epoch,self.val_loss[epoch]))
        
        if len(self.test_loss)>0:
            print("epoch:{}, test loss:{} ".format(epoch,self.test_loss[epoch]))

    def get_val_best_performance(self):
        return self.val_best

    def plot(self):
            
        plt.plot(self.val_loss, label=f"val {self.metric}")
        plt.plot(self.test_loss, label=f"test {self.metric}")
        
        plt.legend()
        plt.show()
