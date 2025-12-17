import os
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import nn
from torch.utils import data
from torchvision.models._utils import IntermediateLayerGetter


class Solver():
    def __init__(self,**kwargs):

        # turn on the CUDA if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loaders = kwargs.get('data_loaders',None)

        if self.loaders!=None:
            self.train_loader = self.loaders['train']
            self.test_loader = self.loaders['test']

        self.ae = kwargs.get('ae',None)
        self.net = kwargs.get('net',None)
        self.loss_fn = kwargs.get('criterion',None)
        self.optim = kwargs.get('optimizer',None)
        self.lr = kwargs.get('lr',1e-3)

        if self.ae!=None:
          self.optim = torch.optim.Adam(list(self.net.parameters())+list(self.ae.parameters()),lr=self.lr)
        else:
          self.optim = torch.optim.Adam(self.net.parameters(),lr=self.lr)

        self.max_epochs = kwargs.get('num_epochs',25)
        self.print_every = 1
        self.ckpt_dir = kwargs.get('output_dir','./checkpoint')
        self.ckpt_name = kwargs.get('ckpt_name','network')

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

    def fit(self):
        best_test_acc = 0
        for epoch in range(self.max_epochs):
            self.net.train()
            if self.ae!=None:
              self.ae.train()
            for step, inputs in enumerate(self.train_loader):
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device).long()

                if self.ae!=None:
                    images = self.ae(images)

                pred = self.net(images)

                loss = self.loss_fn(pred, labels)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            if (epoch + 1) % self.print_every == 0:
                train_acc = self.evaluate(loader=self.train_loader)
                test_acc = self.evaluate(loader=self.test_loader)

                print("Epoch [{}/{}] Loss: {:.3f} Train Acc: {:.3f}, Test Acc: {:.3f}".
                      format(epoch + 1, self.max_epochs, loss.item(), train_acc, test_acc))
            if test_acc>best_test_acc:
                best_test_acc = test_acc
                print("Saving network at epoch [{}/{}]".format(epoch+1,self.max_epochs))
                self.save(self.ckpt_dir, self.ckpt_name, epoch + 1)

    def evaluate(self,**kwargs):
        loader = kwargs.get('loader',None)
        self.net.eval()
        if self.ae != None:
            self.ae.eval()
        num_correct, num_total = 0, 0
        if loader==None:
            loader = self.test_loader
        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                labels = inputs[1].to(self.device).long()

                if self.ae!=None:
                    images = self.ae(images)


                outputs = self.net(images)
                _, preds = torch.max(outputs.detach(), 1)

                num_correct += (preds == labels).sum().item()
                num_total += labels.size(0)

        return num_correct / num_total

    def evaluate_bottleneck(self,loader,outputs):
        i = 0
        num_correct = 0
        num_total = 0
        for inputs in loader:
            images = inputs[0]
            labels = inputs[1]
            _, preds = torch.max(torch.tensor(np.expand_dims(outputs[i,:],0)), 1)
            num_correct += (preds == labels).sum().item()
            num_total += labels.size(0)
            i+=1

        return num_correct / num_total

    def intermediate_output(self,loader):
        self.net.eval()
        self.net.set_intermediate_output(True)

        fc_eval = list()

        with torch.no_grad():
            for inputs in loader:
                images = inputs[0].to(self.device)
                #labels = inputs[1].to(self.device)
                outputs = self.net(images)
                fc_eval.append(outputs.squeeze().detach().cpu().numpy())
        fc_eval = np.array(fc_eval)

        return fc_eval

    def save(self, ckpt_dir, ckpt_name, global_step):
        save_path = os.path.join(
            ckpt_dir, "{}.pth".format(ckpt_name))
        torch.save(self.net.state_dict(), save_path)
        if self.ae!=None:
            save_path = os.path.join(
                ckpt_dir, "{}.pth".format("ae"))
            torch.save(self.ae.state_dict(), save_path)


    def set_network(self,net):
        self.net = net
