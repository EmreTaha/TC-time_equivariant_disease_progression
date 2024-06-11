from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as opt
import numpy as np

from utils import roc_aucs,pr_aucs,f1_score, balanced_accs
from utils import initialize

initialize(allow_tf32=False)

class Linear_Protocoler(object):
    def __init__(self, backbone_net, num_classes: int = 10, out_dim: Optional[int] = None, device : str = 'cpu', finetune=False):
        self.device = device
        self.num_classes = num_classes
        # Copy net
        self.backbone = deepcopy(backbone_net)
        self.finetune = finetune
        if not self.finetune:
            # Turn off gradients
            # Update: this option is turned off due to some problems
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        if finetune:
            self.backbone.train()
        # get out dimension
        if out_dim:
            out_dim = out_dim
        else:
            out_dim = p.shape[0]
        # Add classification layer
        layers = []
        if self.num_classes==1 or self.num_classes==2:
            layers.append(nn.Linear(out_dim, 1))
        else:
            layers.append(nn.Linear(out_dim, self.num_classes))

        self.classifier = torch.nn.Sequential(*layers)
        nn.init.normal_(self.classifier[0].weight,mean=0.0, std=0.01)
        self.classifier[0].bias.data.zero_()

        # Send to device
        self.backbone = self.backbone.to(self.device)
        self.classifier = self.classifier.to(self.device)

    def train(self, dataloader, num_epochs, lr : float = 1e-3, schedule : bool = False, class_weights = 5.0, dictionary=False, amp=False):
        # Define optimizer
        if self.finetune:
            params = list(self.classifier.parameters()) + list(self.backbone.parameters())
            self.backbone.train()
        else:
            print("Warning: You are training only the classifier layer. Make sure you know what you are doing.")
            params = list(self.classifier.parameters())
            self.backbone.eval()

        optimizer = opt.Adam(params, lr)
        # Define loss
        if self.num_classes==1 or self.num_classes==2:
            ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(class_weights))
        else:
            ce_loss = nn.CrossEntropyLoss()
        # Define scheduler
        if schedule:
            scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs,  eta_min=lr*1e-4)
        else:
            scheduler = None
        
        # Train
        self.classifier.train()

        # scaler TODO make this optional?
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        #if not amp:
        #    convert_model_to_fp32(self.backbone)    
        
        for epoch in range(num_epochs):
            for inputs in dataloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs
                x,y = x.to(self.device), y.to(self.device)
                y = y.reshape(-1,1)
                # forward
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone(x)
                    loss = ce_loss(self.classifier(representaions), y.float())

                # backward
                optimizer.zero_grad()
                #loss.backward()
                scaler.scale(loss).backward() if amp else loss.backward()
                #optimizer.step()
                scaler.step(optimizer)  if amp else optimizer.step()
                if amp: scaler.update()

            if scheduler:
                scheduler.step()
    
    def get_3Dmetrics_thr(self, dataloader,valloader, dictionary=False, amp=False):
        # This finds a threshold for the predictions that maximizes the bacc score
        # Store all predictions and true labels
        all_probs = []
        all_labels = []

        with torch.no_grad():
            self.classifier.eval()
            self.backbone.eval()
            for inputs in valloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs

                x, y = x.cuda(), y.cuda()
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone(x)
                    outputs = self.classifier(representaions)

                if self.num_classes==1 or self.num_classes==2:
                    all_probs.extend(torch.sigmoid(outputs).data.reshape(-1).cpu())
                else:
                    all_probs.extend(torch.softmax(outputs, dim=1).data.reshape(-1).cpu())
                all_labels.extend(y.reshape(-1).cpu())

        # Convert lists to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Define a range of possible thresholds
        thresholds = np.linspace(0, 1, num=100)

        # Find the best threshold
        best_accuracy = 0
        best_threshold = 0
        for threshold in thresholds:
            # Binarize predictions based on the threshold
            binarized_predictions = (all_probs > threshold).astype(int)

            # Calculate accuracy
            accuracy = balanced_accs(all_labels, binarized_predictions)

            # Update best accuracy and threshold if current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        print("Best Threshold:", best_threshold)
        print("Best Accuracy:", best_accuracy)
                
        correct = 0
        total = 0
        preds = []
        labels = []

        with torch.no_grad():
            self.classifier.eval()
            self.backbone.eval()
            for inputs in dataloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs

                x, y = x.cuda(), y.cuda()
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone(x)
                    outputs = self.classifier(representaions)

                labels += y.reshape(-1).cpu()
                #TODO unnecassary?
                if self.num_classes==1 or self.num_classes==2:
                    preds += torch.sigmoid(outputs).data.reshape(-1).cpu()
                else:
                    preds += torch.softmax(outputs, dim=1).data.reshape(-1).cpu()

                # the class with the highest energy is what we choose as prediction
                # TODO check if the below code is suitable for multiclass case
                probs = torch.sigmoid(outputs).data
                predicted = probs.round()
                total += y.size(0)
                correct += (predicted == y).sum().item()

            self.classifier.train()
        
        preds_c = (np.array(preds) > best_threshold).astype(int)
        f1 = f1_score(labels,preds_c)
        b_accs = balanced_accs(labels,preds_c)
        prauc = pr_aucs(labels,preds)
        rocauc = roc_aucs(labels,preds)

        print(  f"classification: roc-auc: {rocauc:.3f}.. "
                f"classification: pr-auc: {prauc:.3f}.. "
                f"classification: F1: {f1:.3f}.. "
                f"classification: Balanced Accuracy: {b_accs:.3f}.. "
                f"classification: accuracy: {correct / total:.3f}..")

        return correct / total, f1, prauc, rocauc, b_accs
    
# This is specific to TC, it forwards futures in time for 6 months and combine it with the inital one to make predictions
class Traj_Protocoler(object):
    def __init__(self, backbone_net, num_classes: int = 10, out_dim: Optional[int] = None, device : str = 'cpu', finetune=False, forward_time=0.0):
        self.device = device
        self.num_classes = num_classes
        # Copy net
        self.backbone = deepcopy(backbone_net)
        self.finetune = finetune
        self.forward_time = forward_time
        if not self.finetune:
            # Turn off gradients
            # Update: this option is turned off due to some problems
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()
        if finetune:
            self.backbone.train()
        # get out dimension
        if out_dim:
            out_dim = out_dim
        else:
            out_dim = p.shape[0]
        # Add classification layer
        layers = []
        dim_multiplier = 1
        if self.num_classes==1 or self.num_classes==2:
            layers.append(nn.Linear(out_dim*dim_multiplier, 1))
        else:
            layers.append(nn.Linear(out_dim*dim_multiplier, self.num_classes))

        self.classifier = torch.nn.Sequential(*layers)
        self.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
        self.classifier[0].bias.data.zero_()

        # Send to device
        self.backbone = self.backbone.to(self.device)
        self.classifier = self.classifier.to(self.device)

        self.forward_time = torch.tensor(self.forward_time, dtype=torch.float32)
        self.forward_time = self.forward_time.reshape(1,1)
        self.forward_time = self.forward_time.to(self.device)
    
    def train(self, dataloader, num_epochs, lr : float = 1e-3, schedule : bool = False, class_weights = 5.0, dictionary=False, amp=False):
        # Define optimizer
        if self.finetune:
            params = list(self.classifier.parameters()) + list(self.backbone.parameters())
            self.backbone.train()
        else:
            print("Warning: You are training only the classifier layer. Make sure you know what you are doing.")
            params = list(self.classifier.parameters())
            self.backbone.eval()

        optimizer = opt.Adam(params, lr)
        # Define loss
        if self.num_classes==1 or self.num_classes==2:
            ce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor(class_weights))
        else:
            ce_loss = nn.CrossEntropyLoss()
        # Define scheduler
        if schedule:
            scheduler = opt.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs,  eta_min=lr*1e-4)
        else:
            scheduler = None
        
        # Train
        self.classifier.train()

        # scaler TODO make this optional?
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
        #if not amp:
        #    convert_model_to_fp32(self.backbone)    
        for epoch in range(num_epochs):
            for inputs in dataloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs
                x,y = x.to(self.device), y.to(self.device)
                y = y.reshape(-1,1)
                # forward
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone.get_representations(x)

                    if self.forward_time > 0.0:
                        forward_time = self.forward_time.repeat(y.shape[0], 1)
                        forward_rep = self.backbone.forward_repr(representaions, forward_time)
                        #representaions = torch.cat((representaions, forward_rep), dim=1)       
                        representaions = (forward_rep+representaions)/2.0

                    loss = ce_loss(self.classifier(representaions), y.float())

                # backward
                optimizer.zero_grad()
                #loss.backward()
                scaler.scale(loss).backward() if amp else loss.backward()
                #optimizer.step()
                scaler.step(optimizer)  if amp else optimizer.step()
                if amp: scaler.update()

            if scheduler:
                scheduler.step()
    
    def get_3Dmetrics_thr(self, dataloader,valloader, dictionary=False, amp=False):
        # This finds a threshold for the predictions that maximizes the bacc score
        # Store all predictions and true labels
        all_probs = []
        all_labels = []

        with torch.no_grad():
            self.classifier.eval()
            self.backbone.eval()
            for inputs in valloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs

                x, y = x.cuda(), y.cuda()
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone.get_representations(x)
                    if self.forward_time > 0.0:
                        forward_time = self.forward_time.repeat(y.shape[0], 1)
                        forward_rep = self.backbone.forward_repr(representaions, forward_time)
                        #representaions = torch.cat((representaions, forward_rep), dim=1)       
                        representaions = (forward_rep+representaions)/2.0
                        #representaions = forward_rep

                    outputs = self.classifier(representaions)

                if self.num_classes==1 or self.num_classes==2:
                    all_probs.extend(torch.sigmoid(outputs).data.reshape(-1).cpu())
                else:
                    all_probs.extend(torch.softmax(outputs, dim=1).data.reshape(-1).cpu())
                all_labels.extend(y.reshape(-1).cpu())

        # Convert lists to numpy arrays
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Define a range of possible thresholds
        thresholds = np.linspace(0, 1, num=100)

        # Find the best threshold
        best_accuracy = 0
        best_threshold = 0
        for threshold in thresholds:
            # Binarize predictions based on the threshold
            binarized_predictions = (all_probs > threshold).astype(int)

            # Calculate accuracy
            accuracy = balanced_accs(all_labels, binarized_predictions)

            # Update best accuracy and threshold if current accuracy is higher
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

        print("Best Threshold:", best_threshold)
        print("Best Accuracy:", best_accuracy)
                
        correct = 0
        total = 0
        preds = []
        labels = []

        with torch.no_grad():
            self.classifier.eval()
            self.backbone.eval()
            for inputs in dataloader:
                x,y = (inputs["image"], inputs["label"]) if dictionary else inputs

                x, y = x.cuda(), y.cuda()
                y = y.reshape(-1,1)
                # calculate outputs by running images through the network
                with torch.cuda.amp.autocast(enabled=amp):
                    representaions = self.backbone.get_representations(x)
                    if self.forward_time > 0.0:
                        forward_time = self.forward_time.repeat(y.shape[0], 1)
                        forward_rep = self.backbone.forward_repr(representaions, forward_time)
                        #representaions = torch.cat((representaions, forward_rep), dim=1)       
                        representaions = (forward_rep+representaions)/2.0
                        #representaions = forward_rep

                    outputs = self.classifier(representaions)

                labels += y.reshape(-1).cpu()
                #TODO unnecassary?
                if self.num_classes==1 or self.num_classes==2:
                    preds += torch.sigmoid(outputs).data.reshape(-1).cpu()
                else:
                    preds += torch.softmax(outputs, dim=1).data.reshape(-1).cpu()

                # the class with the highest energy is what we choose as prediction
                # TODO check if the below code is suitable for multiclass case
                probs = torch.sigmoid(outputs).data
                predicted = probs.round()
                total += y.size(0)
                correct += (predicted == y).sum().item()

            self.classifier.train()
        
        preds_c = (np.array(preds) > best_threshold).astype(int)
        f1 = f1_score(labels,preds_c)
        b_accs = balanced_accs(labels,preds_c)
        prauc = pr_aucs(labels,preds)
        rocauc = roc_aucs(labels,preds)

        print(  f"classification: roc-auc: {rocauc:.3f}.. "
                f"classification: pr-auc: {prauc:.3f}.. "
                f"classification: F1: {f1:.3f}.. "
                f"classification: Balanced Accuracy: {b_accs:.3f}.. "
                f"classification: accuracy: {correct / total:.3f}..")

        return correct / total, f1, prauc, rocauc, b_accs