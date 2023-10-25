import os
import sys
import time
sys.path.append(os.path.join(os.path.abspath(os.getcwd())))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.utils as vutils
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from IPython.display import display, clear_output


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            nn.Conv2d(3, 10, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3),
            nn.Dropout2d(0.25),
            nn.ReLU(),
            nn.MaxPool2d(5),
            nn.Flatten(),
            nn.Linear(8820, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 100),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(100, 2),
            nn.Softmax())
        
    def forward(self, x):
        return self.model(x)

    
class Model(object):
    def __init__(self, name, device, train_loader, valid_loader, test_loader, classes):
        self.name = name
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.classes = classes
        self.net = SimpleCNN()
        self.net.to(self.device)
        self.optimizer = None
        self.lr_scheduler = None

        
    def create_optim(self, lr, alpha=0.9, beta=0.999, gamma=0.25):
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=lr, betas=(alpha, beta))
        self.test_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                        lr=lr, betas=(alpha, beta))
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)

        
    def train(self, epochs, esp=4, out_dir='', verbose=True):
        history = pd.DataFrame(columns=['Train loss', 'Validation loss', 
                                'Training time, s', 'Validation time, s', 'Learning rate'])
        history.index.name = 'epoch'
        
        best_loss = 1e8
        es = 0
        for epoch in range(epochs):
            if verbose:
                display(history)
            train_loss = 0.0
            train_losses, valid_losses = [], []

            start_time = time.time()

            # train subcycle
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.long().to(self.device)

                
                self.net.train()

                self.optimizer.zero_grad()
                outputs = self.net(data)

                loss = self.net.loss(outputs, target)
                loss.backward()
                self.optimizer.step()

                train_losses.append(loss.item())
    
            # valid subcycle
            mid_time = time.time()
            self.net.eval()
    
            for data, target in self.valid_loader:
                data, target = data.to(self.device), target.long().to(self.device)
                output = self.net(data)
                loss = self.net.loss(output, target)
                valid_losses.append(loss.item())

            end_time = time.time()
    
            avg_train_loss = np.average(train_losses)
            avg_val_loss = np.average(valid_losses)
            train_time = mid_time - start_time
            val_time = end_time - mid_time
            
            history = history.append(pd.Series([avg_train_loss, avg_val_loss, 
                                        mid_time - start_time, end_time - mid_time,
                                       self.optimizer.state_dict()['param_groups'][0]['lr']],
                                       name=f'{epoch + 1}', index=history.columns))
            if verbose:
                clear_output()
    
            # early stop
        
            if len(history['Validation loss']) > 1:
                if avg_val_loss < history['Validation loss'][-2]:
                    best_loss = avg_val_loss
                
                else:
                    es += 1
                    # scheduler
                    self.lr_scheduler.step()
            else:
                best_loss = avg_val_loss
                
            if es == esp:
                print("Training finished early")
                if verbose:
                    display(history)
                break
                
                
    def eval(self):
        self.net.eval()
        predicted, target = [],[]
        for images, labels in self.test_loader:
            with torch.no_grad():
                _, ans = torch.max(self.net(images.to(self.device)), 1)
            predicted.extend(list(ans.cpu()))
            target.extend(list(labels.cpu()))
        predicted = np.array(predicted)
        sns.heatmap(confusion_matrix(target, predicted), annot=True, fmt='d',
                cmap='summer', xticklabels=self.classes, yticklabels=self.classes);
        print(', '.join([f'Accuracy: {accuracy_score(target,predicted)}',
                      f'Precision: {precision_score(target,predicted)}',
                      f'Recall: {recall_score(target, predicted)}']))
