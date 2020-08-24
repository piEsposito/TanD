import torch
import numpy as np

class Trainer:
    def __init__(self,
                 model,
                 device,
                 criterion,
                 ):

        self.model = model
        self.device = device
        self.criterion = criterion
        self.metric = None

    def train(self,
              dataloader_train,
              dataloader_test,
              epochs,
              log_every,
              task="classification"):

        iteration = 0

        for epoch in range(epochs):
            for i, (datapoints, labels) in enumerate(dataloader_train):
                self.model.optimizer.zero_grad()
                preds = self.model(datapoints.to(self.device))
                if task == "classification":
                    loss = self.criterion(preds, labels.to(self.device).long())
                if task == "regression":
                    loss = self.criterion(preds, labels.reshape(preds.shape).to(self.device))

                loss.backward()
                self.model.optimizer.step()

                iteration += 1
                if iteration % log_every == 0:
                    correct = 0
                    total = 0

                    if task == "classification":
                        with torch.no_grad():
                            for data in dataloader_test:
                                features, labels = data
                                outputs = self.model(features.to(self.device))
                                _, predicted = torch.max(outputs.data, 1)
                                total += labels.size(0)
                                correct += (predicted == labels.to(self.device)).sum().item()

                            acc = 100 * correct / total
                        self.metric = acc
                        print(f"Iteration: {str(iteration)} | Accuracy of the network on the test dataset: {acc:.2f} %")

                    if task == "regression":
                        with torch.no_grad():
                            mse_list = []
                            for data in dataloader_test:
                                features, labels = data
                                outputs = self.model(features.to(self.device))
                                mse_list.append(self.criterion(outputs, labels.reshape(outputs.shape).to(self.device)).cpu().item())
                            mse = np.array(mse_list).mean()
                        self.metric = mse
                        print(f"Iteration: {str(iteration)} | MSE of the network on the test dataset: {mse:.4f}")

