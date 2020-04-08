import torch
import torch.nn as nn

class Trainer:
    def __init__(self, 
                model, 
                train_loader, 
                val_loader,
                optim, 
                criterion, 
                scheduler,
                logger, 
                num_epochs,
                gpu, 
                log_interval=10, 
                start_epoch=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger
        self.num_epochs = num_epochs 
        self.epoch = start_epoch
        self.log_interval = log_interval
        self.gpu = gpu

        self.logger.log("Trainer init!")

        self.log_iter = 0

    def run_train(self):
        loss_sum = 0.0
        total = 0.0
        correct = 0.0
        self.model.train()
        for step, data in enumerate(self.train_loader):
            image_batch, labels_batch = data
            if self.gpu:
                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()

            predict = self.model(image_batch)

            #accuracy
            _, predicted = torch.max(predict.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            #loss
            loss = self.criterion(predict, labels_batch)
            loss_sum += loss.cpu().data.numpy().item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step == 0):
                pass
            elif (step % (self.log_interval-1)) == 0:
                loss_mean = loss_sum / self.log_interval
                accuracy = 100 * correct / total
                self.logger(**{"LossMean/train": loss_mean, "Accuracy/train": accuracy, "Log iter": self.log_iter})
                self.log_iter += 1
                loss_sum = 0.0

        self.epoch += 1

    @torch.no_grad()
    def run_eval(self):
        loss_sum = 0.0
        total = 0.0
        correct = 0.0
        self.model.eval()
        for step, data in enumerate(self.val_loader):
            image_batch, labels_batch = data

            if self.gpu:
                image_batch = image_batch.cuda()
                labels_batch = labels_batch.cuda()

            predict = self.model(image_batch)

            #accuracy
            _, predicted = torch.max(predict.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            #loss
            loss = self.criterion(predict, labels_batch)
            loss_sum += loss.cpu().data.numpy().item()

        loss_mean = loss_sum / (step+1)
        accuracy = 100 * correct / total
        self.logger(**{"LossMean/val": loss_mean, "Accuracy/val": accuracy, "Log iter": (self.epoch - 1)})
        self.scheduler.step()
        torch.save(self.model.module.state_dict(),
                    f"checkpoints/vgg16_bn_epoch_{self.epoch - 1}_loss_{loss_mean:.5f}_accuracy_{accuracy:.3f}.pth")
        self.logger.log(f"Model saved with loss {loss_mean:.5f} and accuracy {accuracy:.3f}")

    