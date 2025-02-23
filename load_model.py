import torch
from utils import utils
import logging, coloredlogs
import numpy as np
import torch.nn.functional as F

def load_initial_labels(path):
    labels = torch.load(path + '/label.pt')
    # print(labels)
    return labels

def load_labels_trained_on_iteration(path, iteration):
    labels = torch.load(path + '/trained_clients_before_aggregation_labels_' + str(iteration) + '.pt')
    # print(labels)
    return labels

def load_labels_selected_on_iteration(path, iteration):
    labels = torch.load(path + '/selected_clients_before_aggregation_labels_' + str(iteration) + '.pt')
    # print(labels)
    return labels

def load_clients_deltas_trained_on_iteration(path, iteration):
    deltas_recovered = torch.load(path + '/trained_clients_before_aggregation_values_' + str(iteration) + '.pt')
    # print(deltas_recovered[0]["fc3.bias"])
    return deltas_recovered

def load_clients_deltas_selected_on_iteration(path, iteration):
    deltas_recovered = torch.load(path + '/selected_clients_before_aggregation_values_' + str(iteration) + '.pt')
    # print(deltas_recovered[0]["fc3.bias"])
    return deltas_recovered

def load_aggregated_model_on_iteration(path, dataset, iteration, device, model):
    model.to(device)
    model.load_state_dict(torch.load(path + '/aggregated_model_' + str(iteration) + '.pth', weights_only=True))
    model.to(device)
    model.eval()
    return model

def load_dataset_info(dataset, num_clients, loader_type, loader_path, test_batch_size):
    if dataset == 'mnist':
        from tasks import mnist
        trainData = mnist.train_dataloader(num_clients,
            loader_type=loader_type, path=loader_path, store=False)
        testData = mnist.test_dataloader(test_batch_size)
        Net = mnist.Net
        criterion = F.cross_entropy
    elif dataset == 'cifar':
        from tasks import cifar10 as cifar
        trainData = cifar.train_dataloader(num_clients,
            loader_type=loader_type, path=loader_path, store=False)
        testData = cifar.test_dataloader(test_batch_size)
        Net = cifar.Net
        criterion = F.cross_entropy
    elif dataset == 'cifar100':
        from tasks import cifar100
        trainData = cifar100.train_dataloader(num_clients,
            loader_type=loader_type, path=loader_path, store=False)
        testData = cifar100.test_dataloader(test_batch_size)
        Net = cifar100.Net
        criterion = F.cross_entropy
    elif dataset == 'imdb':
        from tasks import imdb
        trainData = imdb.train_dataloader(num_clients,
            loader_type=loader_type, path=loader_path, store=False)
        testData = imdb.test_dataloader(test_batch_size)
        Net = imdb.Net
        criterion = F.cross_entropy
    elif dataset == 'fashion_mnist':
        from tasks import fashion_mnist
        trainData = fashion_mnist.train_dataloader(num_clients,
            loader_type=loader_type, path=loader_path, store=False)
        testData = fashion_mnist.test_dataloader(test_batch_size)
        Net = fashion_mnist.Net
        criterion = F.cross_entropy

    model = Net()
    return model, trainData, testData, criterion

def test(model, iteration, dataLoader, criterion):
    logging.info("[Server] Start testing")
    device = "cuda"
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    count = 0
    nb_classes = 10 # for MNIST, Fashion-MNIST, CIFAR-10
    cf_matrix = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for data, target in dataLoader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss
            if output.dim() == 1:
                pred = torch.round(torch.sigmoid(output))
            else:
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += pred.shape[0]
            for t, p in zip(target.view(-1), pred.view(-1)):
                cf_matrix[t.long(), p.long()] += 1
    test_loss /= count
    accuracy = 100. * correct / count
    model.cpu()  ## avoid occupying gpu when idle
    logging.info(
        '[Server] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({}%)\n'.format(
            test_loss, correct, count, accuracy))
    logging.info(f"[Sever] Confusion matrix:\n {cf_matrix.detach().cpu()}")
    cf_matrix = cf_matrix.detach().cpu().numpy()
    row_sum = np.sum(cf_matrix, axis=0) # predicted counts
    col_sum = np.sum(cf_matrix, axis=1) # targeted counts
    diag = np.diag(cf_matrix)
    precision = diag / row_sum # tp/(tp+fp), p is predicted positive.
    recall = diag / col_sum # tp/(tp+fn)
    f1 = 2*(precision*recall)/(precision+recall)
    m_acc = np.sum(diag)/np.sum(cf_matrix)
    results = {'accuracy':accuracy,'test_loss':test_loss,
                'precision':precision.tolist(),'recall':recall.tolist(),
                'f1':f1.tolist(),'confusion':cf_matrix.tolist(),
                'epoch':iteration}
    logging.info(f"[Server] Precision={precision},\n Recall={recall},\n F1-score={f1},\n my_accuracy={m_acc*100.}[%]")

    return test_loss, accuracy

local_folder = "AggData"
experiment = 'Exp_100C_40ep_15start_5_mnist_mudhog_40C_8MLF'
dataset = "mnist"
loader_type = "dirichlet"
aggregation_method = "mudhog"
num_clients = 40
device = torch.device("cuda")
iteration = 3
loader_path = "./data/loader.pk"
test_batch_size = 64

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

path = f"./{local_folder}/{loader_type}/{dataset}/{experiment}/{aggregation_method}"

load_initial_labels(path)

selected_clients_labels = load_labels_selected_on_iteration(path, iteration)
selected_clients = load_clients_deltas_selected_on_iteration(path, iteration)

trained_clients_labels = load_labels_trained_on_iteration(path, iteration)
trained_clients = load_clients_deltas_trained_on_iteration(path, iteration)

model, trainData, testData, criterion = load_dataset_info(dataset, num_clients, loader_type, loader_path, test_batch_size)

aggregated_model = load_aggregated_model_on_iteration(path, dataset, iteration, device, model)

test(model, iteration, testData, criterion)