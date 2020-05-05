import torch
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import dump

from data_manager import DataManager
from classifiers import ConvNN, Vgg16Features

print(f'Cuda available: {torch.cuda.is_available()}')


def log_statistics(writer, epoch_number, index, dataset_size, train_loss, train_accuracy, test_loss, test_accuracy):
    print(
        f'[TIME]: Epoch: {epoch_number}, Index: {index} \n'
        f'[TRAIN]: Loss: {train_loss} , Accuracy: {train_accuracy} \n'
        f'[TEST]: Loss: {test_loss} , Accuracy: {test_accuracy} \n'
        f'-----------------------------------------\n'
    )

    position = epoch_number * dataset_size + index
    writer.add_scalar('Train/Loss', train_loss, position)
    writer.add_scalar('Train/Acc', train_accuracy, position)
    writer.add_scalar('Test/Loss', test_loss, position)
    writer.add_scalar('Test/Acc', test_accuracy, position)


def train_net(model, data_manager, epochs=20):
    def test_net(_model, _criterion, _loader):
        _model.eval()

        with torch.no_grad():
            test_loss = 0
            test_accuracy = 0
            for test_x, test_y in _loader:
                if torch.cuda.is_available():
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()

                test_out = _model(test_x)
                test_loss = _criterion(test_out, test_y)
                _, test_pred = torch.max(test_out.data, 1)

                test_loss += test_loss.item()
                test_accuracy += test_pred.eq(test_y).sum().item() / test_y.size(0)

            test_dataset_size = len(_loader)
            test_loss /= test_dataset_size
            test_accuracy /= test_dataset_size

        _model.train()
        return test_loss, test_accuracy

    writer = SummaryWriter(f'./logs/ConvNN-{datetime.datetime.now()}')

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    train_loader, test_loader = data_manager.get_net_train_test_loaders()

    model.train()
    print(f'Started learn ConvNN {datetime.datetime.now()}')

    for epoch_number in range(epochs):
        for index, (train_x, train_y) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_x, train_y = train_x.cuda(), train_y.cuda()

            optimizer.zero_grad()

            train_out = model(train_x)
            train_loss = criterion(train_out, train_y)
            _, train_pred = torch.max(train_out.data, 1)

            train_loss.backward()
            optimizer.step()

            train_accuracy = train_pred.eq(train_y).sum().item() / train_y.size(0)

            if index % 1 == 0:
                test_loss, test_accuracy = test_net(model, criterion, test_loader)
                log_statistics(
                    writer, epoch_number, index, len(train_loader), train_loss.item(),
                    train_accuracy, test_loss, test_accuracy
                )

    def save_score(_model):
        with open('results/ConvNN-result.txt', 'w') as file:
            score_loader = data_manager.get_net_score_loader()

            if torch.cuda.is_available():
                _model.cuda()

            _model.eval()
            with torch.no_grad():
                for x, _, file_name in score_loader:
                    x = x.cuda()
                    out = _model(x)

                    _, pred = torch.max(out.data, 1)
                    lines = [f'{file_name[i].split("/")[-1]} {pred[i].item()}\n' for i in range(x.size(0))]
                    file.writelines(lines)

            _model.train()

    print(f'Finished learn ConvNN {datetime.datetime.now()}')
    torch.save(model.state_dict(), f'models/ConvNN.model')
    print('ConvNN saved. Saving score...')
    save_score(model)
    print('Score saved')


def train_svm(model, data_manager):
    (x_train, y_train), (x_test, y_test) = data_manager.get_svm_train_test_data()

    print(f'Started learn SVM {datetime.datetime.now()}')
    model.fit(x_train, y_train)
    print(f'Finished learn SVM {datetime.datetime.now()}')

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    def save_score(_model):
        with open('results/SVM-result.txt', 'w') as file:
            score_data = data_manager.get_svm_score_data()
            x, file_names = score_data
            pred = _model.predict(x)
            lines = [f'{file_names[i].split("/")[-1]} {int(pred[i].item())}\n' for i in range(x.shape[0])]
            file.writelines(lines)

    print(f'SVM accuracy: {accuracy}')
    dump(model, 'models/SVM.joblib')
    print('SVM saved. Saving score...')
    save_score(model)
    print('Score saved')


def prepare_data_with_vgg(loader, vgg):
    if torch.cuda.is_available():
        vgg.cuda()
    vgg.eval()
    with torch.no_grad():
        dataset_size = len(loader.dataset)
        result_x, result_y = np.zeros((dataset_size, vgg.features_count)), np.zeros(dataset_size)
        for idx, (x, y) in enumerate(loader):
            if torch.cuda.is_available():
                x = x.cuda()
            out = vgg(x)
            result_x[idx * out.size(0): (idx + 1) * out.size(0)][:] = out.cpu()
            result_y[idx * y.size(0): (idx + 1) * y.size(0)][:] = y
        return result_x, result_y


def train_svm_with_vgg(model, data_manager):
    train_loader, test_loader = data_manager.get_net_train_test_loaders()

    vgg = Vgg16Features()
    x_test, y_test = prepare_data_with_vgg(test_loader, vgg)
    x_train, y_train = prepare_data_with_vgg(train_loader, vgg)

    print(f'Started learn SVM with VGG16 {datetime.datetime.now()}')
    model.fit(x_train, y_train)
    print(f'Finished learn SVM with VGG16 {datetime.datetime.now()}')

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    def save_score(_vgg, _model):
        with open('results/SVM-with-VGG16-result.txt', 'w') as file:
            score_loader = data_manager.get_net_score_loader()

            if torch.cuda.is_available():
                _vgg.cuda()

            _vgg.eval()
            with torch.no_grad():
                for x, _, file_name in score_loader:
                    if torch.cuda.is_available():
                        x = x.cuda()

                    out = _vgg(x)
                    pred = _model.predict(out.cpu())

                    lines = [f'{file_name[i].split("/")[-1]} {pred[i].astype(int)}\n' for i in range(x.size(0))]
                    file.writelines(lines)

    print(f'SVM with VGG16 accuracy: {accuracy}')
    dump(model, 'models/SVM_with_VGG16.joblib')
    print('SVM with VGG16 saved. Saving score...')
    save_score(vgg, model)
    print('Score saved')


if __name__ == '__main__':
    data_manager = DataManager()

    # conv_nn = ConvNN()
    # train_net(conv_nn, data_manager)
    #
    # svm = SVC(kernel='linear', probability=True, random_state=17)
    # train_svm(svm, data_manager)

    svm_for_vgg = SVC(kernel='linear', probability=True, random_state=17, tol=1e-4)
    train_svm_with_vgg(svm_for_vgg, data_manager)
