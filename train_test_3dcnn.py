from config import get_cfg_defaults
from utils.dataloader import DataKAT
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader
import numpy as np
import copy
from utils.reproduce import set_seed
from models.alexnet import AlexNet
from models.vgg import VGG16
from models.inception import InceptionV4

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    config_file = rf'configs/exp1.yaml'
    model_name = "alex"  # "alex" "vgg" "inception"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(config_file)
    cfg.freeze()

    gen_train = DataKAT(cfg)

    condition = 4
    num_epochs = 150
    n_tmax = 3

    acc_list = []
    seed_list = [0, 9, 666, 700, 800, 1000, 2000, 2023, 2028, 5000]

    for iSeed in seed_list:
        seed = iSeed
        set_seed(seed)

        x_data, y_label = gen_train.get_data(condition)

        x_tr = torch.tensor(x_data)
        x_tr = x_tr.permute(4, 2, 0, 1, 3)
        y_tr = torch.LongTensor(y_label)
        dataset = torch.utils.data.TensorDataset(x_tr, y_tr)

        dataset_size = len(dataset)
        shuffle_dataset = True
        train_ratio = cfg.data.train_ratio
        test_ratio = 1 - train_ratio
        train_num = int(np.floor(train_ratio * dataset_size))
        test_num = int(np.floor(test_ratio * dataset_size))
        indices = list(range(dataset_size))
        if shuffle_dataset:
            set_seed(seed)
            np.random.shuffle(indices)
        train_indices = indices[0:train_num]
        test_indices = indices[train_num:]

        # Creating data samplers and loaders:
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset,
                                  batch_size=cfg.params.batch_size,
                                  sampler=train_sampler, )

        test_loader = DataLoader(dataset,
                                 batch_size=cfg.params.batch_size,
                                 sampler=test_sampler,
                                 )

        criteria = nn.CrossEntropyLoss()
        if model_name == "alex":
            classifier = AlexNet(4).cuda()
        elif model_name == "vgg":
            classifier = VGG16(4).cuda()
        elif model_name == "inception":
            classifier = InceptionV4(4).cuda()

        min_loss = 10000
        best_epoch = 1
        learning_rate = 3e-2  # myself 5e-3
        optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=num_epochs // n_tmax,
                                                               eta_min=3e-5)

        classifier.to(device)

        best_model = None

        for iEpoch in range(num_epochs):
            losses = []
            val_losses = []
            test_losses = []

            # Train process
            classifier.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = classifier(inputs)
                loss = criteria(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                lr = scheduler.get_last_lr()
                losses.append(loss.cpu().item())

            # Validation process
            classifier.eval()
            with torch.no_grad():
                for iVal, (inputs_val, labels_val) in enumerate(test_loader):
                    inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                    outputs_val = classifier(inputs_val)

                    loss_val = criteria(outputs_val, labels_val)
                    val_losses.append(loss_val.cpu().item())
                val_loss = sum(val_losses) / (test_num)
                if val_loss < min_loss:
                    min_loss = val_loss
                    best_epoch = iEpoch
                    best_model = copy.deepcopy(classifier)
                    model_path = f'checkpoints/{model_name}_best_{seed}_{condition}_{train_ratio}.pt'
                    torch.save(best_model.state_dict(), model_path)
                print(
                    '[epoch %d] %s loss: %f min loss: %f at epoch %d ' %
                    (iEpoch, 'val', val_loss, min_loss, best_epoch))

        # Test
        best_model.eval()
        with torch.no_grad():
            test_correct_num = 0
            total = 0
            for iTest, (inputs_test, labels_test) in enumerate(test_loader):
                inputs_test, labels_test = inputs_test.to(device), labels_test.to(device)
                outputs_test = best_model(inputs_test)

                _, pred_test = torch.max(outputs_test, 1)
                total += labels_test.size(0)
                test_correct_num += (pred_test == labels_test).sum().item()

            print('Seed: {}, Test Acc: {:.2f} %'.format(seed,
                                                        100 * test_correct_num / total))
        acc_i = 100 * test_correct_num / total
        acc_list.append(acc_i)

    print(acc_list)
    print('Mean: {:.2f}, Std: {:.2f}'.format(np.mean(acc_list), np.std(acc_list)))
