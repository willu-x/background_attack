from torch import nn
from collections import defaultdict
import copy
import os
import time
import random
import numpy as np
import torch
from torchvision import datasets, transforms
from models.resnet import ResNet18
from models.lenet import LeNet
from torch.optim import Adam
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


torch.manual_seed(1)
torch.cuda.manual_seed(1)
log_route = ''
random.seed(0)
writer = None
np.random.seed(0)

# 读取EMINIST DIGITS数据集
def load_Non_IID_clean_data(number_of_total_participants, dirichlet_alpha, num_workers=0):
    train_set = datasets.EMNIST('./data', split="digits", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.RandomCrop(28, padding=2),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    test_set = datasets.EMNIST('./data', split="digits", train=False, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    indices_per_participant = sample_dirichlet_train_data(dataset=train_set,
                                                          total_client=number_of_total_participants, alpha=dirichlet_alpha)

    train_loaders = [get_train(train_set, indices, batch_size=64, num_workers=num_workers) for pos, indices in
                     indices_per_participant.items()]

    test_loader = get_test(test_set, test_batch_size=100,
                           num_workers=num_workers)
    print('load data succeeded')
    print('init Tensor Board succeeded')
    return train_loaders, test_loader

def initTensorBoard(route_):
    start_time = time.time()
    global log_route
    global writer
    log_route = './tblog/' + route_
    writer = SummaryWriter(log_route)
    return writer

def sample_dirichlet_train_data(dataset, total_client, alpha=0.9):
    """
        Input: Number of participants and alpha (param for distribution)
            
        Requires: cifar_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """

    cifar_classes = {}
    for ind, data in enumerate(dataset):
        x, y = data

        if y in cifar_classes:
            cifar_classes[y].append(ind)
        else:
            cifar_classes[y] = [ind]
    class_size = len(cifar_classes[0])
    per_client_list = defaultdict(list)
    no_classes = len(cifar_classes.keys())

    for n in range(no_classes):
        # cifar_classes 字典，key=类别，value=[]，包含对应类的数据下标
        random.shuffle(cifar_classes[n])
        sampled_probabilities = class_size * np.random.dirichlet(  # sampled_probabilities 采样概率 len(sampled_probabilities)：1000,加和=5000
            np.array(total_client * [alpha]))  # np.array(total_client * [alpha]) 1000维数组，value=0.9
        for user in range(total_client):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = cifar_classes[n][:min(
                len(cifar_classes[n]), no_imgs)]
            per_client_list[user].extend(sampled_list)
            cifar_classes[n] = cifar_classes[n][min(
                len(cifar_classes[n]), no_imgs):]

    return per_client_list


def sample_iid_train_data(dataset, total_client):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/total_client)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(total_client):
        dict_users[i] = set(np.random.choice(
            all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        dict_users[i] = list(dict_users[i])
    return dict_users


def get_train(train_set, indices, batch_size, num_workers=0):
    """
        This method is used along with Dirichlet distribution
        :param params:
        :param indices:
        :return:
        """
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                   indices),
                                               num_workers=num_workers)
    return train_loader


def get_test(test_set, test_batch_size, num_workers=0):

    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              num_workers=4)

    return test_loader


def train_cv(target_model, local_model, epoch, criterion, sampled_participants, lr_init=0.001, traget_lr=0.2, train_loaders=None):
    total_benign_l2_norm = 0
    total_benign_l2_norm_after_clip = 0
    s_norm = 0.6
    ### Accumulate weights for all participants.
    # target_model=glabol_model
    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(data)

    global_model_copy = dict()
    for name, param in target_model.named_parameters():
        # for name, param in target_model.state_dict().items():
        global_model_copy[name] = target_model.state_dict(
        )[name].clone().detach().requires_grad_(False)

    total_benign_train_loss = 0

    for participant_id in sampled_participants:
        # local_model 在训练过程中被改变了？虽然使用copy_params，但是有部分参数没有被改变
        # 因为copy_params使用named_parameters()
        # local_model还在使用上一轮参与者的BN层中的部分参数
        model = local_model
        copy_params(model, global_model_copy)
        #-------------------------test-------------------------
        # for name, param in target_model.state_dict().items():
        #     if not torch.allclose(model.state_dict()[name], param):
        #         print(name)
        model.train()
        ## update lr
        lr = lr_init
        if epoch <= 500:
            lr = epoch*(traget_lr - lr_init)/499.0 + \
                lr_init - (traget_lr - lr_init)/499.0
        else:
            lr = epoch*(-traget_lr)/1500 + traget_lr*4.0/3.0

        if lr <= 0.0001:
            lr = 0.0001
        # lr = 0.001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        l2_test1,l2_test1_np = get_l2_norm(global_model_copy, model.named_parameters())
        for internal_epoch in range(2):
            loss = local_train(model, optimizer, criterion,
                               train_loaders[participant_id])
            # loss = local_train(target_model, optimizer, criterion, train_loaders[participant_id])
        l2_test2,l2_test2_np = get_l2_norm(global_model_copy, model.named_parameters())
        for name, data in model.state_dict().items():
            # data - target_model.state_dict()[name] 计算和上一轮全局模型间的差距
            weight_accumulator[name].add_(data - target_model.state_dict()[name])

        weight_difference, difference_flat = get_weight_difference(global_model_copy, model.named_parameters())
        # l2_norm表示局部模型与全局模型的差异，这里的l2_norm很大，表示局部模型异常
        clipped_weight_difference, l2_norm = clip_grad(s_norm, weight_difference, difference_flat)
        weight_difference, difference_flat = get_weight_difference(global_model_copy, clipped_weight_difference)
        copy_params(model, weight_difference)  # ？？？将模型参数更新为差值
        l2_norm_after_clip, l2_norm_np = get_l2_norm(global_model_copy, model.named_parameters())
        total_benign_l2_norm_after_clip += l2_norm_after_clip.item()
        total_benign_l2_norm += l2_norm.item()
    print('l2 norm of benign user before server clipping',
          total_benign_l2_norm / (len(sampled_participants)))
    print('l2 norm of benign user after server clipping',
          total_benign_l2_norm_after_clip / (len(sampled_participants)))
    return weight_accumulator


def copy_params(model, target_params_variables):
    for name, layer in model.named_parameters():
        layer.data = copy.deepcopy(target_params_variables[name])
# def copy_params(model, target_params_variables):
#     for name in model.state_dict():
#         model.state_dict()[name].copy_(copy.deepcopy(target_params_variables[name]))


def local_train(model, optimizer, criterion, train_loader):

    # test_cv(model=model, epoch=2000, data_source=train_loader)
    for inputs, labels in train_loader:
        inputs, labels = inputs.cuda(), labels.cuda()
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
    return loss


def gen_model_grad(model, criterion, train_loaders):
    data = []
    target = []
    for i in range(len(train_loaders)):
        for inputs, labels in train_loaders[i]:
            data.append(inputs)
            target.append(labels)
    data = torch.cat(data, dim=0)
    target = torch.cat(target, dim=0)
    data, target = data.cuda(), target.cuda()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()


def average_shrink_models(weight_accumulator, target_model, epoch):
    """
    Perform FedAvg algorithm and perform some clustering on top of it.

    """
    lr = 1  # 联邦平均

    for name, data in target_model.state_dict().items():
        update_per_layer = weight_accumulator[name] * (1/10) * lr
        update_per_layer = torch.tensor(update_per_layer, dtype=data.dtype)

        data.add_(update_per_layer.cuda())  # 这里保留了上一轮聚合的模型

    return True


def test_cv(epoch, data_source,
            model):
    model.eval()
    total_loss = 0
    correct = 0
    total_test_words = 0

    data_iterator = data_source
    num_data = 0.0
    for batch_id, batch in enumerate(data_iterator):
        data, targets = batch
        data, targets = data.cuda(), targets.cuda()

        output = model(data)
        total_loss += nn.functional.cross_entropy(output, targets,
                                                  reduction='sum').item()  # sum up batch loss
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()
        num_data += output.size(0)

    acc = 100.0 * (float(correct) / float(num_data))
    total_l = total_loss / float(num_data)

    writer.add_scalar('Main Task Accuracy', acc, epoch)
    print('___Test : epoch: {}: Average loss: {:.4f}, '
          'Accuracy: {}/{} ({:.4f}%)'.format(epoch,
                                             total_l, correct, num_data,
                                             acc))

    model.train()
    return total_l, acc


def test(net, testloader, epoch):
    device = torch.device('cuda')
    net.eval()
    total_correct = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
    # 输出测试结果
    print(
        f'Epoch: {epoch},Test Accuracy:{100 * total_correct / len(testloader.dataset)}')


def load_model(path='./cifar10_resnet_Snorm_1_checkpoint_model_epoch_1999.pth'):
    print(path)
    num_classes = 10

    model = ResNet18(num_classes=num_classes)
    model.load_state_dict(torch.load(path))
    model.cuda()
    return model

def load_lenet_model(path='./cifar10_resnet_Snorm_1_checkpoint_model_epoch_1999.pth'):
    print(path)
    # num_classes = 10
    model = LeNet()
    model.load_state_dict(torch.load(path))
    model.cuda()
    return model

def grad_mask_cv(model, dataset_clean, criterion, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""
    model.train()
    model.zero_grad()

    # dataset_clean[]:dataloader对象
    for participant_id in range(len(dataset_clean)):

        train_data = dataset_clean[participant_id]

        for inputs, labels in train_data:
            inputs, labels = inputs.cuda(), labels.cuda()

            output = model(inputs)

            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

    mask_grad_list = []

    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            #grad_list使用上面良性数据所累积的梯度,parms.grad也是一个多维tensor
            grad_list.append(parms.grad.abs().view(-1))
            #某一层的梯度绝对值总和
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            k_layer += 1

    grad_list = torch.cat(grad_list).cuda()
    # len(grad_list):2797610
    #将累计的梯度中绝对值最小的参数坐标取出
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    # len(indices):2517849 (mask=0.9)
    mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            #mask_flat:使用mask_flat_all_layer来记录每一层的参数（由0、1组成）
            mask_flat = mask_flat_all_layer[count:count +
                                            gradients_length].cuda()
            mask_grad_list.append(mask_flat.reshape(parms.grad.size()).cuda())
            #mask_grad_list[0].shape:  torch.Size([32, 3, 3, 3])

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            # grad_abs_percentage_list求某一层梯度占比
            grad_abs_percentage_list.append(
                grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()
    return mask_grad_list  # mask_grad_list:列表，每一个元素对应每一层中应该被替换的参数（良性设备不经常访问的）


    """Generate a gradient mask based on the given dataset"""
    model.train()
    model.zero_grad()

    # dataset_clean[]:dataloader对象
    for participant_id in range(len(dataset_clean)):

        train_data = dataset_clean[participant_id]

        for inputs, labels in train_data:
            inputs, labels = inputs.cuda(), labels.cuda()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward(retain_graph=True)

    mask_grad_list = []

    grad_list = []
    grad_abs_sum_list = []
    indices = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            #grad_list使用上面良性数据所累积的梯度,parms.grad也是一个多维tensor
            parms_grad = parms.grad.abs().view(-1)
            grad_list.append(parms_grad)
            # 获取每一层中梯度topk%的参数坐标
            _,layer_indices_top = torch.topk(-1*parms_grad, int(len(parms_grad)*ratio))
            indices.append(layer_indices_top)
            k_layer += 1

    grad_list = torch.cat(grad_list).cuda()
    # len(grad_list):2797610
    #将累计的梯度中绝对值最小的参数坐标取出
    # _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    # len(indices):2517849 (mask=0.9)
    mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            #mask_flat:使用mask_flat_all_layer来记录每一层的参数（由0、1组成）
            mask_flat = mask_flat_all_layer[count:count +
                                            gradients_length].cuda()
            mask_grad_list[name] = (
                mask_flat.reshape(parms.grad.size()).cuda())
            #mask_grad_list[0].shape:  torch.Size([32, 3, 3, 3])

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            # grad_abs_percentage_list求某一层梯度占比
            grad_abs_percentage_list.append(
                grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1
    model.zero_grad()
    return mask_grad_list  # mask_grad_list:列表，每一个元素对应每一层中应该被替换的参数（良性设备不经常访问的）

def xzy_grad_mask_cv(model, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""

    mask_grad_list = {}
    grad_list = []
    grad_abs_sum_list = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            #grad_list使用上面良性数据所累积的梯度,parms.grad也是一个多维tensor
            grad_list.append(parms.grad.abs().view(-1))
            #某一层的梯度绝对值总和
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())

            k_layer += 1

    grad_list = torch.cat(grad_list).cuda()
    # len(grad_list):2797610
    
    print('Total Count of Neuro',int(len(grad_list)))
    print('Selected Count of Neuro',int(len(grad_list)*ratio))
    #将累计的梯度中绝对值最小的参数坐标取出
    _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    # len(indices):2517849 (mask=0.9)
    mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            #mask_flat:使用mask_flat_all_layer来记录每一层的参数（由0、1组成）
            mask_flat = mask_flat_all_layer[count:count +
                                            gradients_length].cuda()
            mask_grad_list[name] = (
                mask_flat.reshape(parms.grad.size()).cuda())
            #mask_grad_list[0].shape:  torch.Size([32, 3, 3, 3])

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            # grad_abs_percentage_list求某一层梯度占比
            grad_abs_percentage_list.append(
                grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()
    return mask_grad_list  # mask_grad_list:列表，每一个元素对应每一层中应该被替换的参数（良性设备不经常访问的）

def xzy_grad_mask_cv_by_layer(model, ratio=0.5):
    """Generate a gradient mask based on the given dataset"""

    mask_grad_list = {}
    grad_list = []
    grad_abs_sum_list = []
    indices = []
    k_layer = 0
    for _, parms in model.named_parameters():
        if parms.requires_grad:
            #grad_list使用上面良性数据所累积的梯度,parms.grad也是一个多维tensor
            parms_grad = parms.grad.abs().view(-1)
            grad_list.append(parms_grad)
            # 获取每一层中梯度topk%的参数坐标
            _,layer_indices_top = torch.topk(-1*parms_grad, int(len(parms_grad)*ratio))
            indices.append(layer_indices_top)
            grad_abs_sum_list.append(parms.grad.abs().view(-1).sum().item())
            k_layer += 1

    grad_list = torch.cat(grad_list).cuda()
    # len(grad_list):2797610
    indices = torch.cat(indices).cuda()
    print('Total Count of Neuro',int(len(grad_list)))
    print('Selected Count of Neuro',int(len(indices)))
    # #将累计的梯度中绝对值最小的参数坐标取出
    # _, indices = torch.topk(-1*grad_list, int(len(grad_list)*ratio))
    
    # len(indices):2517849 (mask=0.9)
    mask_flat_all_layer = torch.zeros(len(grad_list)).cuda()
    mask_flat_all_layer[indices] = 1.0

    count = 0
    percentage_mask_list = []
    k_layer = 0
    grad_abs_percentage_list = []
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            gradients_length = len(parms.grad.abs().view(-1))
            #mask_flat:使用mask_flat_all_layer来记录每一层的参数（由0、1组成）
            mask_flat = mask_flat_all_layer[count:count + gradients_length].cuda()
            mask_grad_list[name] = (
                mask_flat.reshape(parms.grad.size()).cuda())
            #mask_grad_list[0].shape:  torch.Size([32, 3, 3, 3])

            count += gradients_length

            percentage_mask1 = mask_flat.sum().item()/float(gradients_length)*100.0

            percentage_mask_list.append(percentage_mask1)

            # grad_abs_percentage_list求某一层梯度占比
            grad_abs_percentage_list.append(
                grad_abs_sum_list[k_layer]/np.sum(grad_abs_sum_list))

            k_layer += 1

    model.zero_grad()
    return mask_grad_list  # mask_grad_list:列表，每一个元素对应每一层中应该被替换的参数（良性设备不经常访问的）


def save_model(file_name=None, model=None, epoch=None, new_folder_name='saved_models'):
    if new_folder_name is None:
        new_folder_name = '.'
    else:
        new_folder_name = f'./{new_folder_name}'
        if not os.path.exists(new_folder_name):
            os.mkdir(new_folder_name)
    filename = "%s/%s_model_epoch_%s.pth" % (new_folder_name, file_name, epoch)
    torch.save(model.state_dict(), filename)


outputs = []


def hook_fn(module, input, output):
    # module_name = module.__class__.__name__

    global outputs
    outputs.append(output)
    # print(output[0][0][0])

# def Generate_trigger0(model, mask_grad_list):
#     #根据model和mask_grad_list生成trigger
#     #return:trigger

#     #初始化trigger

#     target_layer_neuron = {}
#     dic_modules = dict(model.named_modules())
#     for name, params in model.named_parameters():
#         indices = torch.where(mask_grad_list[name] == 1)
#         # 为每个有梯度掩码的层注册一个forward hook函数
#         if indices[0].shape != torch.Size([0]):
#             target_layer_neuron[name] = (indices)
#             #为该层注册hook
#             dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

#     model_input = torch.rand(1, 3, 32, 32).cuda()
#     optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
#     optimization_mask[:, :, 29:32, 29:32] = True
#     num_input_params = int(optimization_mask.sum().item())
#     trigger = torch.rand(num_input_params).cuda()
#     trigger.requires_grad = True
#     optim = Adam([trigger], lr=1)
#     scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

#     for i in range(1000):
#         outputs.clear()
#         x = model_input.clone()
#         x[optimization_mask] = trigger
#         y = model(x)
#         loss = 0
#         target_outputs = []
#         for j in range(len(outputs)):
#             target_outputs.append(outputs[j][0][target_layer_neuron[list(
#                 target_layer_neuron.keys())[j]][0].unique()])

#         for j in range(len(target_outputs)):
#             loss += (torch.mean(target_outputs[j])-3)**2

#         # loss=torch.mean((outputs[0][0][0][0]-3)**2)
#         # loss=(outputs[0][0][0][0][0]-3)**2

#         # loss=torch.mean((y-3)**2)
#         optim.zero_grad()
#         loss.backward()
#         optim.step()
#         scheduler.step()
#         if i % 200 == 0:
#             print('Gen trigger0 loss', loss)
#             # print('Gen trigger0 lr',scheduler.get_last_lr())
#     return trigger


def generate_trigger(nameOfTrigger, model, mask_grad_list, locationX, locationY):
    '''
    model             nn.model
    mask_grad_list    dict    containing the gradient mask for each parameter of the model
    locationX         number  the x coordinate of the trigger's location
    locationY         number  the y coordinate of the trigger's location
    Returns   trigger
    '''

    trigger_width = 3
    trigger_height = 3
    graph_width = 28
    target_layer_neuron = {}
    dic_modules = dict(model.named_modules())
    for name, params in model.named_parameters():
        indices = torch.where(mask_grad_list[name] == 1)
        # 为每个有梯度掩码的层注册一个forward hook函数
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    # model_input = torch.rand(1, 3, graph_width, graph_width).cuda()
    model_input = torch.rand(1, 1, graph_width, graph_width).cuda()
    def set_optimization_mask(x_start, y_start):
        # optimization_mask = torch.zeros(1, 3, graph_width, graph_width, dtype=torch.bool).cuda()
        optimization_mask = torch.zeros(1, 1, graph_width, graph_width, dtype=torch.bool).cuda()
        optimization_mask[:, :, x_start:x_start + trigger_width, y_start:y_start + trigger_height] = True
        return optimization_mask
    
    optimization_mask = set_optimization_mask(locationX, locationY)
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    # 将学习率从1调整到0.1
    optim = Adam([trigger], lr=0.2)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(1000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2


        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        writer.add_scalar('trigger_loss/'+nameOfTrigger, loss, i)
        if i % 200 == 0:
            print('Gen %s loss' % (nameOfTrigger), loss)
    return trigger

def generate_dependent_trigger_random_neurons(nameOfTrigger,model,locationX,locationY):
    #根据model生成trigger
    #return:trigger
    trigger_width = 3
    trigger_height = 3
    graph_width = 28
    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        flat = flat.reshape(-1)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 1, graph_width, graph_width).cuda()
    optimization_mask = torch.zeros(1, 1, graph_width, graph_width, dtype=torch.bool).cuda()
    optimization_mask[:, :, locationX:locationX+trigger_width, locationY:locationY+trigger_height] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_'+ nameOfTrigger +'_random_neurons loss', loss)
            # print('Gen trigger1 lr',scheduler.get_last_lr())
    return trigger

def Generate_dependent_trigger0_random_neurons(model):
    #根据model生成trigger
    #return:trigger

    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        flat = flat.reshape(-1)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 3, 32, 32).cuda()
    optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
    optimization_mask[:, :, 29:32, 29:32] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_trigger0_random_neurons loss', loss)
            # print('Gen trigger1 lr',scheduler.get_last_lr())
    return trigger


def Generate_dependent_trigger1_random_neurons(model):
    #根据model生成trigger
    #return:trigger

    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 3, 32, 32).cuda()
    optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
    optimization_mask[:, :, 29:32, 26:29] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_trigger1_random_neurons loss', loss)
            # print('Gen trigger1 lr',scheduler.get_last_lr())
    return trigger


def Generate_dependent_trigger2_random_neurons(model):
    #根据model生成trigger
    #return:trigger

    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 3, 32, 32).cuda()
    optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
    optimization_mask[:, :, 26:29, 26:29] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_trigger2_random_neurons loss', loss)
            # print('Gen trigger2 lr',scheduler.get_last_lr())
    return trigger


def Generate_dependent_trigger3_random_neurons(model):
    #根据model生成trigger
    #return:trigger

    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 3, 32, 32).cuda()
    optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
    optimization_mask[:, :, 26:29, 29:32] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_trigger3_random_neurons loss', loss)
            # print('Gen trigger3 lr',scheduler.get_last_lr())
    return trigger


def Generate_dependent_global_trigger_random_neurons(model):
    #根据model生成trigger
    #return:trigger
    #初始化trigger
    random_neurons = {}
    for name, params in model.named_parameters():
        flat = torch.zeros_like(params)
        num_select = int(len(flat) * 0.005)
        indices = torch.randperm(len(flat))[:num_select]
        flat[indices] = 1
        random_neurons[name] = (flat.reshape(params.size()).cuda())

    dic_modules = dict(model.named_modules())
    target_layer_neuron = {}
    for name, params in model.named_parameters():
        indices = torch.where(random_neurons[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
            #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)

    model_input = torch.rand(1, 3, 32, 32).cuda()
    optimization_mask = torch.zeros(1, 3, 32, 32, dtype=torch.bool).cuda()
    optimization_mask[:, :, 29:32, 29:32] = True
    num_input_params = int(optimization_mask.sum().item())
    trigger = torch.rand(num_input_params).cuda()
    trigger.requires_grad = True
    optim = Adam([trigger], lr=1)
    scheduler = lr_scheduler.StepLR(optim, step_size=2000, gamma=0.1)

    for i in range(2000):
        outputs.clear()
        x = model_input.clone()
        x[optimization_mask] = trigger
        y = model(x)
        loss = 0
        target_outputs = []
        for j in range(len(outputs)):
            target_outputs.append(outputs[j][0][target_layer_neuron[list(
                target_layer_neuron.keys())[j]][0].unique()])

        for j in range(len(target_outputs)):
            loss += (torch.mean(target_outputs[j])-3)**2

        # loss=torch.mean((outputs[0][0][0][0]-3)**2)
        # loss=(outputs[0][0][0][0][0]-3)**2

        # loss=torch.mean((y-3)**2)
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        if i % 200 == 0:
            print('Generate_dependent_global_trigger_random_neurons loss', loss)
            # print('Gen trigger3 lr',scheduler.get_last_lr())
    return trigger


def poison_train_trigger(model, trainloaders, criterion, epoch, trigger, isGray=True, mask_grad_list=None, locationX=29, locationY=29):
    #对model进行训练
    #model:模型
    #trainloader:训练集
    #optimizer:优化器
    #epoch:训练轮数
    #isGray:是否为灰度照片
    #mask_grad_list:梯度mask
    #trigger:生成的trigger
    # locationX: trigger位置X坐标
    # locationY：trigger位置Y坐标
    triggerWidth = 3
    triggerHeight = 3

    criterion = torch.nn.CrossEntropyLoss()

    poison_optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                       momentum=0.9)
    # weight_decay=helper.params['poison_decay'])
    # scheduler = lr_scheduler.StepLR(poison_optimizer, step_size=100, gamma=0.1)
    model.train()
    if(isGray):
        trigger = torch.reshape(trigger, (1, triggerWidth, triggerHeight))
    else:
        trigger = torch.reshape(trigger, (3, triggerWidth, triggerHeight))
    for i in range(epoch):
        # 对多个trainloader中的数据进行训练
        for trainloader in trainloaders:
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.cuda(), target.cuda()
                index = torch.where(target == 8)
                target[index] = 0
                poison_data = data[index]
                poison_data[:, :, locationX:locationX+triggerWidth, locationY:locationY+triggerHeight] = trigger
                data[index] = poison_data
                poison_optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                if mask_grad_list is not None:
                    apply_grad_mask(model, mask_grad_list)
                poison_optimizer.step()
        print('Poison Train Epoch: {} \tLoss: {:.6f}'.format(
            i, loss.item()))
        # print('lr',scheduler.get_lr())
        # scheduler.step()
    # save_model(file_name='poison_resnet18_FULL',model=model,epoch=epoch)


# def poison_train_trigger0(model, trainloaders, criterion, epoch, trigger0, mask_grad_list=None):
#     #对model进行训练
#     #model:模型
#     #trainloader:训练集
#     #optimizer:优化器
#     #epoch:训练轮数
#     #mask_grad_list:梯度mask
#     #trigger:生成的trigger
#     criterion = torch.nn.CrossEntropyLoss()

#     poison_optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
#                                        momentum=0.9)
#     # weight_decay=helper.params['poison_decay'])
#     # scheduler = lr_scheduler.StepLR(poison_optimizer, step_size=100, gamma=0.1)
#     model.train()
#     trigger0 = torch.reshape(trigger0, (3, 3, 3))
#     for i in range(epoch):
#         # 对多个trainloader中的数据进行训练
#         for trainloader in trainloaders:
#             for batch_idx, (data, target) in enumerate(trainloader):
#                 data, target = data.cuda(), target.cuda()
#                 index = torch.where(target == 8)
#                 target[index] = 0
#                 poison_data = data[index]
#                 poison_data[:, :, 29:32, 29:32] = trigger0
#                 data[index] = poison_data
#                 poison_optimizer.zero_grad()
#                 output = model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 if mask_grad_list is not None:
#                     apply_grad_mask(model, mask_grad_list)
#                 poison_optimizer.step()
#         # tt = transforms.ToPILImage()

#         print('Poison Train Epoch: {} \tLoss: {:.6f}'.format(
#             i, loss.item()))

def poison_train_global_trigger(model, trainloaders, criterion, epoch, global_trigger, mask_grad_list=None):
    #对model进行训练
    #model:模型
    #trainloader:训练集
    #optimizer:优化器
    #epoch:训练轮数
    #mask_grad_list:梯度mask
    #trigger:生成的trigger
    criterion = torch.nn.CrossEntropyLoss()

    poison_optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                       momentum=0.9)
    # weight_decay=helper.params['poison_decay'])
    # scheduler = lr_scheduler.StepLR(poison_optimizer, step_size=100, gamma=0.1)
    model.train()
    global_trigger = torch.reshape(global_trigger, (3, 3, 3))
    for i in range(epoch):
        # 对多个trainloader中的数据进行训练
        for trainloader in trainloaders:
            for batch_idx, (data, target) in enumerate(trainloader):
                data, target = data.cuda(), target.cuda()
                index = torch.where(target == 8)
                target[index] = 0
                poison_data = data[index]
                poison_data[:, :, 29:32, 26:29] = global_trigger
                data[index] = poison_data
                poison_optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                if mask_grad_list is not None:
                    apply_grad_mask(model, mask_grad_list)
                poison_optimizer.step()
        print('Poison Train Epoch: {} \tLoss: {:.6f}'.format(
            i, loss.item()))


def test_poison_model(model, testloader, trigger, pos=-1):
    #测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    # writer_dic['main_task'].add_scalar('Accuracy', 100. * correct / len(testloader.dataset), epoch)
    print('Test set:Main Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    # 测试中毒数据
    correct = 0
    trigger = torch.reshape(trigger, (3, 3, 3))
    poison_sample = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)

            target = torch.zeros_like(index[0])
            poison_data = data[index]
            poison_data[:, :, 29:32, 29:32] = trigger
            data = poison_data
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            poison_sample += len(index[0])
    acc = 100. * correct / poison_sample
    print('Test set:Poison Accuracy: {}/{} ({:.0f}%)'.format(
        correct, poison_sample,
        acc))
    return acc


def test_poison_model_with_globol_trigger(model, testloader, globol_trigger):
    #测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('Test set:Main Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    # 测试中毒数据
    correct = 0
    # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
    poison_sample = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)

            target = torch.zeros_like(index[0])
            poison_data = data[index]
            poison_data[:, :, 29:32, 26:32] = globol_trigger
            data = poison_data
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            poison_sample += len(index[0])
            # 显示poison_data
            # plt.imshow(poison_data[0].permute(1,2,0))
            # plt.show()
            # plt.imsave('poison_data.png',poison_data[0].permute(1,2,0))
    print('Test set:Globol Trigger Poison Accuracy: {}/{} ({:.0f}%)'.format(
        correct, poison_sample,
        100. * correct / poison_sample))



def test_poison_model_with_globol_trigger_four_locals(model, testloader, global_trigger):
    #测试模型
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('Test set:Main Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
    # 测试中毒数据
    correct = 0
    # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
    poison_sample = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)

            target = torch.zeros_like(index[0])
            poison_data = data[index]
            # poison_data[:, :, 22:28, 22:28] = globol_trigger
            global_trigger__temp = global_trigger.unsqueeze(0).repeat(poison_data.shape[0], 1, 1, 1)
            poison_data = torch.where(global_trigger__temp != 0, global_trigger__temp, poison_data)
            data = poison_data
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            poison_sample += len(index[0])
            # 显示poison_data
            # plt.imshow(poison_data[0].permute(1,2,0))
            # plt.show()
            # plt.imsave('poison_data.png',poison_data[0].permute(1,2,0))
    # writer_dic['global_trigger'].add_scalar('Accuracy', 100. * correct / poison_sample, epoch)
    print('Test set:Global Trigger Poison Accuracy: {}/{} ({:.0f}%)'.format(
        correct, poison_sample,
        100. * correct / poison_sample))
    return 100. * correct / poison_sample

def test_poison_model_with_local_trigger(model, testloader, local_trigger,locationX,locationY,triggerName):
    #测试模型
    model.eval()
    # 测试中毒数据
    correct = 0
    # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
    poison_sample = 0
    triggerWidth = 3
    triggerHeight = 3
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)

            target = torch.zeros_like(index[0])
            poison_data = data[index]
            poison_data[:, :, locationX:locationX+triggerWidth, locationY:locationY+triggerHeight] = local_trigger
            data = poison_data
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            poison_sample += len(index[0])
    print(('Test set:local_' + triggerName + 'Poison Accuracy: {}/{} ({:.0f}%)').format(
        correct, poison_sample,
        100. * correct / poison_sample))
    return 100. * correct / poison_sample

def test_poison_model_with_local_trigger0(model, testloader, local_trigger0):
    #测试模型
    model.eval()
    # 测试中毒数据
    correct = 0
    # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
    poison_sample = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)

            target = torch.zeros_like(index[0])
            poison_data = data[index]
            poison_data[:, :, 29:32, 29:32] = local_trigger0
            data = poison_data
            output = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            poison_sample += len(index[0])
    print('Test set:local_trigger0 Poison Accuracy: {}/{} ({:.0f}%)'.format(
        correct, poison_sample,
        100. * correct / poison_sample))


# def test_poison_model_with_local_trigger1(model, testloader, local_trigger1):
#     #测试模型
#     model.eval()
#     # 测试中毒数据
#     correct = 0
#     # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
#     poison_sample = 0
#     with torch.no_grad():
#         for data, target in testloader:
#             data, target = data.cuda(), target.cuda()
#             index = torch.where(target == 8)

#             target = torch.zeros_like(index[0])
#             poison_data = data[index]
#             poison_data[:, :, 29:32, 26:29] = local_trigger1
#             data = poison_data
#             output = model(data)
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             poison_sample += len(index[0])
#     print('Test set:local_trigger1 Poison Accuracy: {}/{} ({:.0f}%)'.format(
#         correct, poison_sample,
#         100. * correct / poison_sample))


# def test_poison_model_with_local_trigger2(model, testloader, local_trigger2):
#     #测试模型
#     model.eval()
#     # 测试中毒数据
#     correct = 0
#     # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
#     poison_sample = 0
#     with torch.no_grad():
#         for data, target in testloader:
#             data, target = data.cuda(), target.cuda()
#             index = torch.where(target == 8)

#             target = torch.zeros_like(index[0])
#             poison_data = data[index]
#             poison_data[:, :, 26:29, 26:29] = local_trigger2
#             data = poison_data
#             output = model(data)
#             # get the index of the max log-probability
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#             poison_sample += len(index[0])
#     print('Test set:local_trigger2 Poison Accuracy: {}/{} ({:.0f}%)'.format(
#         correct, poison_sample,
#         100. * correct / poison_sample))


# def test_poison_model_with_local_trigger3(model, testloader, local_trigger3):
    # #测试模型
    # model.eval()
    # # 测试中毒数据
    # correct = 0
    # # globol_trigger=torch.reshape(globol_trigger,(3,3,6))
    # poison_sample = 0
    # with torch.no_grad():
    #     for data, target in testloader:
    #         data, target = data.cuda(), target.cuda()
    #         index = torch.where(target == 8)

    #         target = torch.zeros_like(index[0])
    #         poison_data = data[index]
    #         poison_data[:, :, 26:29, 29:32] = local_trigger3
    #         data = poison_data
    #         output = model(data)
    #         # get the index of the max log-probability
    #         pred = output.argmax(dim=1, keepdim=True)
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #         poison_sample += len(index[0])
    # print('Test set:local_trigger3 Poison Accuracy: {}/{} ({:.0f}%)'.format(
    #     correct, poison_sample,
    #     100. * correct / poison_sample))


def test_all_poisoned_data(poison_model, testloader, trigger):
    #测试模型
    poison_model.eval()
    correct = 0
    trigger = torch.reshape(trigger, (3, 3, 3))
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            # 为data添加触发器
            data[:, :, 29:32, 29:32] = trigger
            target = torch.zeros_like(target)
            output = poison_model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    print('Test set:Poison Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(testloader.dataset),
        100. * correct / len(testloader.dataset)))
# l2范数太大了！！


def clip_grad(norm_bound, weight_difference, difference_flat):
    l2_norm = torch.norm(difference_flat.clone().detach().cuda())
    scale = max(1.0, float(torch.abs(l2_norm / norm_bound)))
    for name in weight_difference.keys():
        weight_difference[name].div_(scale)

    return weight_difference, l2_norm


def get_l2_norm(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)

    l2_norm = torch.norm(difference_flat.clone().detach().cuda())

    l2_norm_np = np.linalg.norm(difference_flat.cpu().numpy())

    return l2_norm, l2_norm_np


def get_weight_difference(weight1, weight2):
    difference = {}
    res = []
    if type(weight2) == dict:
        for name, layer in weight1.items():
            difference[name] = layer.data - weight2[name].data
            res.append(difference[name].view(-1))
    else:
        for name, layer in weight2:
            difference[name] = weight1[name].data - layer.data
            res.append(difference[name].view(-1))

    difference_flat = torch.cat(res)

    return difference, difference_flat


def My_local_train(global_model, local_model, epoch, criterion, sampled_participants, train_loaders, lr_init=0.001, traget_lr=0.2):
    local_weights = dict()
    for name, data in global_model.state_dict().items():
        local_weights[name] = torch.zeros_like(data)

    global_model_copy = dict()
    # for name, param in global_model.named_parameters():
    for name, param in global_model.named_parameters():
        global_model_copy[name] = global_model.state_dict(
        )[name].clone().detach().requires_grad_(False)
    for participant_id in sampled_participants:
        # local_model 在训练过程中被改变了？虽然使用copy_params，但是有部分参数没有被改变
        # 因为copy_params使用named_parameters()
        model = local_model
        copy_params(model, global_model_copy)
        #-------------------------test-------------------------
        # for name, param in target_model.state_dict().items():
        #     if not torch.allclose(model.state_dict()[name], param):
        #         print(name)
        model.train()
        ### update lr
        lr = lr_init
        if epoch <= 500:
            lr = epoch*(traget_lr - lr_init)/499.0 + \
                lr_init - (traget_lr - lr_init)/499.0
        else:
            lr = epoch*(-traget_lr)/1500 + traget_lr*4.0/3.0

        if lr <= 0.0001:
            lr = 0.0001
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=0.9,
                                    weight_decay=5e-4)
        for internal_epoch in range(2):
            loss = local_train(model, optimizer, criterion,
                               train_loaders[participant_id])
            # print(f'participant_id:{participant_id},epoch:{epoch},internal_epoch:{internal_epoch},loss:{loss}')
            # loss = local_train(target_model, optimizer, criterion, train_loaders[participant_id])

        l2_norm, l2_norm_np = get_l2_norm(
            dict(global_model.named_parameters()), dict(model.named_parameters()))
        print(
            f'participant_id:{participant_id},epoch:{epoch},l2_norm:{l2_norm}')
        for name, data in model.state_dict().items():
            local_weights[name].add_(data)
    return local_weights


def apply_grad_mask(model, mask_grad_list):
    mask_grad_list_copy = []
    for key in mask_grad_list.keys():
        mask_grad_list_copy.append(mask_grad_list[key])
    mask_grad_list_copy = iter(mask_grad_list_copy)
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            parms.grad = parms.grad * next(mask_grad_list_copy)


def resolve_conflict(model, trainloaders, mask_grad_list, trigger):
    target_layer_neuron = {}
    before_step_value = []
    after_step_value = []
    dic_modules = dict(model.named_modules())
    for name, params in model.named_parameters():
        indices = torch.where(mask_grad_list[name] == 1)
        if indices[0].shape != torch.Size([0]):
            target_layer_neuron[name] = (indices)
    #         #为该层注册hook
            dic_modules[name.rsplit('.', 1)[0]].register_forward_hook(hook_fn)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001,
                                momentum=0.9)
    # weight_decay=helper.params['poison_decay'])
    # scheduler = lr_scheduler.StepLR(poison_optimizer, step_size=100, gamma=0.1)
    model.train()
    trigger = torch.reshape(trigger, (3, 3, 3))
    # 对多个trainloader中的数据进行训练
    for trainloader in trainloaders:
        for batch_idx, (data, target) in enumerate(trainloader):
            outputs.clear()
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)
            target[index] = 0
            poison_data = data[index]
            poison_data[:, :, 29:32, 26:29] = trigger
            data[index] = poison_data
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            # before_step_value为outputs的平均值,选取模型每层的输出的平均值，不是针对单个神经元，先调试看看效果
            for i in range(len(outputs)):
                before_step_value.append(torch.mean(outputs[i]))
            outputs.clear()
            break
        break
    for trainloader in trainloaders:
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.cuda(), target.cuda()
            index = torch.where(target == 8)
            target[index] = 0
            poison_data = data[index]
            poison_data[:, :, 29:32, 26:29] = trigger
            data[index] = poison_data
            optimizer.zero_grad()
            output = model(data)
            for i in range(len(outputs)):
                after_step_value.append(torch.mean(outputs[i]))
            outputs.clear()
            break
        break
    for i in range(len(before_step_value)):
        if before_step_value[i] >= after_step_value[i]:
            mask_grad_list[list(target_layer_neuron)[i]] = torch.zeros_like(
                mask_grad_list[list(target_layer_neuron)[i]])
    print('resolved_conflict')
    return mask_grad_list
