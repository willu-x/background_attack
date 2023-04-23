import random
import time
import numpy as np
import copy
import torch
from models.resnet import ResNet18
from models.lenet import createLeNet
import eminist_util as util
import matplotlib.pyplot as plt
torch.manual_seed(50)
torch.cuda.manual_seed(1)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(0)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled=False

# random.seed(0)
# np.random.seed(0)
# 读取训练到1950轮的模型
# startRound = 1950
startRound = 1950
local_model = util.load_model('/home/hjb/src/backup/fed_learning_test/saved_models/emnist_resnet_Snorm_1_checkpoint_model_epoch_1950.pth')
target_model = util.load_model('/home/hjb/src/backup/fed_learning_test/saved_models/emnist_resnet_Snorm_1_checkpoint_model_epoch_1950.pth')
# local_model = util.load_lenet_model('/home/hjb/src/backup/fed_learning_test/saved_models/lenet/emnist_lenet_Snorm_1_checkpoint_model_epoch_1950.pth')
# target_model = util.load_lenet_model('/home/hjb/src/backup/fed_learning_test/saved_models/lenet/emnist_lenet_Snorm_1_checkpoint_model_epoch_1950.pth')
round = startRound + 250
participant_population=1000
partipant_sample_size=10
isRandomNeurons = False
isSimpleTrigger = False
generate_overlap_layer = [i for i in range(startRound,startRound+4)]
attack_round0=[i for i in range(startRound+5,startRound+110,20)]
attack_round1=[i for i in range(startRound+10,startRound+110,20)]
attack_round2=[i for i in range(startRound+15,startRound+110,20)]
attack_round3=[i for i in range(startRound+20,startRound+110,20)]

num_classes=10

poison_model=ResNet18(num_classes=num_classes)
# poison_model = createLeNet()
poison_model.cuda()
train_loaders,test_loader=util.load_Non_IID_clean_data(number_of_total_participants=participant_population,dirichlet_alpha=0.9)
criterion = torch.nn.CrossEntropyLoss()
weight_accumulator = None
attacker_index = 0
poison_optimizer = torch.optim.SGD(poison_model.parameters(), lr=0.001,
                                    momentum=0.9,
                                    weight_decay=5e-4)
poison_model=ResNet18(num_classes=num_classes)
# poison_model = createLeNet()
poison_model.cuda()
correct = 0
sample_size = len(test_loader)

writer = util.initTensorBoard('eminist_attack_four_trigger' + str(time.time()))

epoch_loss, epoch_acc = util.test_cv(0, data_source=test_loader,model=target_model)
print('Origin Pretrained Model: benign test loss (after fedavg)', epoch_loss)
print('Origin Pretrained Model: benign test acc (after fedavg)', epoch_acc)
layer_list = []
for i in range(startRound,round):
    if i in generate_overlap_layer:
        poison_model_with_hook=ResNet18(num_classes=num_classes)
        # poison_model_with_hook = createLeNet()
        poison_model_with_hook.cuda()
        poison_model_with_hook.load_state_dict(target_model.state_dict())
        util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[:10])
        # mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
        layer_list = util.gen_overlap_layer_list(model=poison_model_with_hook,layer_list=layer_list,ratio=0.001)
        del poison_model_with_hook
    if i in attack_round0:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==attack_round0[0]:
            # 25,25
            locationX_triger0 = 25
            locationY_triger0 = 25
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            # poison_model_with_hook = createLeNet()
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            # 使用随机训练集生成trigger
            randInt = 0
            # 是否随机选择神经元
            if isRandomNeurons:
                trigger0 = util.generate_dependent_trigger_random_neurons('Trigger0',poison_model_with_hook, 25, 25)
            else:
                if isSimpleTrigger:
                    trigger0 = util.generate_simple_trigger(2.8088)
                else:
                    util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[randInt:randInt+10])
                    # mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
                    # mask_grad_list=util.xzy_grad_mask_cv_by_layer(poison_model_with_hook,ratio=0.001)
                    mask_grad_list = util.xzy_grad_mask_cv_by_overlap_layer_name(poison_model_with_hook,layer_list,ratio=0.001)
                    # trigger0=util.Generate_trigger0(poison_model_with_hook, mask_grad_list)
                    trigger0=util.generate_trigger('Trigger0',poison_model_with_hook, mask_grad_list, 25, 25)
                    # torch.save(trigger0,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger0.pth')
            del poison_model_with_hook

            # trigger0=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger0.pth')
        util.poison_train_trigger(poison_model,train_loaders[randInt:randInt+10], criterion=None, epoch=20 ,trigger=trigger0,isGray=True,locationX=25,locationY=25)
        util.test_poison_model_with_local_trigger(poison_model, test_loader,trigger0.reshape(1,3,3),25,25,'Trigger0_local')
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round1:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i == attack_round1[0]:
            # 25,22
            locationX_triger1 = 0
            locationY_triger1 = 0
            poison_model_with_hook=ResNet18(num_classes = num_classes)
            # poison_model_with_hook = createLeNet()
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            # 使用随机训练集生成trigger
            randInt = 0
            # 是否随机选择神经元
            if isRandomNeurons:
                trigger1 = util.generate_dependent_trigger_random_neurons('Trigger1',poison_model_with_hook, locationX_triger1, locationY_triger1)
            else:
                if isSimpleTrigger:
                    trigger1 = util.generate_simple_trigger(2.8088)
                else:
                    util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[randInt:randInt+10])
                    # mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
                    # mask_grad_list=util.xzy_grad_mask_cv_by_layer(poison_model_with_hook,ratio=0.001)
                    # trigger1=util.Generate_trigger1(poison_model_with_hook, mask_grad_list)
                    mask_grad_list = util.xzy_grad_mask_cv_by_overlap_layer_name(poison_model_with_hook,layer_list,ratio=0.001)
                    trigger1 = util.generate_trigger('Trigger1',poison_model_with_hook, mask_grad_list, locationX_triger1, locationY_triger1)
                    # torch.save(trigger0,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger1.pth')
            del poison_model_with_hook

            # trigger1=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger1.pth')
        util.poison_train_trigger(poison_model,train_loaders[randInt:randInt+10], criterion=None, epoch=20 ,trigger=trigger1,isGray=True,locationX=locationX_triger1,locationY=locationY_triger1)
        util.test_poison_model_with_local_trigger(poison_model, test_loader,trigger1.reshape(1,3,3),locationX_triger1,locationY_triger1,'Trigger1_local')
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round2:

        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==attack_round2[0]:
            # 22,22
            locationX_triger2 = 0
            locationY_triger2 = 25
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            # poison_model_with_hook = createLeNet()
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            # 使用随机训练集生成trigger
            randInt = 0
            # 是否随机选择神经元
            if isRandomNeurons:
                trigger2 = util.generate_dependent_trigger_random_neurons('Trigger2',poison_model_with_hook, locationX_triger2, locationY_triger2)
                del poison_model_with_hook
            else:
                if isSimpleTrigger:
                    trigger2 = util.generate_simple_trigger(2.8088)
                else:
                    util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[randInt:randInt+10])
                    # mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
                    # mask_grad_list=util.xzy_grad_mask_cv_by_layer(poison_model_with_hook,ratio=0.001)
                    # trigger2=util.Generate_trigger2(poison_model_with_hook, mask_grad_list)
                    mask_grad_list = util.xzy_grad_mask_cv_by_overlap_layer_name(poison_model_with_hook,layer_list,ratio=0.001)
                    trigger2 = util.generate_trigger('Trigger2',poison_model_with_hook, mask_grad_list, locationX_triger2, locationY_triger2)
                    # torch.save(trigger2,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger2.pth')
            del poison_model_with_hook

            # trigger2=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger2.pth')

        util.poison_train_trigger(poison_model,train_loaders[randInt:randInt+10], criterion=None, epoch=20 ,trigger=trigger2,isGray=True,locationX=locationX_triger2,locationY=locationY_triger2)
        util.test_poison_model_with_local_trigger(poison_model, test_loader,trigger2.reshape(1,3,3),locationX_triger2,locationY_triger2,'Trigger2_local')
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round3:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==attack_round3[0]:
            # 22,25
            locationX_triger3 = 25
            locationY_triger3 = 0
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            # poison_model_with_hook = createLeNet()
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            # 使用随机训练集生成trigger
            randInt = 0
            # 是否随机选择神经元
            if isRandomNeurons:
                trigger3 = util.generate_dependent_trigger_random_neurons('Trigger3',poison_model_with_hook, locationX_triger3, locationY_triger3)
                del poison_model_with_hook
            else:
                if isSimpleTrigger:
                    trigger3 = util.generate_simple_trigger(2.8088)
                else:
                    util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[randInt:randInt+10])
                    # mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
                    # mask_grad_list=util.xzy_grad_mask_cv_by_layer(poison_model_with_hook,ratio=0.001)
                    # trigger3 = util.Generate_trigger3(poison_model_with_hook, mask_grad_list)
                    mask_grad_list = util.xzy_grad_mask_cv_by_overlap_layer_name(poison_model_with_hook,layer_list,ratio=0.001)
                    trigger3 = util.generate_trigger('Trigger3',poison_model_with_hook, mask_grad_list, locationX_triger3, locationY_triger3)
                    
                # torch.save(trigger3,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger3.pth')
            del poison_model_with_hook

            # trigger3=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger3.pth')

        util.poison_train_trigger(poison_model,train_loaders[randInt:randInt+10], criterion=None, epoch=20 ,trigger=trigger3,isGray=True,locationX=locationX_triger3,locationY=locationY_triger3)
        util.test_poison_model_with_local_trigger(poison_model, test_loader,trigger3.reshape(1,3,3),locationX_triger3,locationY_triger3,'Trigger3_local')
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    else:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size)
        print(f'select client {sampled_participants}')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
    util.average_shrink_models(target_model=target_model,weight_accumulator=weight_accumulator, epoch=i)
    epoch_loss, epoch_acc = util.test_cv(i, data_source=test_loader,
                                                           model=target_model)
    print('benign test loss (after fedavg)', epoch_loss)
    print('benign test acc (after fedavg)', epoch_acc)
    # if i>startRound+10 and i<startRound+20:
    #     util.test_poison_model_with_local_trigger(target_model, test_loader,trigger1.reshape(1,3,3),25,25,'trigger1')
    if i>startRound+20:
        # global_trigger0=torch.cat((trigger1.reshape(1,3,3),trigger0.reshape(1,3,3)),dim=2)
        # global_trigger1=torch.cat((trigger2.reshape(1,3,3),trigger3.reshape(1,3,3)),dim=2)
        # global_trigger=torch.cat((global_trigger1,global_trigger0),dim=1)
        acc0 = util.test_poison_model_with_local_trigger(target_model, test_loader,trigger0.reshape(1,3,3),locationX_triger0,locationY_triger0,'Trigger0')
        acc1 = util.test_poison_model_with_local_trigger(target_model, test_loader,trigger1.reshape(1,3,3),locationX_triger1,locationY_triger1,'Trigger1')
        acc2 = util.test_poison_model_with_local_trigger(target_model, test_loader,trigger2.reshape(1,3,3),locationX_triger2,locationY_triger2,'Trigger2')
        acc3 = util.test_poison_model_with_local_trigger(target_model, test_loader,trigger3.reshape(1,3,3),locationX_triger3,locationY_triger3,'Trigger3')
        global_trigger = torch.zeros((1,28,28)).cuda()
        global_trigger[:,25:28, 25:28] = trigger0.reshape(1,3,3).clone()
        global_trigger[:,0:3, 0:3] = trigger1.reshape(1,3,3).clone()
        global_trigger[:,0:3, 25:28] = trigger2.reshape(1,3,3).clone()
        global_trigger[:,25:28, 0:3] = trigger3.reshape(1,3,3).clone()
        acc_global = util.test_poison_model_with_globol_trigger_four_locals(target_model, test_loader,global_trigger)

        writer.add_scalars('Accuracy', {
            'trigger0':acc0,
            'trigger1':acc1,
            'trigger2':acc2,
            'trigger3':acc3,
            'global_trigger':acc_global
        },i)
    print(f'Done in {time.time()-start_time} sec.')

util.save_model('attack_from_epoch1800_four_triggers_eminist.pth',target_model,epoch=round)
