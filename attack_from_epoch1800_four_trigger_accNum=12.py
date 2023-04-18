import random
import time
import numpy as np
import copy
import torch
from models.resnet import ResNet18
import eminist_util as util
import matplotlib.pyplot as plt
torch.manual_seed(50)
torch.cuda.manual_seed(1)

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled=False

random.seed(0)
np.random.seed(0)
round=500
participant_population=1000
partipant_sample_size=10
attack_round0=[i for i in range(300,400,40)]
attack_round1=[i for i in range(305,400,40)]
attack_round2=[i for i in range(310,400,40)]
attack_round3=[i for i in range(315,400,40)]

num_classes=10
# 读取训练到1900轮的模型
# local_model = util.load_model('./backup/fed_learning_test/saved_models/cifar10_resnet_Snorm_1_checkpoint_model_epoch_1800.pth')
# target_model = util.load_model('./backup/fed_learning_test/saved_models/cifar10_resnet_Snorm_1_checkpoint_model_epoch_1800.pth')
local_model = util.load_model('./backup/fed_learning_test/saved_models/emnist_resnet_Snorm_1_checkpoint_model_epoch_1900.pth')
target_model = util.load_model('./backup/fed_learning_test/saved_models/emnist_resnet_Snorm_1_checkpoint_model_epoch_1900.pth')
poison_model=ResNet18(num_classes=num_classes)
poison_model.cuda()
train_loaders,test_loader=util.load_Non_IID_clean_data(number_of_total_participants=participant_population,dirichlet_alpha=0.9)
criterion = torch.nn.CrossEntropyLoss()
weight_accumulator = None
attacker_index=0
poison_optimizer = torch.optim.SGD(poison_model.parameters(), lr=0.001,
                                    momentum=0.9,
                                    weight_decay=5e-4)
poison_model=ResNet18(num_classes=num_classes)
poison_model.cuda()
for i in range(1900,round):
    if i in attack_round0:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==1800:
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[:10])
            mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
            # trigger0=util.Generate_trigger0(poison_model_with_hook, mask_grad_list)
            trigger0=util.generate_trigger('Trigger0',poison_model_with_hook, mask_grad_list, 25, 25)
            # torch.save(trigger0,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger0.pth')
            del poison_model_with_hook
            # trigger0=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger0.pth')
        util.poison_train_trigger0(poison_model,train_loaders[:10], criterion=None,epoch=20 ,trigger0=trigger0)
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round1:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        # 如果是攻击的第一轮，初始化trigger
        if i==1805:
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[:10])
            mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
            # trigger1=util.Generate_trigger1(poison_model_with_hook, mask_grad_list)
            trigger1 = util.generate_trigger('Trigger1',poison_model_with_hook, mask_grad_list, 25, 22)
            # torch.save(trigger0,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger1.pth')
            del poison_model_with_hook
            # trigger1=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger1.pth')

        util.poison_train_trigger1(poison_model,train_loaders[:10], criterion=None,epoch=20 ,trigger1=trigger1)
        
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round2:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==1810:
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[:10])
            mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
            # trigger2=util.Generate_trigger2(poison_model_with_hook, mask_grad_list)
            trigger2 = util.generate_trigger('Trigger2',poison_model_with_hook, mask_grad_list, 22, 22)
            # torch.save(trigger2,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger2.pth')
            del poison_model_with_hook
            # trigger2=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger2.pth')

        util.poison_train_trigger2(poison_model,train_loaders[:10], criterion=None,epoch=20 ,trigger2=trigger2)
        
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    elif i in attack_round3:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size-1)
        print(f'select client {sampled_participants}')
        print('------------------poisoning-------------------')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
        poison_model.load_state_dict(target_model.state_dict())
        if i==1815:
            poison_model_with_hook=ResNet18(num_classes=num_classes)
            poison_model_with_hook.cuda()
            poison_model_with_hook.load_state_dict(target_model.state_dict())
            util.gen_model_grad(poison_model_with_hook, criterion, train_loaders[:10])
            mask_grad_list=util.xzy_grad_mask_cv(poison_model_with_hook,ratio=0.001)
            # trigger3 = util.Generate_trigger3(poison_model_with_hook, mask_grad_list)
            trigger3 = util.generate_trigger('Trigger3',poison_model_with_hook, mask_grad_list, 22, 25)
            # torch.save(trigger3,'/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger3.pth')
            del poison_model_with_hook
            # trigger3=torch.load('/home/will/backdoor_attack/attack_from_epoch1800_four_trigger_accNum=12/trigger3.pth')

        util.poison_train_trigger3(poison_model,train_loaders[:10], criterion=None,epoch=20 ,trigger3=trigger3)
        
        for name, data in poison_model.state_dict().items():
            weight_accumulator[name].add_((data - target_model.state_dict()[name]))
    else:
        start_time = time.time()
        sampled_participants = random.sample(range(10,participant_population), partipant_sample_size)
        print(f'select client {sampled_participants}')
        weight_accumulator = util.train_cv(target_model,local_model, i, criterion, sampled_participants,train_loaders=train_loaders)
    util.average_shrink_models(target_model=target_model,
                                     weight_accumulator=weight_accumulator, epoch=i)
    epoch_loss, epoch_acc = util.test_cv(i, data_source=test_loader,
                                                           model=target_model)
    print('benign test loss (after fedavg)', epoch_loss)
    print('benign test acc (after fedavg)', epoch_acc)

    if i>1815:
        global_trigger0=torch.cat((trigger1.reshape(3,3,3),trigger0.reshape(3,3,3)),dim=2)
        global_trigger1=torch.cat((trigger2.reshape(3,3,3),trigger3.reshape(3,3,3)),dim=2)
        global_trigger=torch.cat((global_trigger1,global_trigger0),dim=1)
        util.test_poison_model_with_local_trigger0(target_model, test_loader,trigger0.reshape(3,3,3))
        util.test_poison_model_with_local_trigger1(target_model, test_loader,trigger1.reshape(3,3,3))
        util.test_poison_model_with_local_trigger2(target_model, test_loader,trigger2.reshape(3,3,3))
        util.test_poison_model_with_local_trigger3(target_model, test_loader,trigger3.reshape(3,3,3))

        util.test_poison_model_with_globol_trigger_four_locals(target_model, test_loader,global_trigger)
    print(f'Done in {time.time()-start_time} sec.')

util.save_model('attack_from_epoch1800_four_triggers.pth',target_model,epoch=round)
