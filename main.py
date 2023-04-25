import torch
import argparse
import json
import wandb
from data_process.kwai_process import create_file_by_data as kwai_process_data
from data_process.kddcup2015_process import create_file_by_data as kddcup2015_process_data
from dataloader.load_data import get_data_loader
from run import run
from model.RNN import RNN
from model.DPCNN import DPCNN
from model.LR import LR
from model.LSCNN import LSCNN
from model.CFIN import CFIN
from model.CLSA import CLSA
from model.MyModel import MyModel

model_dict = {
    "RNN": RNN,
    "DPCNN": DPCNN,
    "LR": LR,
    "LSCNN": LSCNN,
    "CFIN": CFIN,
    "CLSA": CLSA,
    "MyModel": MyModel,
}


def load_model_param(config_file):
    f = open(config_file, 'r')
    model_param = json.load(f)
    return model_param


def init_wandb(project_name, user_name, model_params, api_key):
    wandb.login(key=api_key)
    wandb.init(project=project_name, name=user_name)
    wandb.watch_called = False
    config = wandb.config

    config.seed = model_params['seed']
    config.batch_size = model_params['batch_size']
    config.learning_rate = model_params['learning_rate']
    config.weight_decay = model_params['weight_decay']
    config.max_iter = model_params['max_iter']
    config.DataSet = model_params['DataSet']
    config.day = model_params['day']
    config.future_day = model_params['future_day']
    config.data_dilution_ratio = model_params['data_dilution_ratio']
    config.whether_process = model_params['whether_process']
    config.loss_func = model_params['loss_func']
    config.cuda = model_params['cuda']
    config.model_name = model_params['model_name']
    config.bce_weight = model_params['bce_weight']
    config.multi_task_enable = model_params['multi_task_enable']
    config.fine_grained = model_params['fine_grained']


def main():
    # Namespace of Hyper-parameter
    parser = argparse.ArgumentParser()
    # training process
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size e.g. 32 64')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate e.g. 0.001 0.01')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay e.g. 1e-5 5e-5')
    parser.add_argument('--max_iter', type=int, default=50, help='max_iter e.g. 100 200 ...')
    # dataset
    parser.add_argument('--DataSet', type=str, default='kwai', help='dataset\'s name')
    parser.add_argument('--day', type=int, default=23, help='past day')
    parser.add_argument('--future_day', type=int, default=7, help='future day')
    parser.add_argument('--data_dilution_ratio', type=float, default=1.0)
    parser.add_argument('--whether_process', type=bool, default=False, help='Whether to process dataset')
    # loss
    parser.add_argument('--loss_func', type=str, default='MSE')
    # gpu
    parser.add_argument('--cuda', type=int, default=0)
    # model
    parser.add_argument('--model_name', type=str, default='LSCNN')
    # bce_weight
    parser.add_argument('--bce_weight', type=float, default=0.05)
    # MyModel parameters
    parser.add_argument('--multi_task_enable', type=int, default=0, help='Enable multitask prediction, 0 means not enable')
    parser.add_argument('--fine_grained', type=int, default=0, help='Enable fine-grained, 0 means not enable')
    # other
    parser.add_argument('--use_wandb', type=bool, default=False, help='Whether to use wandb, True means use')
    parser.add_argument('--save_result', type=bool, default=False, help='Whether to save prediction results, True means save')

    params = parser.parse_args()
    # GPU settings
    device = torch.device("cuda:" + str(params.cuda) if torch.cuda.is_available() else "cpu")

    # hyper-parameter(dict)
    param = vars(params)
    param['device'] = device

    # data pre process
    if params.whether_process:
        if params.DataSet == 'kwai':
            kwai_process_data(params.day, params.future_day, params.data_dilution_ratio)
        elif params.DataSet == 'kddcup2015':
            kddcup2015_process_data(params.day, params.future_day, params.data_dilution_ratio)
        else:
            pass

    """    
    obtain Dataset: 
        ui_train、uv_train: [user_num, user_image_feature_num]
        ai_train、av_train: [user_num, past_day, action_feature_num]
        time_train: [user_num, past_day + future_day, 4]
        y: [user_num, truth + total_activity_day + day1_1 + ... + dayN_feature_num]
    """
    if params.DataSet == 'kwai':
        train_set, valid_set, test_set, day_numpy, param = get_data_loader(params.batch_size, param, 'kwai')
    elif params.DataSet == 'kddcup2015':
        train_set, valid_set, test_set, day_numpy, param = get_data_loader(params.batch_size, param, 'kddcup2015')
    else:
        pass

    # Create Model
    # The difference between params and param: params does not contain dataset information, such as 'u_feat_size'.
    model_name = params.model_name
    model_params = load_model_param('./config/' + model_name + '.json')
    model_params.update(param)
    params = argparse.Namespace(**model_params)
    print(model_params)
    model = model_dict[model_name](model_params)
    model.to(device)

    # log file
    file_name = './log/' + str(params.DataSet) + '_' + str(params.model_name) + '_' + str(params.loss_func) + '_' + str(
        params.day) + '_' + str(params.future_day) + '_' + str(params.seed) + '_' + str(
        params.learning_rate) + '_' + str(params.weight_decay) + '_' + str(params.max_iter) + '-'
    f = open(file_name + 'log.txt', 'w')
    f.write(str(model_params) + '\n')

    # create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    # wandb
    if model_params['use_wandb']:
        init_wandb('WDAPM', 'Zikai Li', model_params, '34505f6e3ffacfbcb4aa26f7af20171b3c31b63a')
        wandb.watch(model, log='all')

    # best result
    best_epoch = 0
    best_valid_rmse = best_valid_df = best_valid_MAE = 1e9
    best_rmse = best_df = best_MAE = 1e9

    # train model
    for i in range(params.max_iter):
        model.train()
        run(epoch=i, dataset=train_set, model=model, optimizer=optimizer, device=device, model_name=model_name,
            run_type='train', loss_func=params.loss_func, write_file=f, model_params=model_params)
        model.eval()
        valid_rmse, valid_df, valid_MAE = run(epoch=-1, dataset=valid_set, model=model, optimizer=optimizer,
                                              device=device, model_name=model_name, run_type='valid', loss_func=None,
                                              write_file=f, model_params=model_params)
        if model_params['use_wandb']:
            wandb.log({'Valid rmse': valid_rmse, 'valid df': valid_df, 'valid MAE': valid_MAE})
        print(
            'epoch: %.4f\n  valid_rmse %.4f  valid_df %.4f  valid_MAE %.4f' % (i + 1, valid_rmse, valid_df, valid_MAE))
        if valid_rmse < best_valid_rmse:
            model.eval()
            best_valid_MAE, best_valid_rmse, best_valid_df, best_epoch = valid_MAE, valid_rmse, valid_df, i + 1
            test_rmse, test_df, test_MAE = run(epoch=-1, dataset=test_set, model=model, optimizer=optimizer,
                                               device=device, model_name=model_name, run_type='test', loss_func=None,
                                               write_file=f, model_params=model_params)
            if model_params['use_wandb']:
                wandb.log({'test rmse': test_rmse, 'test df': test_df, 'test MAE': test_MAE})
            print('  test_rmse %.4f  test_df %.4f  test_MAE %.4f' % (test_rmse, test_df, test_MAE))
            best_rmse, best_df, best_MAE = test_rmse, test_df, test_MAE

    best_log = 'best_epoch: %.4f\n  best_valid_rmse %.4f  best_valid_df %.4f  best_valid_MAE %.4f \n  best_rmse %.4f  best_df %.4f best_MAE %.4f' % (
        best_epoch, best_valid_rmse, best_valid_df, best_valid_MAE, best_rmse, best_df, best_MAE)
    print(best_log)
    f.write(best_log + '\n')
    f.close()


if __name__ == '__main__':
    main()
