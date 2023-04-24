import matplotlib.pyplot as plt
import numpy as np

params = {
    'average_RMSE': {},
    'average_df': {},
    'average_MAE': {},
}


def statistical(file_path, model_params):
    global params
    for dataset_name in model_params['dataset']:
        params['average_RMSE'].update({dataset_name: []})
        params['average_df'].update({dataset_name: []})
        params['average_MAE'].update({dataset_name: []})
        for model_name in model_params['model']:
            total_RMSE = 0
            total_df = 0
            total_MAE = 0
            for seed_ in model_params['seed']:
                file_name = dataset_name + '_' + \
                            model_name + '_' + \
                            model_params['loss_func'] + '_' + \
                            str(model_params['day']) + '_' + \
                            str(model_params['future_day']) + '_' + \
                            str(seed_) + '_' + \
                            str(model_params['learning_rate']) + '_' + \
                            str(model_params['wd']) + '_' + \
                            str(model_params['max_iter']) + '-log.txt'

                with open(file_path + dataset_name + '/' + model_name + '/' + file_name, 'r') as f:
                    temp_result = f.readlines()[-1].split()
                    total_RMSE += float(temp_result[1])
                    total_df += float(temp_result[3])
                    total_MAE += float(temp_result[5])

            average_RMSE = np.round(total_RMSE / len(model_params['seed']), 5)
            average_df = np.round(total_df / len(model_params['seed']), 5)
            average_MAE = np.round(total_MAE / len(model_params['seed']), 5)

            params['average_RMSE'].get(dataset_name).append(average_RMSE)
            params['average_df'].get(dataset_name).append(average_df)
            params['average_MAE'].get(dataset_name).append(average_MAE)


def draw_result(dataset, model, day=23, future_day=7, save_path=None):
    global params
    for dataset_name in dataset:
        RMSE = params['average_RMSE'].get(dataset_name)
        df = params['average_df'].get(dataset_name)
        MAE = params['average_MAE'].get(dataset_name)
        print(dataset_name + ' RMSE: ', RMSE)
        print(dataset_name + ' df: ', df)
        print(dataset_name + ' MAE: ', MAE, '\n')

        plt.plot(model, RMSE, 'r', marker='.', markersize=4, label='RMSE')
        plt.plot(model, df, 'g', marker='*', markersize=4, label='df')
        plt.plot(model, MAE, 'b', marker='X', markersize=4, label='MAE')
        plt.xlabel("Model Name")
        plt.ylabel("Value")
        plt.title(dataset_name + ' dataset, use ' + str(day) + ' to predict ' + str(future_day))
        plt.legend(loc="upper left")
        for x1, y1 in zip(model, RMSE):
            plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(model, df):
            plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
        for x1, y1 in zip(model, MAE):
            plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)

        plt.savefig(save_path + dataset_name + '_' + str(day) + '_' + str(future_day) + '.png')
        plt.show()


def run_main():
    log_file_path = '../log/'
    pic_save_path = '../log/result_picture/'
    model_params = {
        'dataset': ['kwai'],
        'model': ['CFIN', 'CLSA', 'DPCNN', 'MyModel', 'LSCNN', 'RNN'],
        'loss_func': 'MSE',
        'seed': [1, 2, 3, 4, 5],
        'day': 14,
        'future_day': 16,
        'learning_rate': 0.001,
        'wd': 1e-5,
        'max_iter': 50,
    }

    statistical(file_path=log_file_path, model_params=model_params)
    draw_result(dataset=model_params['dataset'], model=model_params['model'], day=model_params['day'],
                future_day=model_params['future_day'], save_path=pic_save_path)


if __name__ == '__main__':
    run_main()
