import numpy as np
import torch
import pandas as pd
import wandb
import torch.nn.functional as F

"""
cal_eval_result() function's purpose:
    input: loss value(float), y_preds([batch_size, 1](float), y_truths([batch_size, 1]))
    output: rmse, df, MAE

    rmse: root mean square error(The smaller the better)
    df: degrees of freedom(The smaller the better)
    MAE: mean absolute error(The smaller the better)
"""
def cal_eval_result(all_loss, predict_results, run_type, model_name, write_file=None):
    if model_name == 'MyModel':
        user_ids, y_preds_1, y_truths_1, y_preds_2, y_truths_2 = predict_results
    else:
        user_ids, y_preds_1, y_truths_1 = predict_results

    error = y_truths_1 - y_preds_1
    # RMSE
    rmse = ((error) ** 2).mean() ** 0.5
    # df
    df = abs(y_truths_1.mean() - y_preds_1.mean()) / y_truths_1.mean()
    # MAE
    MAE = np.absolute(error).mean()

    log_str = '%20s loss %3.6f  rmse %.4f  df(ActivateDay.Avg) %.4f  MAE %.4f' % (run_type, all_loss, rmse, df, MAE)

    if write_file is not None:
        write_file.write(log_str + "\n")

    return rmse, df, MAE


"""
cal_save_result() function's purpose:
    input: model_params(dict), model_name(str), predict_results(list), save_path(str)

    df_task_1: Save the probability that whether the user in test set is active in the next 'future_day' days.
    df_task_2: Save the probability that whether the user in test set is active in each 'future_day' days.
    df_task_3: Save the probability that whether the user in test set is active in each 'future_day' each action.
"""
def cal_save_result(model_params, model_name, predict_results, save_path):
    if model_name == 'MyModel':
        user_ids, y_preds_1, y_truths_1, y_preds_2, y_truths_2 = predict_results
        user_ids = user_ids.astype(int)
        y_preds_1 = np.round(y_preds_1, 5)
        y_preds_2 = np.round(y_preds_2, 5)
    else:
        user_ids, y_preds_1, y_truths_1 = predict_results
        user_ids = user_ids.astype(int)
        y_preds_1 = np.round(y_preds_1, 5)

    file_name = model_params['DataSet'] + '_' + model_name + '_' + str(model_params['day']) + '_' + \
                str(model_params['future_day']) + '_' + str(model_params['multi_task_enable']) + '_' + str(
        model_params['fine_grained'])

    df_task_1 = pd.DataFrame(None, columns=['user_id', 'active_or_not'])
    df_task_1['user_id'] = user_ids
    df_task_1['active_or_not'] = y_preds_1
    df_task_1.to_csv(save_path + file_name + '_pred_task_1.csv', index=False)

    if model_name == 'MyModel' and model_params['multi_task_enable'] == 1:
        if model_params['fine_grained'] == 0:
            column = []
            for i in range(model_params['future_day']):
                column.append('day_' + str(i + 1) + '_active_or_not')
            y_preds_2 = y_preds_2.reshape(len(y_preds_2), -1)
            df_task_2 = pd.DataFrame(data=y_preds_2[0:, 0:], columns=column)
            df_task_2['user_id'] = user_ids
            df_task_2.insert(0, 'user_id', df_task_2.pop('user_id'))
            df_task_2.to_csv(save_path + file_name + '_pred_task_2.csv', index=False)

        elif model_params['fine_grained'] == 1:
            column = []
            for i in range(model_params['future_day']):
                for j in range(len(y_preds_2[0][0])):
                    column.append('day_' + str(i + 1) + '_' + str(j))
            y_preds_2 = y_preds_2.reshape(len(y_preds_2), -1)
            df_task_3 = pd.DataFrame(data=y_preds_2[0:, 0:], columns=column)
            df_task_3['user_id'] = user_ids
            df_task_3.insert(0, 'user_id', df_task_3.pop('user_id'))
            df_task_3.to_csv(save_path + file_name + '_pred_task_2.csv', index=False)


def run(epoch,
        dataset,
        model,
        optimizer,
        device,
        model_name,
        run_type,
        loss_func=None,
        write_file=None,
        model_params=None):
    user_ids = np.array([])
    y_truths_1 = np.array([])
    y_preds_1 = np.array([])
    y_truths_2 = None
    y_preds_2 = None
    all_loss = 0

    if run_type == 'train':
        model.train()
    else:
        model.eval()

    for index, value in enumerate(dataset):
        user_id, ui, uv, ai, av, y, time = value
        user_id = user_id.to(device)
        ui = ui.to(device)
        uv = uv.to(device)
        ai = ai.to(device)
        av = av.to(device)
        y = y.to(device)
        time = time.to(device)

        if run_type == 'train':
            optimizer.zero_grad()

        if model_name != 'MyModel':
            y = y[:, 0].reshape(-1, 1)
            loss, y_pred_1 = model.forward(ui, uv, ai, av, y, time, loss_func)
            # Record each user's user_id
            user_ids = np.concatenate((user_ids, user_id.reshape(-1).detach().cpu().numpy()), axis=0)
            # y_truths is the list of proportion of active days.
            y_truths_1 = np.concatenate((y_truths_1, y.reshape(-1).detach().cpu().numpy()), axis=0)
            # y_preds is the list of user active or not in future days
            y_preds_1 = np.concatenate((y_preds_1, y_pred_1.reshape(-1).detach().cpu().numpy()), axis=0)

        else:
            day = model_params['day']
            future_day = model_params['future_day']
            batch_size = model_params['batch_size']

            # y_1_input: [batch_size, 1], represent whether active within the next 'future_day'
            y_1_input = y[:, 0].reshape(-1, 1)

            # y_2: [batch_size, (day + future_day) * a_field_size]
            y_2 = y[:, 2:].detach().to(device)
            y_2 = y_2.reshape(batch_size, day + future_day, -1)
            y_2 = y_2[:, day:, :]
            y_2_input = y_2.clone()
            one = torch.ones_like(y_2_input)
            zero = torch.zeros_like(y_2_input)
            # y_2_input: [batch_size, future_day, a_field_size],
            # represents the daily activity level for the next 'future_day', and predict each action's number
            y_2_input = torch.where(y_2_input == 0, zero, one)

            # time: [batch_size, day + future_day, 4]
            time = time.to(device)

            loss, y_pred_1, y_pred_2 = model.forward(ui, uv, ai, av, y_1_input, y_2_input, time)
            user_ids = np.concatenate((user_ids, user_id.reshape(-1).detach().cpu().numpy()), axis=0)
            y_truths_1 = np.concatenate((y_truths_1, y_1_input.reshape(-1).detach().cpu().numpy()), axis=0)
            y_preds_1 = np.concatenate((y_preds_1, y_pred_1.reshape(-1).detach().cpu().numpy()), axis=0)

            if y_truths_2 is None:
                y_truths_2 = y_2_input.detach().cpu().numpy()
                y_preds_2 = y_pred_2.detach().cpu().numpy()
            else:
                y_truths_2 = np.concatenate((y_truths_2, y_2_input.detach().cpu().numpy()), axis=0)
                y_preds_2 = np.concatenate((y_preds_2, y_pred_2.detach().cpu().numpy()), axis=0)

        if run_type == 'train':
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if epoch != -1:
        run_type = "train: epoch " + str(epoch)

    # Model evaluation
    if model_name != 'MyModel':
        predict_results = user_ids, y_preds_1, y_truths_1
    else:
        predict_results = user_ids, y_preds_1, y_truths_1, y_preds_2, y_truths_2

    # Save result
    if run_type == 'test' and model_params['save_result']:
        save_path = './log/result_predict/'
        cal_save_result(model_params, model_name, predict_results, save_path)

    return cal_eval_result(all_loss, predict_results, run_type, model_name, write_file)
