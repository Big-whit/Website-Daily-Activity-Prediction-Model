import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score

"""
calEvalResult() function's purpose:
    input: loss value(float), y_preds([batch_size, 1](float), y_truths([batch_size, 1]))
    output: rmse, df, MAE
    
    rmse: root mean square error(The smaller the better)
    df: degrees of freedom(The smaller the better)
    MAE: mean absolute error(The smaller the better)
"""


def calEvalResult(all_loss, predict_results, run_type, model_name, write_file=None):
    if model_name == 'MyModel':
        y_preds, y_truths, y_preds_2, y_truths_2 = predict_results
    else:
        y_preds, y_truths, = predict_results

    error = y_truths - y_preds
    # RMSE
    rmse = ((error) ** 2).mean() ** 0.5
    # df
    df = abs(y_truths.mean() - y_preds.mean()) / y_truths.mean()
    # MAE
    MAE = np.absolute(error).mean()

    log_str = '%20s loss %3.6f  rmse %.4f  df(ActivateDay.Avg) %.4f  MAE %.4f' % (run_type, all_loss, rmse, df, MAE)

    if write_file is not None:
        write_file.write(log_str + "\n")

    return rmse, df, MAE


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
    y_truths = np.array([])
    y_preds = np.array([])
    y_truths_2 = None
    y_preds_2 = None
    all_loss = 0
    dataset_user_num = 0

    if run_type == 'train':
        model.train()
    else:
        model.eval()

    for index, value in enumerate(dataset):
        dataset_user_num += len(value[0])
        ui, uv, ai, av, y, time = value
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
            loss, y_pred = model.forward(ui, uv, ai, av, y, time, loss_func)
            # y_truths is the list of proportion of active days.
            y_truths = np.concatenate((y_truths, y.detach().cpu().numpy().reshape(-1)), axis=0)
            # y_preds is the list of user active or not in future days
            y_preds = np.concatenate((y_preds, y_pred.reshape(-1).detach().cpu().numpy()), axis=0)

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
            # y_2_input: [batch_size, future_day, a_field_size], represents the daily activity level for the next 'future_day', and predict each action's number
            y_2_input = torch.where(y_2_input == 0, zero, one)

            # time: [batch_size, day + future_day, 4]
            time = time.to(device)

            loss, y_pred_1, y_pred_2 = model.forward(ui, uv, ai, av, y_1_input, y_2_input, time)
            y_truths = np.concatenate((y_truths, y_1_input.detach().cpu().numpy().reshape(-1)), axis=0)
            y_preds = np.concatenate((y_preds, y_pred_1.reshape(-1).detach().cpu().numpy()), axis=0)

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

    if model_name != 'MyModel':
        predict_results = y_preds, y_truths
    else:
        predict_results = y_preds, y_truths, y_preds_2, y_truths_2

        # if run_type == 'test':
        #     pred_file_name = model_params['DataSet'] + '_' + model_name + '_' + str(model_params['day']) + '_' + \
        #                      str(model_params['future_day']) + '_pred_1.npy'
        #     np.save(pred_file_name, y_preds_2)
        #     truth_file_name = model_params['DataSet'] + '_' + model_name + '_' + str(model_params['day']) + '_' + \
        #                       str(model_params['future_day']) + '_truth_1.npy'
        #     np.save(truth_file_name, y_truths_2)

    return calEvalResult(all_loss, predict_results, run_type, model_name, write_file)
