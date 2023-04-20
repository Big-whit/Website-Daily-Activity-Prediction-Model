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
def calEvalResult(all_loss, y_preds, y_truths, run_type, write_file=None):
    y_preds_bool = np.copy(y_preds)
    y_preds_bool[y_preds >= 0.5] = 1.0
    y_preds_bool[y_preds < 0.5] = 0.0

    eps = 1e-5
    y_truths_bool = np.copy(y_truths)
    y_truths_bool[y_truths >= eps] = 1.0
    y_truths_bool[y_truths < eps] = 0.0

    # rmse
    rmse = ((y_truths - y_preds) ** 2).mean() ** 0.5
    # df
    df = abs(y_truths.mean() - y_preds.mean()) / y_truths.mean()
    # MAE
    MAE = np.absolute(y_truths - y_preds).mean()

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
        write_file=None):
    y_truths = np.array([])
    y_preds = np.array([])
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

        if run_type == 'train':
            loss.backward()
            optimizer.step()
        all_loss += loss.item() / y.shape[0]

    if epoch != -1:
        run_type = "train: epoch " + str(epoch)

    if model_name != 'MyModel':
        return calEvalResult(all_loss, y_preds, y_truths, run_type, write_file)
    else:
        pass
