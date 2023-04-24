import numpy as np
import pandas as pd
import datetime
import torch
from torch.utils.data import DataLoader
from dataloader.load_data import DataSet
import matplotlib.pyplot as plt

feature = {
    'act_feat': ['0#num', '1#num', '2#num', '3#num', '4#num', '5#num'],
    'user_image_feat': ['register_type', 'device_type'],
    'id_name': ['user_id'],
    'truth_tag': ['truth_rate', 'total_activity_day', 'day1', 'day2', 'day3', 'day4', 'day5', 'day6', 'day7'],
    'feat_num': 6,
}
params = {}


def dataparse(df_u):
    all_data = df_u
    # feature dimension
    feat_dim = 0
    feat_dict = dict()
    for f in feature['user_image_feat']:
        cat_val = all_data[f].unique()
        # Packed into a dictionary, such as {0:0, 1:1, 2:2}, feat_dam is cumulative
        feat_dict[f] = dict(zip(cat_val, range(feat_dim, len(cat_val) + feat_dim)))
        feat_dim += len(cat_val)

    u_data_index = all_data.copy()
    data_value = all_data.copy()
    for f in all_data.columns:
        if f in feature['user_image_feat']:
            u_data_index[f] = u_data_index[f].map(feat_dict[f])
            data_value[f] = 1.
        else:
            u_data_index.drop(f, axis=1, inplace=True)
            data_value.drop(f, axis=1, inplace=True)

    return feat_dim, u_data_index, data_value


def user_activate_day_count(future_day, label):
    day_list = []
    for i in range(0, future_day + 1):
        cur_day_count = (label['total_activity_day'] == i).sum()
        day_list.append(cur_day_count)
    day_numpy = np.array(day_list)
    return day_numpy


def dataloader():
    df_u = pd.DataFrame()
    df_a = pd.DataFrame()
    df_id = pd.DataFrame()
    past_day = 23
    future_day = 7
    data_name = 'kwai'
    batch_size = 32
    feature_file_path = './data/kwai/processed_data/feature/'
    label_file_path = './data/kwai/processed_data/info/'
    time_file_path = './data/kwai/processed_data/info/'

    for i in range(1, past_day + 1):
        file_name = 'day_' + str(i) + '_activity_feature.csv'
        df = pd.read_csv(feature_file_path + file_name)
        temp_df_id = df[feature['id_name']]
        df_id = pd.concat([df_id, temp_df_id])
        df_id.drop_duplicates(subset=feature['id_name'], inplace=True)

    for i in range(1, past_day + 1):
        file_name = 'day_' + str(i) + '_activity_feature.csv'
        df = pd.read_csv(feature_file_path + file_name)
        temp_df_u = df[feature['id_name'] + feature['user_image_feat']]
        temp_df_a = df[feature['id_name'] + feature['act_feat']]
        temp_df_a = pd.merge(df_id, temp_df_a, on=feature['id_name'], how='left')
        temp_df_a = temp_df_a.fillna(0)
        df_u = pd.concat([df_u, temp_df_u])
        df_u.drop_duplicates(subset=feature['id_name'], inplace=True)
        df_a = pd.concat([df_a, temp_df_a])

    file_name = data_name + '_user_info.csv'
    label = pd.read_csv(label_file_path + file_name)
    label = label[feature['id_name'] + feature['truth_tag']]
    label = pd.merge(df_id, label, on=feature['id_name'], how='left')

    file_name = data_name + '_time.csv'
    df_time = pd.read_csv(time_file_path + file_name)
    df_time = pd.merge(df_id, df_time, on=feature['id_name'], how='left')
    df_time.sort_values(feature['id_name'], inplace=True)
    del df_time[feature['id_name'][0]]
    df_time_split = pd.DataFrame()
    for i in range(1, past_day + future_day + 1):
        df_time_split['year' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.year
        df_time_split['month' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.month
        df_time_split['day' + str(i)] = pd.to_datetime(df_time["day" + str(i)]).dt.day
        df_time_split['week' + str(i)] = df_time["week" + str(i)]

    # print(df_a)
    """
           user_id     0#num     1#num     2#num      3#num     4#num     5#num
    0       744025  1.938814  0.911943 -0.054015  -0.033281  0.000000 -0.005168
    1      1270299  3.033775 -0.057995 -0.054015  15.211075  0.000000 -0.005168
    2       571220  0.374584 -0.057995 -0.054015  -0.033281  0.000000 -0.005168
    3      1308501  2.408083 -0.057995 -0.054015  -0.033281  0.000000 -0.005168
    4       745554  0.113879 -0.057995 -0.054015  -0.033281  0.000000 -0.005168
    ...        ...       ...       ...       ...        ...       ...       ...
    37441   845775 -0.140228 -0.073824 -0.069553  -0.042746 -0.005168 -0.012059
    37442  1050025 -0.140228 -0.073824 -0.069553  -0.042746 -0.005168 -0.012059
    37443   990897 -0.140228 -0.073824 -0.069553  -0.042746 -0.005168 -0.012059
    37444   126272 -0.140228 -0.073824 -0.069553  -0.042746 -0.005168 -0.012059
    37445   265870 -0.140228 -0.073824 -0.069553  -0.042746 -0.005168 -0.012059
    [149784 rows x 7 columns]
    """
    # print(df_u)
    """
           user_id  register_type  device_type
    0       744025              1          283
    1      1270299              1          259
    2       571220              1            2
    3      1308501              0           23
    4       745554              2            0
    ...        ...            ...          ...
    37441   845775              1          555
    37442  1050025              1          904
    37443   990897              0           60
    37444   126272              1          322
    37445   265870              0           22
    [37446 rows x 3 columns]
    """
    # print(label)
    """
           user_id  day1_1  day1_2  ...  day30_6  truth  total_activity_day
    0       744025    39.0     1.0  ...      0.0    0.0                 0.0
    1      1270299    60.0     0.0  ...      0.0    0.0                 0.0
    2       571220     9.0     0.0  ...      0.0    1.0                 5.0
    3      1308501    48.0     0.0  ...      0.0    1.0                 2.0
    4       745554     4.0     0.0  ...      0.0    1.0                 1.0
    ...        ...     ...     ...  ...      ...    ...                 ...
    37441   845775     0.0     0.0  ...      0.0    0.0                 0.0
    37442  1050025     0.0     0.0  ...      0.0    0.0                 0.0
    37443   990897     0.0     0.0  ...      0.0    1.0                 5.0
    37444   126272     0.0     0.0  ...      0.0    1.0                 7.0
    37445   265870     0.0     0.0  ...      0.0    1.0                 1.0
    [37446 rows x 183 columns]
    """
    # print(df_time_split)
    """
           year1  month1  day1  week1  ...  year30  month30  day30  week30
    14722   2023       4     1      6  ...    2023        4     30       7
    5981    2023       4     1      6  ...    2023        4     30       7
    21624   2023       4     1      6  ...    2023        4     30       7
    17301   2023       4     1      6  ...    2023        4     30       7
    24850   2023       4     1      6  ...    2023        4     30       7
    ...      ...     ...   ...    ...  ...     ...      ...    ...     ...
    36545   2023       4     1      6  ...    2023        4     30       7
    16590   2023       4     1      6  ...    2023        4     30       7
    29245   2023       4     1      6  ...    2023        4     30       7
    12060   2023       4     1      6  ...    2023        4     30       7
    35424   2023       4     1      6  ...    2023        4     30       7
    [37446 rows x 120 columns]
    """

    u_feat_dim, u_data_index, u_data_value = dataparse(df_u)

    ui = np.asarray(u_data_index.loc[df_u.index], dtype=int)
    uv = np.asarray(u_data_value.loc[df_u.index], dtype=np.float32)
    params["u_feat_size"] = u_feat_dim
    params["u_field_size"] = len(ui[0])
    ai = np.asarray([range(len(feature['act_feat'])) for x in range(len(df_a))], dtype=int)
    av = np.asarray(df_a[feature['act_feat']], dtype=np.float32)
    params["a_feat_size"] = len(av[0])
    params["a_field_size"] = len(ai[0])
    av = av.reshape((-1, past_day, len(feature['act_feat'])))
    ai = ai.reshape((-1, past_day, len(feature['act_feat'])))
    params['act_feat_num'] = len(feature['act_feat'])
    time_npy = np.asarray(df_time_split, dtype=np.float32)
    time_npy = time_npy.reshape((-1, past_day + future_day, 4))
    y = np.asarray(label[feature['truth_tag']], dtype=np.float32)

    # print(ui.shape)
    # print(uv.shape)
    # print(ai.shape)
    # print(av.shape)
    # print(y.shape)
    # print(time_npy.shape)
    """
    (37446, 2)
    (37446, 2)
    (149784, 6)
    (149784, 6)
    (37446, 182)
    (37446, 30, 4)
    """

    ui = torch.tensor(ui)
    uv = torch.tensor(uv)
    ai = torch.tensor(ai)
    av = torch.tensor(av)
    time = torch.tensor(time_npy)
    y = torch.tensor(y)

    data_num = len(y)
    indices = np.arange(data_num)
    np.random.seed(1)
    np.random.shuffle(indices)
    split_1 = int(0.6 * data_num)
    split_2 = int(0.8 * data_num)
    ui_train, ui_valid, ui_test = ui[indices[:split_1]], ui[indices[split_1:split_2]], ui[indices[split_2:]]
    uv_train, uv_valid, uv_test = uv[indices[:split_1]], uv[indices[split_1:split_2]], uv[indices[split_2:]]
    ai_train, ai_valid, ai_test = ai[indices[:split_1]], ai[indices[split_1:split_2]], ai[indices[split_2:]]
    av_train, av_valid, av_test = av[indices[:split_1]], av[indices[split_1:split_2]], av[indices[split_2:]]
    time_train, time_valid, time_test = time[indices[:split_1]], time[indices[split_1:split_2]], \
                                        time[indices[split_2:]]
    y_train, y_valid, y_test = y[indices[:split_1]], y[indices[split_1:split_2]], y[indices[split_2:]]

    # print(ui_train.shape)
    # print(uv_train.shape)
    # print(ai_train.shape)
    # print(av_train.shape)
    # print(time_train.shape)
    # print(y_train.shape)

    """torch.Size([22467, 2])
    torch.Size([22467, 2])
    torch.Size([22467, 6])
    torch.Size([22467, 6])
    torch.Size([22467, 30, 4])
    torch.Size([22467, 182])

    22467 = 37446 * 0.6
    """

    train_dataset = DataSet(ui_train, uv_train, ai_train, av_train, y_train, time_train)
    valid_dataset = DataSet(ui_valid, uv_valid, ai_valid, av_valid, y_valid, time_valid)
    test_dataset = DataSet(ui_test, uv_test, ai_test, av_test, y_test, time_test)
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_set, valid_set, test_set


def test_for_dataloader(multi_task_enable=True):
    train_set, valid_set, test_set = dataloader()
    for index, value in enumerate(train_set):
        ui, uv, ai, av, y, time = value
        y_1 = y[:, 2:].detach()
        one = torch.ones_like(y_1)
        zero = torch.zeros_like(y_1)
        # 未来每天是否活跃
        # y_1: [batch_size, future_day]
        y_1 = torch.where(y_1 == 0, zero, one)

        # 总活跃天数
        # y_2: [batch_size, 1]
        y_2 = y[:, 1].detach().long()
        # 未来 7 天活跃了几天，one-hot 编码
        # y_2_input: [batch_size, future_day + 1]
        y_2_input = torch.nn.functional.one_hot(y_2, num_classes=7 + 1).float()

        if multi_task_enable == True:
            # 未来 7 天的活跃比
            y_2 = y_2 / 7
        else:
            # 未来 7 天是否活跃
            y_2 = y[:, 0]

        y_2_labels = torch.argmax(y_2_input, 1) / (y_2_input.shape[1])
        print(y_2_labels)

        # print(ui.shape)           # torch.Size([32, 2])
        # print(uv.shape)           # torch.Size([32, 2])
        # print(ai.shape)           # torch.Size([32, 23, 6])
        # print(av.shape)           # torch.Size([32, 23, 6])
        # print(y.shape)            # torch.Size([32, 9])
        # print(time.shape)         # torch.Size([32, 30, 4])


# **************************************** For tool.py **************************************** #
def setMask(y_truth, y_pred, proportions=None):
    if proportions is None:
        proportions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    filtered_truth = []
    filtered_pred = []
    for i in range(0, len(proportions) - 1):
        mask = torch.logical_and(y_truth.ge(proportions[i]), y_truth.lt(proportions[i + 1]))
        filtered_truth.append(torch.masked_select(y_truth, mask))
        filtered_pred.append(torch.masked_select(y_pred, mask))

    return filtered_truth, filtered_pred


def test_for_model_and_run():
    y_truth = torch.rand(32, 1)
    y_pred = torch.rand(32, 1)

    filtered_truth, filtered_pred = setMask(y_truth, y_pred)
    print(filtered_truth)
    print(filtered_pred)


# **************************************** For draw.py **************************************** #
def draw_result():
    model_name = ['CFIN', 'CLSA', 'LSCNN', 'DPCNN', 'LR', 'RNN']
    # kwai_23_7
    # rmse = [0.27336, 0.25374, 0.23214, 0.23450, 0.52988, 0.23440]
    # df = [0.05942, 0.02866, 0.05132, 0.01938, 0.07954, 0.03372]
    # mae = [0.19674, 0.17874, 0.16510, 0.16286, 0.24940, 0.16644]

    # kddcup2015_23_7
    rmse = [0.12244, 0.11458, 0.10908, 0.12224, 0.13368, 0.12236]
    df = [0.06394, 0.04886, 0.04098, 0.01650, 0.09774, 0.03802]
    mae = [0.08476, 0.06450, 0.06140, 0.08146, 0.09090, 0.07170]

    plt.plot(model_name, rmse, 'r', marker='.', markersize=4, label='RMSE')
    plt.plot(model_name, df, 'g', marker='*', markersize=4, label='df')
    plt.plot(model_name, mae, 'b', marker='X', markersize=4, label='MAE')
    plt.xlabel("Model Name")
    plt.ylabel("Value")
    plt.title("Kddcup2015 dataset, use 23 to predict 7")
    plt.legend(loc="upper left")
    for x1, y1 in zip(model_name, rmse):
        plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
    for x1, y1 in zip(model_name, df):
        plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
    for x1, y1 in zip(model_name, mae):
        plt.text(x1, y1, str(y1), ha='center', va='bottom', fontsize=10)
    plt.show()


# ****************************************************************************************** #
if __name__ == '__main__':
    a = np.load('./kwai_MyModel_23_7_truth_1.npy', allow_pickle=True)
    print(a.shape)
    print(a[1])
    print(a[2])
    print(a[3])
    print(a[4])
    print(a[5])
    print(a[6])
    print(a[7])
    print(a[8])