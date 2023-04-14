from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import torch

# feature tags
feature = {
    'act_feat': [],
    'user_image_feat': [],
    'day_act_tag': [],
    'truth_tag': [],
    'id_name': [],
    'act_feat_num': 0,
}

"""
create_feature_tag() function's purpose:
    To create tags in feature dictionary, for example, if the dataset is kwai:
    feature = {
        'act_feat': ['0#num', '1#num', '2#num', '3#num', '4#num', '5#num'],
        'user_image_feat': ['register_type', 'device_type'],
        'day_act_tag': ['day1_1', 'day1_2', ..., 'day30_6'],
        'truth_tag': ['truth_rate','total_activity_day'],
        'id_name': ['user_id'],
        'act_feat_num': 6,
    }
"""
def create_feature_tag(past_day=23, future_day=7, data_name='kwai'):
    global feature
    if data_name == 'kwai':
        feature['id_name'].append('user_id')
        act_feat_num = 6
        for index in range(act_feat_num):
            feature['act_feat'].append(str(index) + '#num')
        feature['user_image_feat'].append('register_type')
        feature['user_image_feat'].append('device_type')
    elif data_name == 'kddcup2015':
        pass
    else:
        pass

    for i in range(1, past_day + future_day + 1):
        day = 'day' + str(i)
        for j in range(1, feature['act_feat_num'] + 1):
            day_num = day + '_' + str(j)
            feature['day_act_tag'].append(day_num)

    feature['truth_tag'] = ['truth_rate', 'total_activity_day']


"""
load_data() function's purpose:
    1 - Load various feature information, for example, if the dataset is kwai:
    df_u(titles): ['user_id', '0#num', '1#num',' 2#num', '3#num', '4#num', '5#num']
    df_a(titles): ['user_id', 'register_type', 'device_type']
    label(titles): ['user_id', 'day1_1', 'day1_2',..., 'day30_6', 'truth', 'total_activity_day']
    df_time_split(titles): ['year1', 'month1', 'day1', 'week1',..., 'year30', 'month30', 'day30', 'week30']
"""
def load_data(past_day, future_day, data_name, data_dilution_ratio):
    global feature
    df_u = pd.DataFrame()
    df_a = pd.DataFrame()
    df_id = pd.DataFrame()

    # Define file read path.
    feature_file_path = ''
    label_file_path = ''
    time_file_path = ''
    if data_name == 'kwai':
        feature_file_path = './data/kwai/processed_data/feature/'
        label_file_path = './data/kwai/processed_data/info/'
        time_file_path = './data/kwai/processed_data/info/'
    elif data_name == 'kddcup2015':
        feature_file_path = './data/kddcup2015/processed_data/feature/'
        label_file_path = './data/kddcup2015/processed_data/info/'
        time_file_path = './data/kddcup2015/processed_data/info'

    # Get unique user id between 1 ~ past_day.
    for i in range(1, past_day + 1):
        file_name = 'day_' + str(i) + '_activity_feature.csv'
        df = pd.read_csv(feature_file_path + file_name)
        temp_df_id = df[feature['id_name']]
        df_id = pd.concat([df_id, temp_df_id])
        df_id.drop_duplicates(subset=feature['id_name'], inplace=True)
    # user dilution
    r, c = df_id.shape
    r = int(r * data_dilution_ratio)
    df_id = df_id.iloc[:r]

    # Get the user's action statistics and user image feature between 1 ~ past_day.
    for i in range(1, past_day + 1):
        file_name = 'day_' + str(i) + '_activity_feature.csv'
        df = pd.read_csv(feature_file_path + file_name)
        temp_df_u = df[feature['id_name'] + feature['user_image_feat']]
        temp_df_a = df[feature['id_name'] + feature['act_feat']]
        temp_df_u = pd.merge(df_id, temp_df_u, on=feature['id_name'], how='left')
        temp_df_a = pd.merge(df_id, temp_df_a, on=feature['id_name'], how='left')
        temp_df_a = temp_df_a.fillna(0)
        df_u = pd.concat([df_u, temp_df_u])
        df_u.drop_duplicates(subset=feature['id_name'], inplace=True)
        df_a = pd.concat([df_a, temp_df_a])

    # Add the user image feature between past_day+1 ~ past_day+future_day.
    for i in range(past_day + 1, past_day + future_day + 1):
        file_name = 'day_' + str(i) + '_activity_feature.csv'
        df = pd.read_csv(feature_file_path + file_name)
        temp_df_u = df[feature['id_name'] + feature['user_image_feat']]
        temp_df_u = pd.merge(df_id, temp_df_u, on=feature['id_name'], how='left')
        df_u = pd.concat([df_u, temp_df_u])
        df_u.drop_duplicates(subset=feature['id_name'], inplace=True)

    # Get detailed action statistics of users between day ~ day+future_day, as well as future activity label.
    file_name = data_name + '_act_statistics.csv'
    label_1 = pd.read_csv(label_file_path + file_name)
    label_1 = label_1[feature['id_name'] + feature['day_act_tag']]
    file_name = data_name + '_user_info.csv'
    label_2 = pd.read_csv(label_file_path + file_name)
    label_2 = label_2[feature['id_name'] + feature['truth_tag']]
    label = pd.merge(df_id, label_1, on=feature['id_name'], how='left')
    label = pd.merge(label, label_2, on=feature['id_name'], how='left')

    # Sort by id_name.
    df_a.sort_values(feature['id_name'], inplace=True)
    df_u.sort_values(feature['id_name'], inplace=True)
    label.sort_values(feature['id_name'], inplace=True)

    # Get time information and split it into year、month、day、week.
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

    return df_u, df_a, label, df_time_split


"""
data_parse() function's purpose:
    1 - Transform discrete user profile information into continuous information for embedding.
    2 - Return total user image feature dimension.
"""
def data_parse(df_u):
    all_data = df_u
    # feature dimension(total fields)
    feat_dim = 0
    feat_dict = dict()
    for f in feature['user_image_feat']:
        cat_val = all_data[f].unique()
        # Packed into a dictionary, such as {0:0, 1:1, 2:2}, feat_dam is cumulative
        feat_dict[f] = dict(zip(cat_val, range(feat_dim, len(cat_val) + feat_dim)))
        feat_dim += len(cat_val)

    data_indices = all_data.copy()
    data_value = all_data.copy()
    for f in all_data.columns:
        if f in feature['user_image_feat']:
            data_indices[f] = data_indices[f].map(feat_dict[f])
            data_value[f] = 1.
        else:
            data_indices.drop(f, axis=1, inplace=True)
            data_value.drop(f, axis=1, inplace=True)

    return feat_dim, data_indices, data_value


"""
user_activate_day_count() function's purpose:
    1 - Count the number of people who are active from 0 ~ future_day in the future future_day
"""
def user_activate_day_count(future_day, label):
    day_list = []
    for i in range(0, future_day + 1):
        cur_day_count = (label['total_activity_day'] == i).sum()
        day_list.append(cur_day_count)
    day_numpy = np.array(day_list)
    return day_numpy


class DataSet(Dataset):
    def __init__(self, ui, uv, ai, av, y, time):
        super(Dataset, self).__init__()
        self.ui = ui
        self.uv = uv
        self.ai = ai
        self.av = av
        self.y = y
        self.time = time
        self.len = ui.shape[0]

    def __getitem__(self, item):
        return self.ui[item], self.uv[item], self.ai[item], self.av[item], self.y[item], self.time[item],

    def __len__(self):
        return self.len


def get_data_loader(batch_size=64, params={}, data_name='kwai'):
    past_day = params['day']
    future_day = params['future_day']
    data_dilution_ratio = params['data_dilution_ratio']
    data_dict = ['ui_train', 'uv_train', 'ai_train', 'av_train', 'y_train', 'time_train',
                 'ui_valid', 'uv_valid', 'ai_valid', 'av_valid', 'y_valid', 'time_valid',
                 'ui_test', 'uv_test', 'ai_test', 'av_test', 'y_test', 'time_test',
                 'day_numpy', 'params']
    save_path = './data/' + data_name + '/model_input_data/'
    # For example: _23_7_1_0.1
    save_name = '_' + str(past_day) + '_' + str(future_day) + '_' + str(params['seed']) + '_' + str(data_dilution_ratio)
    # For example: ./data/kwai/model_input_data/ui_train_23_7_1_0.1.pt
    if os.path.exists(save_path + data_dict[0] + save_name + '.pt'):
        ui_train = torch.load(save_path + data_dict[0] + save_name + '.pt')
        uv_train = torch.load(save_path + data_dict[1] + save_name + '.pt')
        ai_train = torch.load(save_path + data_dict[2] + save_name + '.pt')
        av_train = torch.load(save_path + data_dict[3] + save_name + '.pt')
        y_train = torch.load(save_path + data_dict[4] + save_name + '.pt')
        time_train = torch.load(save_path + data_dict[5] + save_name + '.pt')

        ui_valid = torch.load(save_path + data_dict[6] + save_name + '.pt')
        uv_valid = torch.load(save_path + data_dict[7] + save_name + '.pt')
        ai_valid = torch.load(save_path + data_dict[8] + save_name + '.pt')
        av_valid = torch.load(save_path + data_dict[9] + save_name + '.pt')
        y_valid = torch.load(save_path + data_dict[10] + save_name + '.pt')
        time_valid = torch.load(save_path + data_dict[11] + save_name + '.pt')

        ui_test = torch.load(save_path + data_dict[12] + save_name + '.pt')
        uv_test = torch.load(save_path + data_dict[13] + save_name + '.pt')
        ai_test = torch.load(save_path + data_dict[14] + save_name + '.pt')
        av_test = torch.load(save_path + data_dict[15] + save_name + '.pt')
        y_test = torch.load(save_path + data_dict[16] + save_name + '.pt')
        time_test = torch.load(save_path + data_dict[17] + save_name + '.pt')

        day_numpy = np.load(save_path + data_dict[18] + save_name + '.npy')
        params_load = np.load(save_path + data_dict[19] + save_name + '.npy', allow_pickle=True).item()

        params_load.update(params)
        params = params_load
        print("Model input data loaded")
    else:
        create_feature_tag(past_day, future_day, data_name)
        print('Feature tags create successfully')
        df_u, df_a, label, time_split = load_data(past_day, future_day, data_name, data_dilution_ratio)
        print('Load df_a、df_u、label、time_split over')

        u_feat_dim, u_data_indices, u_data_value = data_parse(df_u)
        print('The user images are serialized')

        # Turn dataframe → array
        ui = np.asarray(u_data_indices.loc[df_u.index], dtype=int)
        uv = np.asarray(u_data_value.loc[df_u.index], dtype=np.float32)
        params['u_feat_size'] = u_feat_dim
        params['u_field_size'] = len(ui[0])
        ai = np.asarray([range(len(feature['act_feat'])) for x in range(len(df_a))], dtype=int)
        av = np.asarray(df_a[feature['act_feat']], dtype=np.float32)
        params['a_feat_size'] = len(av[0])
        params['a_field_size'] = len(ai[0])
        # av、ai: [data_num, day, action_feature_num]
        av = av.reshape((-1, params['day'], len(feature['act_feat'])))
        ai = ai.reshape((-1, params['day'], len(feature['act_feat'])))
        params['act_feat_num'] = feature['act_feat_num']
        time_npy = np.asarray(time_split, dtype=np.float32)
        time_npy = time_npy.reshape((-1, past_day + future_day, 4))
        y = np.asarray(label[feature['truth_tag'] + feature['day_act_tag']], dtype=np.float32)

        # np.save(save_path + 'av' + save_name + '.npy', av)
        # np.save(save_path + 'y' + save_name + '.npy', y)

        # Turn array → tensor
        ui = torch.tensor(ui)
        uv = torch.tensor(uv)
        ai = torch.tensor(ai)
        av = torch.tensor(av)
        time = torch.tensor(time_npy)
        y = torch.tensor(y)

        # Divide training set, validation set, test set(6:2:2).
        data_num = len(y)
        indices = np.arange(data_num)
        np.random.seed(params['seed'])
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
        print('The data is divided')

        # Count the number of people who are active from 0 ~ future_day in the future future_day,
        # and use it to analyze the long-tail effect.
        label = label.iloc[indices[:split_1]]
        day_numpy = user_activate_day_count(future_day, label)
        
        # Saving model input data
        torch.save(ui_train, save_path + data_dict[0] + save_name + '.pt')
        torch.save(uv_train, save_path + data_dict[1] + save_name + '.pt')
        torch.save(ai_train, save_path + data_dict[2] + save_name + '.pt')
        torch.save(av_train, save_path + data_dict[3] + save_name + '.pt')
        torch.save(y_train, save_path + data_dict[4] + save_name + '.pt')
        torch.save(time_train, save_path + data_dict[5] + save_name + '.pt')

        torch.save(ui_valid, save_path + data_dict[6] + save_name + '.pt')
        torch.save(uv_valid, save_path + data_dict[7] + save_name + '.pt')
        torch.save(ai_valid, save_path + data_dict[8] + save_name + '.pt')
        torch.save(av_valid, save_path + data_dict[9] + save_name + '.pt')
        torch.save(y_valid, save_path + data_dict[10] + save_name + '.pt')
        torch.save(time_valid, save_path + data_dict[11] + save_name + '.pt')

        torch.save(ui_test, save_path + data_dict[12] + save_name + '.pt')
        torch.save(uv_test, save_path + data_dict[13] + save_name + '.pt')
        torch.save(ai_test, save_path + data_dict[14] + save_name + '.pt')
        torch.save(av_test, save_path + data_dict[15] + save_name + '.pt')
        torch.save(y_test, save_path + data_dict[16] + save_name + '.pt')
        torch.save(time_test, save_path + data_dict[17] + save_name + '.pt')

        np.save(save_path + data_dict[18] + save_name + '.npy', day_numpy)
        np.save(save_path + data_dict[19] + save_name+ '.npy', params)
        print('The model input data is saved')

    # packaged dataset
    # ui_train、uv_train: [user_num, user_image_feature_num]
    # ai_train、av_train: [user_num, past_day, action_feature_num]
    # time_train: [user_num, past_day + future_day, 4]
    # y: [user_num, truth + total_activity_day + day1_1 + ... + dayN_feature_num]
    train_dataset = DataSet(ui_train, uv_train, ai_train, av_train, y_train, time_train)
    valid_dataset = DataSet(ui_valid, uv_valid, ai_valid, av_valid, y_valid, time_valid)
    test_dataset = DataSet(ui_test, uv_test, ai_test, av_test, y_test, time_test)
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_set = DataLoader(valid_dataset, batch_size=batch_size, drop_last=True)
    test_set = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

    return train_set, valid_set, test_set, day_numpy, params


if __name__ == '__main__':
    pass
    # my_params = {"day": 23, "future_day": 7, "seed": 1, "data_dilution_ratio": 1.0}
    # my_params = dict(my_params)
    # get_data_loader(params=my_params)
