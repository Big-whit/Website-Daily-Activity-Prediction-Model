import pandas as pd
import datetime
from sklearn import preprocessing

"""
extraction_data() function's purpose:
    1 - Filter the users who are 'day' before the number of registration days and save it as kwai_user_info.csv.
    2 - Obtain all activity information of users who registered in the previous 'day' and save it as all_activity_log.csv.
"""
def extraction_data(activity_log_file_path, register_file_path, file_path, day):
    activity_log = pd.read_csv(activity_log_file_path, sep='\t',
                               names=['user_id', 'act_day', 'page', 'video_id', 'author_id', 'act_type'])
    register = pd.read_csv(register_file_path, sep='\t',
                           names=['user_id', 'register_day', 'register_type', 'device_type'])

    # Step 1 - Filter users whose registration date is ≤ day.
    register = register[register['register_day'] <= day]
    # Step 2 - Delete 'register_day' field.
    # register: ['user_id', 'register_type', 'device_type']
    register = register.drop(['register_day'], axis=1)
    register.to_csv(file_path + 'info/kwai_user_info.csv', index=False)
    print('Kwai user info is created successfully')

    # Step 3 - Delete 'page'、'video_id'、'author_id' fields from activity log.
    # activity_log: ['user_id', 'act_day', 'act_type']
    activity_log = activity_log.drop(['page', 'video_id', 'author_id'], axis=1)
    # Step 4 - Merge register info and activity log.
    # activity_log: ['user_id', 'act_day', 'act_type', 'register_type', 'device_type']
    activity_log = pd.merge(activity_log, register, how='inner', on='user_id')
    activity_log.to_csv(file_path + 'log/total_activity_log.csv', index=False)
    print('Total activity data is created successfully')


"""
preprocess_data() function's purpose:
    1 - Count the actions of each registered user between 1 ~ day.
    2 - Determine whether each registered user is active within the future future_day、
     whether it is active every day in the future and the active rate.
"""
def preprocess_data(total_activity_log_file_path, kwai_user_info_file_path, file_path, day, future_day):
    total_activity_log = pd.read_csv(total_activity_log_file_path)
    register = pd.read_csv(kwai_user_info_file_path)

    # step 1 - Saving 'user_id' field for subsequent action.
    # reg_user_id: ['user_id']
    reg_user_id = register.copy()
    reg_user_id = reg_user_id.drop(['register_type', 'device_type'], axis=1)

    # Step 2 - Count and save user activities from 1 to 'day', if there is no activity record, it will be regarded as 0.
    action_type_dict = [0, 1, 2, 3, 4, 5]
    for i in range(1, day + 1):
        temporary_log_file = total_activity_log.copy()
        temporary_log_file = temporary_log_file[temporary_log_file['act_day'] == i]
        temporary_log_file.rename(columns={'act_day': 'day_id'}, inplace=True)
        day_log = temporary_log_file.groupby('user_id').agg({'act_type': 'count'})
        day_log.rename(columns={'act_type': 'all#num'}, inplace=True)
        for a in action_type_dict:
            action_ = (temporary_log_file['act_type'] == a)
            temporary_log_file[str(a) + '#num'] = action_
            action_num = temporary_log_file.groupby('user_id').sum(numeric_only=True)[[str(a) + '#num']]
            day_log = pd.merge(day_log, action_num, left_index=True, right_index=True)
        del temporary_log_file
        day_log = pd.merge(day_log, register, how='right', on='user_id')
        # Fill NAN as 0
        day_log = day_log.fillna(0)
        day_log.insert(loc=1, column='day_id', value=i)
        # day_log: ['user_id', 'day_id', 'all#num', '0#num', '1#num', '2#num', '3#num', '4#num', '5#num',
        # 'register_type', 'device_type']
        day_log.to_csv(file_path + 'log/day_' + str(i) + '_activity_log.csv', index=False)
        print('Day ' + str(i) + ' activity log is created successfully')
        del day_log

    # Step 3 - Add truth information, 1 means active, 0 means inactive.
    future_day_activity_log = total_activity_log.copy()
    future_day_activity_log = future_day_activity_log[
        (day + 1 <= future_day_activity_log['act_day']) & (future_day_activity_log['act_day'] <= day + 1 + future_day)]
    truth = future_day_activity_log.groupby('user_id').count()[['act_type']]
    truth[truth.act_type > 0] = 1
    truth.columns = ['truth']
    # user_truth: ['user_id', 'truth']
    user_truth = pd.merge(reg_user_id, truth, how='left', on='user_id')
    # Fill NAN as 0
    user_truth['truth'] = user_truth['truth'].fillna(0)
    # register: ['user_id', 'register_type', 'device_type', 'truth']
    register = pd.merge(register, user_truth, how='inner', on='user_id')
    print('User add truth information over')

    # Step 4 - Add whether the 'future_day' day is active or not
    # register: ['user_id', 'register_type', 'device_type', 'truth', 'first_day', 'second_day', ..., 'future_day']
    for i in range(day + 1, day + future_day + 1):
        temporary_log_file = total_activity_log.copy()
        temporary_log_file = temporary_log_file[temporary_log_file['act_day'] == i]
        future_truth = temporary_log_file.groupby('user_id').count()[['act_type']]
        del temporary_log_file
        future_truth[future_truth.act_type > 0] = 1
        future_truth.columns = ['day' + str(i - day)]
        user_future_truth = pd.merge(reg_user_id, future_truth, how='left', on='user_id')
        del future_truth
        user_future_truth['day' + str(i - day)] = user_future_truth['day' + str(i - day)].fillna(0)
        register = pd.merge(register, user_future_truth, how='inner', on='user_id')
        del user_future_truth
    print('Kwai user info add ' + str(day + 1) + ' to ' + str(day + future_day) + ' activity over')

    # Step 5 - Add day+1 ~ day+future_day total active day information
    # register: ['user_id', 'register_type', 'device_type', 'truth', 'first_day', 'second_day', ..., 'future_day',
    # 'total_activity_num']
    register['total_activity_day'] = 0
    for i in range(day + 1, day + future_day + 1):
        register['total_activity_day'] += register['day' + str(i - day)]
    print('Kwai user info add total activity day over')

    # Step 6 - Redefine truth: truth = total_activity_day / future_day
    # register: ['user_id', 'register_type', 'device_type', 'truth', 'first_day', 'second_day', ..., 'future_day',
    # 'total_activity_num', 'truth_rate']
    register['truth_rate'] = register['total_activity_day'] / future_day
    register.to_csv(file_path + 'info/kwai_user_info.csv', index=False)


"""
standardscaler() function's purpose: 
    1 - For standardization. 
"""
def standardscaler(file_path, days):
    scaler = preprocessing.StandardScaler()
    action_feats = ['all#num', '0#num', '1#num', '2#num', '3#num', '4#num', '5#num']
    for i in range(1, days + 1):
        path = file_path + 'log/day_' + str(i) + '_activity_log.csv'
        df = pd.read_csv(path, engine='python')
        newX = scaler.fit_transform(df[action_feats])
        for j, n_f in enumerate(action_feats):
            df[n_f] = newX[:, j]
        df.to_csv(file_path + 'feature/day_' + str(i) + '_activity_feature.csv', index=False)
        print('Day ' + str(i) + ' activity feature is created successfully')


"""
create_date_info() function's purpose:
    1 - Assign activity logs to time information.
"""
def create_date_info(file_path, save_path, day, future_day):
    df = pd.read_csv(file_path)
    data = df[['user_id']]
    data['day1'] = datetime.datetime.strptime('2023-04-01', '%Y-%m-%d').date()
    data['week1'] = datetime.datetime.strptime('2023-04-01', '%Y-%m-%d').date().isoweekday()
    for i in range(day + 1, day + future_day):
        data['day' + str(i + 1 - day)] = ''
        data['week' + str(i + 1 - day)] = ''
        date = data.iloc[0][1] + datetime.timedelta(days=i)
        data['day' + str(i + 1 - day)] = date
        data['week' + str(i + 1 - day)] = date.isoweekday()
    data.to_csv(save_path + 'info/kwai_time.csv', index=False)


def create_file_by_data(day, future_day, dilution_ratio=1.0):
    # Source files path
    activity_log_file_path = './data/kwai/source_data/user_activity_log.txt'
    register_file_path = './data/kwai/source_data/user_register_log.txt'

    # Activity logs and user info file path
    total_activity_log_file_path = './data/kwai/processed_data/log/total_activity_log.csv'
    kwai_user_info_file_path = './data/kwai/processed_data/info/kwai_user_info.csv'

    # Kwai data file path
    file_path = './data/kwai/processed_data/'

    extraction_data(activity_log_file_path, register_file_path, file_path, day)
    preprocess_data(total_activity_log_file_path, kwai_user_info_file_path, file_path, day, future_day)
    standardscaler(file_path, day)
    create_date_info(kwai_user_info_file_path, file_path, 0, day + future_day)


if __name__ == '__main__':
    pass
    # create_file_by_data(30, 0)
    # create_file_by_data(23, 7)
