import pandas as pd
day = 23
future_day = 7

total_activity_log = pd.read_csv('./data/kwai/processed_data/total_activity_log.csv')
register = pd.read_csv('./data/kwai/processed_data/kwai_user_info.csv')
# step 1 - Saving 'user_id' field for subsequent action.
# reg_user_id: ['user_id']
reg_user_id = register.copy()
reg_user_id = reg_user_id.drop(['register_type', 'device_type'], axis=1)
# Step 2 - Count and save user activities from 1 to 'day', if there is no activity record, it will be regarded as 0.
action_type_dict = [0, 1, 2, 3, 4, 5]
for i in range(1, 2):
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
    day_log.to_csv('./data/kwai/processed_data/' + 'day_' + str(i) + '_activity_log.csv', index=False)
    print('Day ' + str(i) + ' activity log is created successfully')
    del day_log

# Step 3 - Add truth information, 1 means active, 0 means inactive.
future_day_activity_log = total_activity_log.copy()
future_day_activity_log = future_day_activity_log[(day + 1 <= future_day_activity_log['act_day']) & (future_day_activity_log['act_day'] <= day + 1 + future_day)]
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
print('Kwai user info add ' + str(day + 1) + ' to ' + str(day + 1 + future_day + 1) + ' activity over')