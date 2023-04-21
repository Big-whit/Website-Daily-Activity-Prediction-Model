import pandas as pd
import datetime
from sklearn import preprocessing


def extraction_data(source_file_path, save_file_path, day):
    # activity: ['enrollment_id', 'time', 'source', 'event', 'object']
    activity_train = pd.read_csv(source_file_path + 'log_train.csv')
    activity_test = pd.read_csv(source_file_path + 'log_test.csv')
    # register: ['enrollment_id', 'username', 'course_id']
    register_train = pd.read_csv(source_file_path + 'enrollment_train.csv')
    register_test = pd.read_csv(source_file_path + 'enrollment_test.csv')
    # truth: ['enrollment_id', 'truth']
    truth_train = pd.read_csv(source_file_path + 'truth_train.csv', names=['enrollment_id', 'truth'])
    truth_test = pd.read_csv(source_file_path + 'truth_test.csv', names=['enrollment_id', 'truth'])
    # course_info: ['course_id', 'from', 'to']
    course_info = pd.read_csv(source_file_path + 'date.csv')

    # Step 1 - Merge files
    activity = pd.concat([activity_train, activity_test], axis=0)
    register = pd.concat([register_train, register_test], axis=0)
    truth = pd.concat([truth_train, truth_test], axis=0)
    activity = activity.drop_duplicates()
    register = register.drop_duplicates()
    truth = truth.drop_duplicates()

    # Step 2 - Delete useless fields
    # activity: ['enrollment_id', 'time', 'event']
    activity = activity.drop(['source', 'object'], axis=1)
    # register: ['enrollment_id', 'course_id']
    register = register.drop(['username'], axis=1)

    # Step 3 - Process time, each course last 30 days
    # course: ['enrollment_id', 'course_id', 'from', 'to']
    course = pd.merge(register, course_info, on='course_id', how='inner')
    # activity: ['enrollment_id', 'time', 'event', 'course_id', 'from', 'to']
    activity = pd.merge(activity, course, on='enrollment_id', how='left')
    # activity: ['enrollment_id', 'time', 'event', 'from', 'to']
    activity = activity.drop(['course_id'], axis=1)
    activity['time'] = pd.to_datetime(activity['time'])
    activity['from'] = pd.to_datetime(activity['from'])
    activity['to'] = pd.to_datetime(activity['to'])
    activity['day'] = (activity['time'] - activity['from']).dt.days + 1
    activity = activity.drop(['time', 'from', 'to'], axis=1)
    # activity: ['enrollment_id', 'day', 'event']
    activity = activity[['enrollment_id', 'day', 'event']]

    # Step 4 - Add user image
    register = register.drop(['course_id'], axis=1)
    # register: ['enrollment_id', 'user_image_1', 'user_image_2']
    register['user_image_1'] = register['user_image_2'] = 0
    # activity: ['enrollment_id', 'day', 'event', 'user_image_1', 'user_image_2']
    activity['user_image_1'] = activity['user_image_2'] = 0

    # Step 5 - Save files
    activity.to_csv(save_file_path + 'log/total_activity_log.csv', index=False)
    print('Total activity data is created successfully')
    register.to_csv(save_file_path + 'info/kddcup2015_user_info.csv', index=False)
    print("Kddcup2015 user info is created successfully")


def process_data(total_activity_log_file_path, user_info_file_path, file_path, day, future_day):
    total_activity_log = pd.read_csv(total_activity_log_file_path)
    register = pd.read_csv(user_info_file_path)

    # step 1 - Saving 'user_id' field for subsequent action.
    reg_user_id = register.copy()
    # reg_user_id: ['enrollment_id']
    reg_user_id = reg_user_id.drop(['user_image_1', 'user_image_2'], axis=1)

    # Step 2 - Count and save user activities from 1 to 'day', if there is no activity record, it will be regarded as 0.
    action_type_dict = ['problem', 'video', 'access', 'wiki', 'discussion', 'navigate', 'page_close']
    for i in range(1, day + 1):
        temporary_log_file = total_activity_log.copy()
        temporary_log_file = temporary_log_file[temporary_log_file['day'] == i]
        temporary_log_file.rename(columns={'day': 'day_id'}, inplace=True)
        day_log = temporary_log_file.groupby('enrollment_id').agg({'event': 'count'})
        day_log.rename(columns={'event': 'all#num'}, inplace=True)
        for a in action_type_dict:
            action_ = (temporary_log_file['event'] == a)
            temporary_log_file[a + '#num'] = action_
            action_num = temporary_log_file.groupby('enrollment_id').sum(numeric_only=True)[[a + '#num']]
            day_log = pd.merge(day_log, action_num, left_index=True, right_index=True)
        del temporary_log_file
        day_log = pd.merge(day_log, register, how='right', on='enrollment_id')
        day_log = day_log.fillna(0)
        day_log.insert(loc=1, column='day', value=i)
        # day_log: ['enrollment_id', 'day_id', 'all#num', 'problem#num', 'video#num', 'access#num', 'wiki#num',
        # 'discussion#num', 'navigate#num', 'page_close#num', 'user_image_1', 'user_image_2']
        day_log.to_csv(file_path + 'log/day_' + str(i) + '_activity_log.csv', index=False)
        print('Day ' + str(i) + ' activity log is created successfully')
        del day_log

    # Step 3 - Add truth information, 1 means active, 0 means inactive.
    future_day_activity_log = total_activity_log.copy()
    future_day_activity_log = future_day_activity_log[
        (day + 1 <= future_day_activity_log['day']) & (future_day_activity_log['day'] <= day + 1 + future_day)]
    truth = future_day_activity_log.groupby('enrollment_id').count()[['event']]
    truth[truth.event > 0] = 1
    truth.columns = ['truth']
    user_truth = pd.merge(reg_user_id, truth, how='left', on='enrollment_id')
    user_truth['truth'] = user_truth['truth'].fillna(0)
    # register: ['enrollment_id', 'user_image_1', 'user_image_1', 'truth']
    register = pd.merge(register, user_truth, how='left', on='enrollment_id')
    print('User add truth information over')

    # Step 4 - Add whether the 'future_day' day is active or not
    # register: ['enrollment_id', 'user_image_1', 'user_image_2', 'truth', 'first_day', 'second_day', ..., 'future_day']
    for i in range(day + 1, day + future_day + 1):
        temporary_log_file = total_activity_log.copy()
        temporary_log_file = temporary_log_file[temporary_log_file['day'] == i]
        future_truth = temporary_log_file.groupby('enrollment_id').count()[['event']]
        del temporary_log_file
        future_truth[future_truth.event > 0] = 1
        future_truth.columns = ['day' + str(i - day)]
        user_future_truth = pd.merge(reg_user_id, future_truth, how='left', on='enrollment_id')
        del future_truth
        user_future_truth['day' + str(i - day)] = user_future_truth['day' + str(i - day)].fillna(0)
        register = pd.merge(register, user_future_truth, how='left', on='enrollment_id')
        del user_future_truth
    print('Kddcup2015 user info add ' + str(day + 1) + ' to ' + str(day + future_day) + ' activity over')

    # Step 5 - Add day+1 ~ day+future_day total active day information
    # register: ['enrollment_id', 'user_image_1', 'user_image_2', 'truth', 'first_day', 'second_day', ..., 'future_day',
    # 'total_activity_num']
    register['total_activity_day'] = 0
    for i in range(day + 1, day + future_day + 1):
        register['total_activity_day'] += register['day' + str(i - day)]
    print('Kddcup2015 user info add total activity day over')

    # Step 6 - Redefine truth: truth = total_activity_day / future_day
    # register: ['enrollment_id', 'user_image_1', 'user_image_2', 'truth', 'first_day', 'second_day', ..., 'future_day',
    # 'total_activity_num', 'truth_rate']
    register['truth_rate'] = register['total_activity_day'] / future_day
    register.to_csv(file_path + 'info/kddcup2015_user_info.csv', index=False)


def create_act_statistic_info(file_path, save_path, day, future_day, act_num):
    # total_log: ['enrollment_id', 'day', 'event', 'user_image_1', 'user_image_2']
    total_log = pd.read_csv(file_path + 'log/total_activity_log.csv')
    # total_log: ['enrollment_id', 'day', 'event]
    total_log = total_log.drop(['user_image_1', 'user_image_2'], axis=1)

    # df: ['enrollment_id']
    df = pd.read_csv(file_path + 'info/kddcup2015_user_info.csv')
    df = df["enrollment_id"]

    action_type_dict = ['problem', 'video', 'access', 'wiki', 'discussion', 'navigate', 'page_close']
    temp_dict = {'problem': 0, 'video': 1, 'access': 2, 'wiki': 3, 'discussion': 4, 'navigate': 5, 'page_close': 6}
    for i in range(1, day + future_day + 1):
        for a in action_type_dict:
            temp_day_act = total_log.copy()
            temp_day_act = temp_day_act[(temp_day_act.day == i) & (temp_day_act.event == a)]
            day_act_count = temp_day_act.groupby('enrollment_id').count()[['day', 'event']]
            day_act_count = day_act_count.drop(['event'], axis=1)
            day_act_count = day_act_count.rename(columns={'day': 'day' + str(i) + '_' + str(temp_dict.get(a))})
            df = pd.merge(df, day_act_count, how='left', on='enrollment_id')
            del temp_day_act
            del day_act_count

    # df: ['enrollment_id', 'day_1_problem', 'day_1_video', ..., 'day_30_page_close']
    df = df.fillna(0)
    df.to_csv(save_path + 'info/kddcup2015_act_statistics.csv', index=False)
    print('User act statistics information create successfully! ')


def create_date_info(file_path, save_path, day, future_day):
    df = pd.read_csv(file_path)
    data = df[['enrollment_id']]
    data['day1'] = datetime.datetime.strptime('2023-04-01', '%Y-%m-%d').date()
    data['week1'] = datetime.datetime.strptime('2023-04-01', '%Y-%m-%d').date().isoweekday()
    for i in range(day + 1, day + future_day):
        data['day' + str(i + 1 - day)] = ''
        data['week' + str(i + 1 - day)] = ''
        date = data.iloc[0][1] + datetime.timedelta(days=i)
        data['day' + str(i + 1 - day)] = date
        data['week' + str(i + 1 - day)] = date.isoweekday()
    data.to_csv(save_path + 'info/kddcup2015_time.csv', index=False)
    print('User act time information create successfully! ')


def standardscaler(file_path, days):
    scaler = preprocessing.StandardScaler()
    action_feats = ['all#num', 'problem#num', 'video#num', 'access#num', 'wiki#num',
                    'discussion#num', 'navigate#num', 'page_close#num']
    for i in range(1, days + 1):
        path = file_path + 'log/day_' + str(i) + '_activity_log.csv'
        df = pd.read_csv(path, engine='python')
        newX = scaler.fit_transform(df[action_feats])
        for j, n_f in enumerate(action_feats):
            df[n_f] = newX[:, j]
        df = df.rename(
            columns={'problem#num': '0#num', 'video#num': '1#num', 'access#num': '2#num', 'wiki#num': '3#num',
                     'discussion#num': '4#num', 'navigate#num': '5#num', 'page_close#num': '6#num'})
        df.to_csv(file_path + 'feature/day_' + str(i) + '_activity_feature.csv', index=False)
        print('Day ' + str(i) + ' activity feature is created successfully')


def create_file_by_data(day, future_day, dilution_ratio=1.0):
    # Source files path
    source_file_path = './data/kddcup2015/source_data/'
    save_file_path = './data/kddcup2015/processed_data/'

    # Activity logs and user info file path
    total_activity_log_file_path = './data/kddcup2015/processed_data/log/total_activity_log.csv'
    kddcup2015_user_info_file_path = './data/kddcup2015/processed_data/info/kddcup2015_user_info.csv'

    # Kddcup2015 data file path
    file_path = './data/kddcup2015/processed_data/'

    extraction_data(source_file_path, save_file_path, day)
    process_data(total_activity_log_file_path, kddcup2015_user_info_file_path, file_path, day, future_day)
    standardscaler(file_path, day)
    create_date_info(kddcup2015_user_info_file_path, file_path, 0, day + future_day)
    create_act_statistic_info(file_path, file_path, day, future_day, 7)


if __name__ == '__main__':
    create_file_by_data(23, 7)
