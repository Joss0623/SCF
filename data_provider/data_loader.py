"""Time-MMD chronological data loader and causal calendar prior."""
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        self.tot_len = len(self.data_x) - self.seq_len - self.pred_len + 1

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        if self.args.use_closedllm==0:
            text_name='Final_Search_'+str(self.args.text_len)
        else:
            print("!!!!!!!!!!!!Using output of closed source llm and Bert as encoder!!!!!!!!!!!!!!!")
            text_name="Final_Output"
        df_raw = df_raw[['date'] + cols + [self.target]+['prior_history_avg']+['start_date']+['end_date']+[text_name]]
        train_ratio = float(getattr(self.args, 'train_ratio', 0.7))
        test_ratio = float(getattr(self.args, 'test_ratio', 0.2))
        if train_ratio <= 0 or test_ratio <= 0 or train_ratio + test_ratio >= 1:
            raise ValueError(
                f'Expected positive train/test ratios summing to less than one, '
                f'got {train_ratio} and {test_ratio}'
            )
        num_train = int(len(df_raw) * train_ratio)
        num_test = int(len(df_raw) * test_ratio)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        self.border1 = border1
        self.border2 = border2

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]
            df_data_prior = df_raw[['prior_history_avg']]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            data_prior = self.scaler.transform(df_data_prior.values[:,-1].reshape(-1, 1))
        else:
            data = df_data.values
            data_prior = df_data_prior.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_prior = data_prior[border1:border2]
        self.full_data_y = data

        # Build a strictly causal, horizon-aligned historical-average prior.
        # The released ``prior_history_avg`` column cannot safely be sliced at
        # future rows: for some domains those rows summarize targets observed
        # after the current forecast origin.  Instead, each future calendar
        # phase is assigned the mean of target observations with the same phase
        # whose absolute indices are strictly smaller than the origin.
        full_dates = pd.to_datetime(df_raw['date'])
        freq = str(self.freq).lower()
        if freq.startswith('m'):
            phase_keys = [('month', int(ts.month)) for ts in full_dates]
        elif freq.startswith('w'):
            iso_weeks = full_dates.dt.isocalendar().week.astype(int)
            phase_keys = [('week', int(week)) for week in iso_weeks]
        elif freq.startswith('d'):
            phase_keys = [('day', int(ts.month), int(ts.day)) for ts in full_dates]
        else:
            phase_keys = [('position', int(i % max(1, self.pred_len)))
                          for i in range(len(full_dates))]

        self.prior_phase_keys = phase_keys
        phase_indices = {}
        for absolute_index, key in enumerate(phase_keys):
            phase_indices.setdefault(key, []).append(absolute_index)
        self.prior_phase_indices = {
            key: np.asarray(indices, dtype=np.int64)
            for key, indices in phase_indices.items()
        }
        self.prior_phase_prefix_sums = {}
        for key, indices in self.prior_phase_indices.items():
            values = self.full_data_y[indices, 0].astype(np.float64)
            self.prior_phase_prefix_sums[key] = np.concatenate(
                [np.zeros(1, dtype=np.float64), np.cumsum(values)]
            )


        self.data_stamp = data_stamp
        self.date=df_raw[['date']][border1:border2].values
        self.start_date=df_raw[['start_date']][border1:border2].values
        self.end_date=df_raw[['end_date']][border1:border2].values
        self.text=df_raw[[text_name]][border1:border2].values
    def get_prior_y(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        prior_rows = []
        for s_begin, s_end in zip(s_begins, s_ends):
            absolute_origin = self.border1 + int(s_end)
            target_indices = absolute_origin + np.arange(self.pred_len)
            if target_indices[-1] >= len(self.full_data_y):
                raise IndexError(
                    f'Prior target index {target_indices[-1]} exceeds data length '
                    f'{len(self.full_data_y)}'
                )

            fallback = float(
                self.full_data_y[max(0, absolute_origin - self.seq_len):absolute_origin, 0].mean()
            )
            values = []
            for target_index in target_indices:
                key = self.prior_phase_keys[int(target_index)]
                historical_indices = self.prior_phase_indices[key]
                count = int(np.searchsorted(
                    historical_indices, absolute_origin, side='left'
                ))
                if count:
                    prefix = self.prior_phase_prefix_sums[key]
                    value = float(prefix[count] / count)
                else:
                    value = fallback
                values.append([value])

                # This assertion is intentionally kept in the runtime path: a
                # future refactor must not silently reintroduce post-origin
                # target information into the statistical prior.
                if count and historical_indices[count - 1] >= absolute_origin:
                    raise AssertionError('Historical prior accessed a future target')
            prior_rows.append(values)
        return np.asarray(prior_rows, dtype=self.full_data_y.dtype)
    def get_text(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        
        text=np.array([self.text[s_end] for s_end in s_ends])
        return text
    def get_date(self, indices):
        if isinstance(indices, torch.Tensor):
            indices = indices.numpy()

        s_begins = indices % self.tot_len
        s_ends = s_begins + self.seq_len
        r_begins = s_ends - self.label_len
        r_ends = r_begins + self.label_len + self.pred_len

        x_start_dates = np.array([self.start_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        x_end_dates = np.array([self.end_date[s_beg:s_end] for s_beg, s_end in zip(s_begins, s_ends)])
        
        

        return x_start_dates, x_end_dates
    def __getitem__(self, index):
        feat_id = index // self.tot_len
        s_begin = index % self.tot_len

        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end, feat_id:feat_id + 1]
        seq_y = self.data_y[r_begin:r_end, feat_id:feat_id + 1]
        
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark,index

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
