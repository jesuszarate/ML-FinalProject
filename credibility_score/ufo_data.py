import pandas as pd
import datetime, time
import numpy as np

class UFOData:
    def __init__(self, cols=None, country = None):
        
        cols_names =['datetime', 'city', 'state', 'country',
                   'shape', 'duration(seconds)', 'duration(hours/min)', 'comments',
                   'date posted', 'longitude', 'latitude']

        _data = pd.read_csv('../ufo-scrubbed-geocoded-time-standardized.csv',
                           names=cols_names,
                           dtype=object,
                           error_bad_lines=False, warn_bad_lines=False)

        if cols is not None and 'comments' in cols:
            # Get rid of unicode characters
            _data['comments'] = _data["comments"].apply(lambda x: ''.join([" " if ord(i) < 32 or ord(i) > 126 else i for i in str(x)]))

        if country is not None:
            _data = _data[_data['country'] == country]

        if cols is None:
            self.data = _data[cols_names]
        else:
            self.data = _data[cols]

    def encoded(self):

        _encoded_data = self.data.copy()
        # Make date into ints
        if 'datetime' in _encoded_data:
            _encoded_data['timestamp'] = _encoded_data.datetime.apply(lambda x: self.dateToInt(x))
            _encoded_data['timestamp'] = _encoded_data['timestamp'].apply(pd.to_numeric)

        # Make date into ints
        if 'date posted' in _encoded_data:
            _encoded_data['posted_ts'] = _encoded_data['date posted'].apply(lambda x: self.datePostedToInt(x))
            _encoded_data['posted_ts'] = _encoded_data['posted_ts'].apply(pd.to_numeric)

        # Filter rows that have strings in the duration column
        if 'duration(seconds)' in _encoded_data:
            _encoded_data = _encoded_data[_encoded_data['duration(seconds)'].apply(lambda x : str.isdigit(x))]
            _encoded_data['duration(seconds)'] = _encoded_data['duration(seconds)'].apply(pd.to_numeric)
        
        if 'shape' in _encoded_data:
            labelMap = self.getLabelMap(_encoded_data['shape'])
            # _encoded_data = _encoded_data.iloc[_encoded_data['shape'].isnull().values, 'shape'].copy()
            # _encoded_data['shape'] = 'unknown'
            _encoded_data['shape'].fillna('unknown', inplace=True)
            _encoded_data['shape'] = _encoded_data['shape'].apply(lambda x: labelMap[x])

        if 'state' in _encoded_data:
            # _encoded_data = _encoded_data.iloc[_encoded_data['state'].isnull().values].copy()
            # _encoded_data['state'] = 'unknown'
            _encoded_data['state'].fillna('unknown', inplace=True)
            stateMap = self.getLabelMap(_encoded_data['state'])
            _encoded_data['state'] = _encoded_data['state'].apply(lambda x: stateMap[x])

        if 'latitude' in _encoded_data:
            _encoded_data['latitude'] = _encoded_data['latitude'].apply(pd.to_numeric)

        if 'longitude' in _encoded_data:
            _encoded_data['longitude'] = _encoded_data['longitude'].apply(pd.to_numeric)

        # Add keys to be able to merge
        # self.data['key'] = [i for i in range(self.data.shape[0])]
        # cols.append('key')
        return _encoded_data


    def datePostedToInt(self, x):
        _date = x.split('/')
        t = datetime.datetime(int(_date[2]), int(_date[0]), int(_date[1])).timestamp()
        return t


    def dateToInt(self, x):
        _date = x.split('/')

        year, time = _date[2].split()
        hour, minute = time.split(':')

        if int(hour) > 23:
            hour = 23
        if int(hour) < 0:
            hour = 0

        t = datetime.datetime(int(year), int(_date[0]), int(_date[1]), int(hour), int(minute), 0, 0).timestamp()
        return t

    def getLabelMap(self, column):
        # Map shape to an integer
        # labels = self.data['shape']
        column = column.unique()
        nums = [i for i in range(len(column))]
        return dict(zip(column, nums))

if __name__ == '__main__':
    names=['datetime', 'state','shape', 'duration(seconds)', 'comments','longitude', 'latitude']
    d = UFOData(names)

    d = d.encoded()
    pass