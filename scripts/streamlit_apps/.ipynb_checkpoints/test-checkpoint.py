import sys
import os
import json
from pandas.io.json import json_normalize
import pandas as pd
sys.path.insert(0, '/home/trz846/representation_learning/scripts/')
#import utils


##### utils ####
class GetData():
    def __init__(self):
        self.df = self.get_df()

    def _read_json(self):
        '''reads in the performance table saved in 
        models.downstream_performance.json 
        '''
        path = os.path.dirname(os.path.abspath(__file__))
        path = str(path.split('/scripts')[0])\
            +'/models/downstream_performance/downstream_performance.json'
        with open(path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict


    def get_df(self):
        '''reads json and converts to pandas df.
        returns df of preformance table'''
        data_dict = self._read_json()
        model_data = []
        nn_names = []
        for key in data_dict.keys():
            model_data.append(data_dict[key])
            nn_names.append(key)

        df = json_normalize(model_data)
        df.insert(0, 'model', nn_names, True)
        return df


# read in table in streamlit 
#@st.cache
df = GetData().df
print(df)
# select a specific model 
# nn_model = st.multiselect(
#     "Choose nn pretrained model", list(df.index)
# )



