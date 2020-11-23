import sys
import os
import numpy as np
import json
import pandas as pd
from pandas.io.json import json_normalize
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image


##### utils ####
class GetData():
    def __init__(self):
        
        self.data_dict = self.read_json()
        self.df = self.get_df()


    def read_json(self):
        '''reads in the performance table saved in 
        models.downstream_performance.json 
        '''
        path = os.path.dirname(os.path.abspath(__file__))
        path = str(path.split('/scripts')[0])\
            +'/models/downstream_performance/downstream_performance.json'
        with open(path, 'r') as json_file:
            data_dict = json.load(json_file)
        return data_dict
    

    def reorganize_df(self):
        order = [ 'DECODER.acc',
             'TAPE.stability.MSE',
             'TAPE.stability.MAE',
             'TAPE.stability.S_Corr',
             'TAPE.fluorescence.MSE',
             'TAPE.fluorescence.MAE',
             'TAPE.fluorescence.S_Corr',
             'DMS.protein_g.MSE',
             'DMS.protein_g.P_corr',
             'DMS.1D5R.MSE',
             'DMS.1D5R.P_corr',
             'DMS.2H11.MSE',
             'DMS.2H11.P_corr',
             'MODEL_PARAMS.latent_size',
             'MODEL_PARAMS.layers',
             'MODEL_PARAMS.fully_con',
             'MODEL_PARAMS.compression_seq',
             'MODEL_PARAMS.channels',
             'MODEL_PARAMS.ks_conv',
             'MODEL_PARAMS.str_conv',
             'MODEL_PARAMS.pad_conv',
             'MODEL_PARAMS.ks_pool',
             'MODEL_PARAMS.str_pool',
             'MODEL_PARAMS.pad_pool',
             'PATHS.tape_dir',
             'PATHS.log_file',
             'PATHS.DMS_dir',
             'PLOTS.stability',
             'PLOTS.fluorescence',
             'PLOTS.protein_g',
             'PLOTS.1D5R',
             'PLOTS.2H11',
             'DECODER.pad_acc',
             'NOTES',
             'DATE_LAST_ADDED',
             'MODEL',
             'models'
            ]
        return order
    
    def rename_models(self,nn_names):
        renamed_nn = []
        for name in nn_names:
            name = str(name.split('.py')[0])
            name = str(name.split('cnn_')[-1])
            renamed_nn.append(name)
        
        return renamed_nn
        
        
    @st.cache
    def get_df(self):
        '''reads json and converts to pandas df.
        returns df of preformance table'''
        data_dict = self.read_json()
        model_data = []
        nn_names = []
        for key in data_dict.keys():
            model_data.append(data_dict[key])
            nn_names.append(key)

        df = json_normalize(model_data)
        df['models'] = nn_names
        df.index = df['models']
        df['MODEL'] = self.rename_models(nn_names)

        order = self.reorganize_df()
        df = df[order]
        return df
    
class Selections(GetData):
    '''return seelections in streamlit interactions'''
    def __init__(self):
        super().__init__()

#         self.entire_df = self.df
            

    def get_models(self, select_models):
        '''returns all or selected models of df '''
        if not select_models:
            df_models = self.df
        elif 'all' in select_models: 
            df_models = self.df
        else: 
            df_models = self.df.loc[select_models]  

        return df_models

    def get_tasks(self, select_tasks):
        ''' converts from multiselect options to df column names'''
        tasks=[]

        if  'stability' in select_tasks or 'fluorescence' in select_tasks:
            tape = [[f'TAPE.{i}.MAE',f'TAPE.{i}.MSE',f'TAPE.{i}.S_Corr'] \
                    for i in select_tasks if i in ['stability', 'fluorescence']]
            tape = [i for sublist in tape for i in sublist]
            tasks = tasks+tape
            print('tape',tasks)

        if  'DMS_2H11' in select_tasks or 'DMS_1D5R' in select_tasks:
            dms = [i[-4:] for i in select_tasks  if i in ['DMS_2H11', 'DMS_1D5R']]
            dms =  [[f'DMS.{i}.MSE',f'DMS.{i}.P_corr'] for i in dms ]
            dms = [i for sublist in dms for i in sublist] 
            tasks = tasks+dms
            print('dms',tasks)

        if  'DDG_protein_g' in select_tasks:
            i = 'protein_g'
            ddg = [[f'DMS.{i}.MSE',f'DMS.{i}.P_corr']]
            ddg = [i for sublist in ddg for i in sublist] 
            tasks = tasks+ddg

        if 'all TAPE' in  select_tasks:
            tape = ['TAPE.fluorescence.MAE','TAPE.fluorescence.MSE','TAPE.fluorescence.S_Corr',\
                   'TAPE.stability.MAE','TAPE.stability.MSE','TAPE.stability.S_Corr']
            tasks = tasks + tape

        if 'all DMS for single proteins' in select_tasks: 
            dms = ['2H11', '1D5R', 'protein_g']
            dms =  [[f'DMS.{i}.MSE',f'DMS.{i}.P_corr'] for i in dms ]
            dms = [i for sublist in dms for i in sublist] 
            tasks = tasks+dms 

        if not select_tasks or 'all' in select_tasks: 
            tasks = []
            tape = ['TAPE.fluorescence.MAE','TAPE.fluorescence.MSE','TAPE.fluorescence.S_Corr',\
                   'TAPE.stability.MAE','TAPE.stability.MSE','TAPE.stability.S_Corr']
            tasks = tasks + tape
            dms = ['2H11', '1D5R', 'protein_g']
            dms =  [[f'DMS.{i}.MSE',f'DMS.{i}.P_corr'] for i in dms ]
            dms = [i for sublist in dms for i in sublist] 
            tasks = tasks+dms 

        return tasks  

        
    def get_model_params(self,select_model):
        ''' return saved params for model'''
        
        model_params = self.data_dict[select_model]['MODEL_PARAMS']

        return  model_params

#     def add_unirep(self,df,task):
#         '''adds performance from tape's pretrained models'''


class Handle_DF():
    ''' deals with organising the used DF'''
    
    @classmethod   
    def add_unirep(cls, df_input):
        '''adds performance from tape's pretrained models'''
        # unirep
        df = pd.DataFrame(columns=['models','MODEL','TAPE.stability.S_Corr','TAPE.fluorescence.S_Corr'])
        models = ['UniRep','ResNet','Transformer', 'One-hot']
        df['models'] = models
        df['MODEL']=['TAPE BENCHMARKS']*4
        stability_S_Corr    = [0.73,0.73,0.73,0.19]
        fluorescence_S_Corr = [0.67,0.21,0.68,0.14]
        df['TAPE.stability.S_Corr'] = stability_S_Corr
        df['TAPE.fluorescence.S_Corr'] = fluorescence_S_Corr
        df.index = df['models']

        df_output = pd.concat([df, df_input])
        return df_output    

    

    @classmethod
    def order_models(cls):
        name_order = [
                'cnn_strided_ls500_2fc_seq63.py_ks3',
                'cnn_strided_ls500_2fc_seq63.py_ks5',

                 'cnn_strided_ls500_1fc_seq63.py_ks3', 
    #              'cnn_strided_ls500_1fc_seq63.py_ks7',
                 'cnn_strided_ls500_1fc_seq63.py_ks9',
                 'cnn_strided_ls500_1fc_seq63.py_ks11',



                 'cnn_strided_ls500_0fc_seq4.py_ks3',
                 'cnn_strided_ls500_0fc_seq4.py_ks5',  
    #              'cnn_strided_ls500_0fc_seq4.py_ks7',
                 'cnn_strided_ls500_0fc_seq4.py_ks9',
                 'cnn_strided_ls500_0fc_seq4.py_ks11',

                 'cnn_strided_ls504_0fc_seq63.py_ks3',
                 'cnn_strided_ls504_0fc_seq63.py_ks11',

                 'cnn_avgpool_ls500_1fc_seq63.py_ks2_kp2',
                 'cnn_avgpool_ls500_1fc_seq63.py_ks5_kp5',
    #              'cnn_avgpool_ls500_1fc_seq63.py_ks7_kp7',
                 'cnn_avgpool_ls500_1fc_seq63.py_ks9_kp3',

                  'cnn_autoencode_AA.py'
                ]

        axis_names = ['ks 3','ks 5',
                     'ks 3','ks 9','ks 11',
                     'ks 3','ks 5','ks 9','ks 11',
                     'ks 3','ks 11',
                     'ks 2 kp 2', 'ks 5 kp 5', 'ks 9 kp 3',

                     'ks 5']

        return  name_order, axis_names 


    
    
class PremadePlots():
    def __init__(self,selected_models,df):
        self.selected = selected_models
        self.df_models = df.loc[selected_models]
        self.premade_plots = \
            ['PLOTS.stability','PLOTS.fluorescence',\
             'PLOTS.protein_g','PLOTS.2H11','PLOTS.1D5R']

        self.plots, self.captions =  self.get_image_plots()
        
    def _plot_exists(self,plot):
            path = self.df_models[plot]
            if type(path)==str and os.path.isfile(path):
                return True
            else: return False

    def get_image_plots(self):
        plots=[]
        captions=[]
        for plot in self.premade_plots:
            if self._plot_exists(plot): 
                if plot in ['PLOTS.protein_g','PLOTS.2H11','PLOTS.1D5R']:
                    path =  self.df_models[plot][:-3]+'png'
                else:
                    path = self.df_models[plot]
                image = Image.open(path)
                plots.append(image)
                captions.append(plot.split('.')[-1])
                
        return plots,captions

    

    
    
    
#### === #####    

# get data dictionary as flattened pandas df
# initiate selections class for easier selections 
selections = Selections()
# test = GetData().df
df = selections.df

st.title("Representation Learning - Downstream Performance")
st.write("See left bar for visualisation options")

################################################################
#          SHOW TABLES 
################################################################
st.sidebar.markdown('# VISUALIZATIONS OPTIONS')

# add option to select/show nn model 
st.sidebar.markdown('## Result table')
if st.sidebar.checkbox('Show and investigate result table?'):
    st.header("Result table")

    option_models = ['all'] +  [i for i in list(df['models'])]
    select_models = st.multiselect(
            "Choose a pretrained model", option_models)
    df_models = selections.get_models(select_models)

    # write out table
    # st.write("### Performance  of pretrained nn models ", df_models )


    # add option to select task from table    
    option_tasks = ['all', 'all TAPE', 'all DMS single proteins', \
                    'stability', 'fluorescence',\
                    'DDG_protein_g', 'DMS_2H11', 'DMS_1D5R']

    select_tasks = st.multiselect(
            "Choose a downstream task", option_tasks)
    if not select_tasks:
        st.write("Performance  of pretrained nn models ", df_models)
    else:
        tasks = selections.get_tasks(select_tasks)
        df_tasks = df_models[tasks]
        st.write("Performance  of pretrained nn models ", df_tasks)

################################################################      
#             PLOTTING 
################################################################


#-------------------------------------------------------------
#             MODEL DECODER COMPARISONS 
#-------------------------------------------------------------
st.header("Upstream NN models' decoder performances ")
order_plot_by= st.radio(
            "order plot by:", ['model','fully connected layers','CNN layers','compressed seq','compressed channels'])
if  order_plot_by is 'model':                   
    model_order, axis_names = Handle_DF.order_models()
    fig = px.bar(df, x='models', y='DECODER.acc',
                 hover_data=[ 'TAPE.stability.S_Corr',
                             'TAPE.fluorescence.S_Corr', 
                             'DMS.protein_g.P_corr', 
                             'DMS.1D5R.P_corr', 
                             'DMS.2H11.P_corr'], 
                 color='MODEL',
                 labels={'DECODER.acc':'Accuracy of decoder','strided_ls500_0fc_seq4':'Stride 0 fully con seq 4'},
                 barmode="group")
    fig.update_layout(
                title='Decoder Accuracy',
                xaxis_tickangle=45,
             xaxis={'categoryorder':'array', 'categoryarray':model_order})

    fig.update_xaxes(tickmode = "array",
                     tickvals = np.arange(len(axis_names)+1), 
                     ticktext = axis_names )


elif order_plot_by is 'fully connected layers':
    sort = 'MODEL_PARAMS.fully_con'
    fig = px.bar(df, x='models', y='DECODER.acc', facet_col=sort,
             hover_data=[ 'TAPE.stability.S_Corr',
                         'TAPE.fluorescence.S_Corr', 
                         'DMS.protein_g.P_corr', 
                         'DMS.1D5R.P_corr', 
                         'DMS.2H11.P_corr'], 
             color='MODEL',
             labels={'DECODER.acc':'Accuracy of decoder','MODEL_PARAMS.fully_con':'FC layers:'},
             barmode="group")

elif order_plot_by is 'CNN layers':
    sort = 'MODEL_PARAMS.layers'
    fig = px.bar(df, x='models', y='DECODER.acc', facet_col=sort,
             hover_data=[ 'TAPE.stability.S_Corr',
                         'TAPE.fluorescence.S_Corr', 
                         'DMS.protein_g.P_corr', 
                         'DMS.1D5R.P_corr', 
                         'DMS.2H11.P_corr'], 
             color='MODEL',
             labels={'DECODER.acc':'Accuracy of decoder', 'MODEL_PARAMS.layers':'CNN layers:'},
             barmode="group")
        
elif order_plot_by is 'compressed seq':
    sort = 'MODEL_PARAMS.compression_seq'
    fig = px.bar(df, x='models', y='DECODER.acc', facet_col=sort,
             hover_data=[ 'TAPE.stability.S_Corr',
                         'TAPE.fluorescence.S_Corr', 
                         'DMS.protein_g.P_corr', 
                         'DMS.1D5R.P_corr', 
                         'DMS.2H11.P_corr'], 
             color='MODEL',
             labels={'DECODER.acc':'Accuracy of decoder', 'MODEL_PARAMS.compression_seq':'seq:'},
             barmode="group")

elif order_plot_by is 'compressed channels':
    sort = 'MODEL_PARAMS.channels'
    fig = px.bar(df, x='models', y='DECODER.acc', facet_col=sort,
             hover_data=[ 'TAPE.stability.S_Corr',
                         'TAPE.fluorescence.S_Corr', 
                         'DMS.protein_g.P_corr', 
                         'DMS.1D5R.P_corr', 
                         'DMS.2H11.P_corr'], 
             color='MODEL',
             labels={'DECODER.acc':'Accuracy of decoder', 'MODEL_PARAMS.channels':'chnls:'},
             barmode="group")
    
    
st.plotly_chart(fig)




#-------------------------------------------------------------
#             MODEL COMPARISONS 
#-------------------------------------------------------------

st.sidebar.markdown('## Compare models by downstream performance')
if st.sidebar.checkbox('Compare downstream performances?',value=True):
    st.header("Downstream model performances ")

    # plotting options on side 

    st.markdown('Options:')
    # add option to select models to plot   
    option_plot_models =['all'] +  [i for i in list(df['models'])]
    select_plot_models = st.multiselect(
            "Choose models to plot", option_plot_models,['all'])
    df_plot_models = selections.get_models(select_plot_models)


    # add option to select task to plot    
    option_plot_tasks = ['all', 'all TAPE', 'all DMS for single proteins', \
                        'stability', 'fluorescence', \
                        'DDG_protein_g', 'DMS_2H11', 'DMS_1D5R']

    select_plot_tasks = st.multiselect(
            "Choose downstream task(s)", option_plot_tasks,['all'])
    select_plot_tasks = selections.get_tasks(select_plot_tasks)


    # select plotting metric
    select_metric = st.radio(
            "Choose metric for plotting", ['Correlation','MSE','MAE'])
    if select_metric is 'Correlation':
        select_metric = 'orr' # hacked
    select_plot_tasks = [task for task in select_plot_tasks if select_metric in task]
    df_plot = df_plot_models[select_plot_tasks + ['models','MODEL']]  # needed for plotting 
    
    # add option to sort by: 
    sort_by_performance = st.checkbox('sort by decoder accuracy performance?')

    # checkbox to show table over selections in plotting
    if st.checkbox('Show table of selected plotting inputs'):
        st.write("### Selected models for comparing }", df_plot)


    for task in select_plot_tasks:
        title = task.split('.')[1]
        model_order, axis_names = Handle_DF.order_models()
        
        # add tapes benchsmarks to df 
        
        if task in ['TAPE.fluorescence.S_Corr','TAPE.stability.S_Corr']:
            df_tape_corr = df_plot[['TAPE.fluorescence.S_Corr','TAPE.stability.S_Corr','models','MODEL']]
            # add unirep, resnet, transformer and one-hot performance to df
            df_tape_corr  = Handle_DF.add_unirep(df_tape_corr )
            model_order += ['UniRep','ResNet','Transformer', 'One-hot']
            axis_names += ['UniRep','ResNet','Transformer', 'One-hot']
            fig = px.bar(df_tape_corr , x="models", y=task, color="MODEL",barmode="group")
        
        else: 
            fig = px.bar(df_plot, x="models", y=task, color="MODEL",barmode="group")
        # hacked
        if select_metric is 'orr' and title in ['protein_g','2H11','1D5R']:
            select_metric = 'Pearson'
        elif select_metric is 'orr' and title not in ['protein_g','2H11','1D5R']:
            select_metric = 'Spearman'
            
            
        fig.update_layout(
            title=title,
            yaxis_title=select_metric,
            xaxis_tickangle=45,
            xaxis={'categoryorder':'array', 'categoryarray':model_order},
            font=dict(
                #family="Courier New, monospace",
                    size=14,
                    color="#7f7f7f"))
        
        fig.update_xaxes(tickmode = "array",
                 tickvals = np.arange(len(axis_names)+1), 
                 ticktext = axis_names )

        st.plotly_chart(fig)

     
    #--------------------------------------------------------------
    ### TEST TO SORT BY DECODER PERFORMANCE
    task
    model_order = [model for _, model in sorted(zip(df['DECODER.acc'], df['MODEL']))]

    fig = px.bar(df_plot, x="models", y=task, color="MODEL",barmode="group")

    fig.update_layout(
    title=title,
    yaxis_title=select_metric,
    xaxis_tickangle=45,
    xaxis={'categoryorder':'array', 'categoryarray':model_order},
    font=dict(
        #family="Courier New, monospace",
            size=14,
            color="#7f7f7f"))
    st.plotly_chart(fig)

#------------------------------------------------------------
#       Plotting individual performance from path 
#-------------------------------------------------------------
# write title and info
st.sidebar.markdown('## Individual model performance')
st.sidebar.markdown('Show a single models downstream performances in details \
                    or look at upstream details such as confusion matrix, \
                    training parameters or training infor  ')

# sidebar option to plot individual model info
if st.sidebar.checkbox('Investigate individual model performance?'):
    st.header("Individual model performance")

    # options to chose model    
    st.subheader('Options ')
    single_model = st.selectbox("Choose individual model to investigate", list(df['models']))
    
    # output model info about chosen model
    st.subheader(f"Showing {single_model}") 
    st.write('Upstream model paramters:')
    # get model params for output
    model_params = selections.get_model_params(single_model)
    output_params = ''
    for i in ['ks_conv','str_conv','ks_pool','str_pool',\
              'latent_size','fully_con','compression_seq',\
              'layers','channels']: 
        if model_params[i] is not None:
            output_params += f"{i}: {model_params[i]}, "  
    st.write(output_params) 

    
    # options for displaying 
    sel_indv_display = st.radio(
            "What to show? ", ['Downstream experimental vs prediction plots on all tasks',
                               'Downstream experimental vs prediction histograms for TAPE tasks',
                               'Downstream training plots for TAPE tasks',
                               'Downstream log file showing parameters used',
                               'Upstream training plots',
                               'Upstream log file showing all parameters and model layers info',
                               'Upstream pytorch achitecture from python file',
                               'Upstream confusion matrix',
                               "Upstream performance on protein's C-, N-tails",
                               "Upstream performance on predictions of each AA",
                               ])
    
    
    # plot all predictions vs targets 
    if sel_indv_display is "Downstream experimental vs prediction plots on all tasks":
        st.write('sorry the bad resolution, bug in streamlit for showing large images')
        premade_plots = PremadePlots(single_model, df)
        if premade_plots.plots:
            st.image(premade_plots.plots, caption=premade_plots.captions, width=700,format='PNG')

        else:
            st.text(f"No premade/saved plots exists for model {single_model}")
    
    
    ### display histograms for tape
    if sel_indv_display is 'Downstream experimental vs prediction histograms for TAPE tasks':
        images = []
        captions = []
        df=df.loc[single_model]
        fluor_path =  df['PATHS.tape_dir'] + 'plot_histogr_fluorescence.png'
        stabil_path =  df['PATHS.tape_dir'] + '../stability/plot_histogr_stability.png'
        if os.path.exists(fluor_path):
            image = Image.open(fluor_path)
            images.append(image)
            captions.append('fluorescence')
        if os.path.exists(stabil_path):
            image = Image.open(stabil_path)
            images.append(image)
            captions.append('stability')  
        if images: 
            st.image(images, caption=captions, width=900,format='PNG')
            
        else:
            st.write('No such plot saved for this model')
            
    ### display downstream training for tape
    if sel_indv_display is 'Downstream training plots for TAPE tasks':
        images = []
        captions = []
        df=df.loc[single_model]
        fluor_path =  df['PATHS.tape_dir'] + 'plot_downstream_training_fluorescence.png'
        stabil_path =  df['PATHS.tape_dir'] + '../stability/plot_downstream_training_stability.png'
        if os.path.exists(fluor_path):
            image = Image.open(fluor_path)
            images.append(image)
            captions.append('fluorescence')
        if os.path.exists(stabil_path):
            image = Image.open(stabil_path)
            images.append(image)
            captions.append('stability')  
        if images: 
            st.image(images, caption=captions, width=900,format='PNG')
            
        else:
            st.write('No such plot saved for this model')

     ### display downstream logfile
    if sel_indv_display is 'Downstream log file showing parameters used':
        df=df.loc[single_model]
        path =  df['PATHS.log_file'] 
        if os.path.exists(path):
            with open(path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    st.text(line)
          
        
     ### display upstream training of model 
    if sel_indv_display is 'Upstream training plots':
        st.write('sorry the bad resolution, bug in streamlit for showing large images')

        df=df.loc[single_model]
        st.write('### OBS if validation looks weird (starts high), its simply a bug in  plotting upon a restart of the training')
        path =  df['PATHS.tape_dir'] + '../../plot_training_performance.png'
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption='Upstream training', width=900,format='PNG')
            
        else:
            st.write('No such plot saved for this model')

    ### display upstream logfile 
    if sel_indv_display is 'Upstream log file showing all parameters and model layers info':
        df=df.loc[single_model]
        path =  df['PATHS.tape_dir'] +'../../logfile_params.log'
        path
        if os.path.exists(path):
            with open(path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    st.text(line)
                    if "Estimated Total Size" in line:
                        break
                
            
        else:
            st.write('No such plot saved for this model')        



    ### display upstream architecture
    if sel_indv_display is 'Upstream pytorch achitecture from python file':
        st.write('Showing puthon/pytorch nn architecture nn model')
        df=df.loc[single_model]
        path = os.path.realpath(__file__).split('representation_learning')[0]
        path = path + 'representation_learning/scripts/nn_models/cnn_'+ df['MODEL']+'.py'
        path
        if os.path.exists(path):
            with open(path,"r") as f:
                lines = f.readlines()
                for line in lines:
                    st.text(line)                
            
        else:
            st.write('Cannot find file, look in scripts/nn_models/')        

                  
            
                
     ### display upstream confusion matrix 
    if sel_indv_display is  'Upstream confusion matrix':
        st.write('sorry the bad resolution, bug in streamlit for showing large images')

        df=df.loc[single_model]
        path =  df['PATHS.tape_dir'] + '../../plot_confusion_matrix_test.png'
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption='Upstream training', width=800,format='PNG')
            
        else:
            st.write('No such plot saved for this model')

           
                
     ### display upstream  confusion matrix
    if sel_indv_display is  'Upstream confusion matrix':
        st.write('sorry the bad resolution, bug in streamlit for showing large images')

        df=df.loc[single_model]
        path =  df['PATHS.tape_dir'] + '../../plot_confusion_matrix_test.png'
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption='Upstream training', width=700,format='PNG')
            
        else:
            st.write('No such plot saved for this model')        

     ### display upstream C-,N-terminal performance 
    if sel_indv_display is "Upstream performance on protein's C-, N-tails":
        st.write('sorry the bad resolution, bug in streamlit for showing large images')
        df=df.loc[single_model]
        images = []
        captions = []
        N_path =  df['PATHS.tape_dir'] + '../../plot_pos_acc_N-term.png'
        C_path =  df['PATHS.tape_dir'] + '../../plot_pos_acc_C-term.png'
        if os.path.exists(N_path):
            image = Image.open(N_path)
            images.append(image)
            captions.append('N-terminal')
        if os.path.exists(C_path):
            image = Image.open(C_path)
            images.append(image)
            captions.append('C-terminal')  
        if images: 
            st.image(images, caption=captions, width=1200,format='PNG')
        
        else: 
            st.write('Sorry this model was trained before these plots was added to training scrip. No such plots')
    
     ### display upstream performance on prediction each aa
    if sel_indv_display is "Upstream performance on predictions of each AA":
        st.write('sorry the bad resolution, bug in streamlit for showing large images')

        df=df.loc[single_model]
        path =  df['PATHS.tape_dir'] + '../../plot_acc_per_aa_test.png'
        if os.path.exists(path):
            image = Image.open(path)
            st.image(image, caption='Upstream training', width=700,format='PNG')
            
        else:
            st.write('No such plot saved for this model')        

              