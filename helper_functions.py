#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, make_scorer, auc, plot_roc_curve
from sklearn import svm
import seaborn as sns
#import umap

from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


def save_obj(obj, name ):
    with open('obj_'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj_' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def load_data(path,filenames_input):
    df_date_assessment=pd.read_csv(r"tidy_data_tiles/date_assessment.csv",index_col='Unnamed: 0')

    tic = time.perf_counter()
    df_input=pd.DataFrame()
    for filename in filenames_input:
        df_temp=pd.read_csv(path+filename,index_col='Unnamed: 0')
        df_input=pd.concat([df_input,df_temp],axis=1)
    df_label=pd.read_csv(path+'OutcomesRes.csv',index_col='Unnamed: 0')
    toc = time.perf_counter()
    print('Time to read the data: '+str(toc-tic))


    df_label[["date_assessment"]]=df_date_assessment

    print("All consented individuals: ", df_input.shape[0])
    return (df_input,df_label,df_date_assessment)

def create_summary_table(df_tempt,column_lables):
    
    sum_dict1=dict()
    for col_label in column_lables:
        sum_dict1={**sum_dict1,**{col_label:"{var1:0.2f} ({var2:0.2f})".format(var1=df_tempt[col_label].mean(),
                                                              var2=df_tempt[col_label].std())}}
    sum_table=pd.DataFrame(sum_dict1,index=[0])   
    sum_table=sum_table.T.sort_index(axis=0)
    return sum_table

def create_summary_table_custom(df_input, df_label):
    indices_tempt=df_label[df_label["outcomes"]==1].index
    age_at_diagnoses=df_input.loc[indices_tempt,"Age at recruitment"]+df_label.loc[indices_tempt,"outcome_months"]/12
    sum_dict={
        "n":df_input.shape[0],
        "female":"{var1:0.2f} (%{var2:0.2f})".format(var1=df_input[df_input["gender"]==0].shape[0],
                                                    var2=df_input[df_input["gender"]==0].shape[0]/df_input.shape[0]*100),
        "Age at recruitment":"{var1:0.2f} ({var2:0.2f})".format(var1=df_input["Age at recruitment"].mean(),var2=df_input["Age at recruitment"].std()),
        "Age at first diagnosis":"{var1:0.2f} ({var2:0.2f})".format(var1=age_at_diagnoses.mean(),var2=age_at_diagnoses.std()),
        "Townsend deprivation index at recruitment":"{var1:0.2f} ({var2:0.2f})".format(var1=df_input['Townsend deprivation index at recruitment'].mean(),
                                                                                           var2=df_input['Townsend deprivation index at recruitment'].std()),
        "Follow-up duration (years)":"{var1:0.2f} ({var2:0.2f})".format(var1=(df_label["outcome_months"]/12).mean(),var2=(df_label["outcome_months"]/12).std()),
        "number of events":"{var1:0.0f} ({var2:0.2f}%)".format(var1=age_at_diagnoses.shape[0],var2=age_at_diagnoses.shape[0]/df_input.shape[0]*100),
        "Incidence rate, per 1000 person-years":"{var1:0.0f} ".format(var1=age_at_diagnoses.shape[0]/(df_label.outcome_months.sum()/12)*1000)   
    }
    sum_table=pd.DataFrame(data=sum_dict,index=[0])
    sum_table.T

    #Ethnicity
    ethnicity=df_input[[col for col in df_input.columns.values if "Ethnicity" in col]].sum()/df_input.shape[0]*100
    ethnicity=ethnicity.sort_values(ascending=False)
    threshold=0.5
    ethnicity_sum=pd.concat([pd.DataFrame(ethnicity[ethnicity>threshold]),
                             pd.DataFrame({0:ethnicity[ethnicity<=threshold].sum()},index=["Ethnicity_other"])]
                            ,axis=0)
    #print(pd.DataFrame(ethnicity[ethnicity>threshold]))
    sum_table=pd.concat([sum_table.T,ethnicity_sum],axis=0)
    return sum_table


# In[1]:


class UKBB_DATA():
    def __init__(self,df_input,df_label,df_date_assessment):
        self.df_input=df_input.copy()
        self.df_label=df_label.copy()
        self.df_date_assessment=df_date_assessment.copy()
        
        self.selected_categories_exclusions=load_obj('selected_categories_exclusions')
        self.selected_categories_tidy_columns=load_obj('selected_categories_tidy_columns')
        self.selected_categories_tidy=load_obj('selected_categories_tidy')

        self.selected_outcome_categories_exclusions=load_obj('selected_outcome_categories_exclusions')
        self.selected_outcome_categories_tidy_columns=load_obj('selected_outcome_categories_tidy_columns')
        self.selected_outcome_categories_tidy=load_obj('selected_outcome_categories_tidy')
        
    
    def drop_lost_to_follow_up(self):
        lost_follow_up=pd.read_csv(r"tidy_data_tiles/lost_follow_up.csv",index_col='Unnamed: 0')
        lost_follow_up_indxs=lost_follow_up[lost_follow_up.notna()["190-0.0"]].index

        self.df_input.drop(index=lost_follow_up_indxs,inplace=True)
        #df_label.drop(index=lost_follow_up_indxs,inplace=True)
        print("{} individuals lost to follow up and were excluded".format(len(lost_follow_up_indxs)))
        print("the remaining number of individuals is {}" .format(self.df_input.shape))
    
    def apply_exclusions(self):
        
        exclusion_indx=self.df_input.index[pd.concat([self.df_input[self.selected_categories_exclusions],
        self.df_label[self.selected_outcome_categories_exclusions]],axis=1).any(axis=1)]

        self.df_input.drop(index=exclusion_indx,inplace=True)
        self.df_label.drop(index=exclusion_indx,inplace=True)

        self.df_input.drop(columns=self.selected_categories_exclusions,inplace=True)
        self.df_label.drop(columns=self.selected_outcome_categories_exclusions,inplace=True)
        
        print("Applying exclusion based on {} and {}".format(self.selected_categories_exclusions,
                                                             self.selected_outcome_categories_exclusions))
        print("The remaining number of observatinos is {}".format(self.df_label.shape))
        
    def prepare_mortality_outcomes(self):
        indx_all_death=self.df_label[~self.df_label["death"].isna()].index
        indx_all_alive=self.df_label[self.df_label["death"].isna()].index
        delta=pd.to_datetime(self.df_label.loc[indx_all_death]["death"],format='%Y-%m-%d').sub(pd.to_datetime(self.df_label.loc[indx_all_death]["date_assessment"],format='%Y-%m-%d'))
        delta2=pd.to_datetime("2017/5/1",format='%Y-%m-%d')-pd.to_datetime(self.df_label.loc[indx_all_alive]["date_assessment"],format='%Y-%m-%d')

        self.df_label.loc[indx_all_death,'outcome_months']=delta/np.timedelta64(1, 'M')
        self.df_label.loc[indx_all_alive,'outcome_months']=delta2/np.timedelta64(1, 'M')

        self.df_label.loc[indx_all_death,"outcomes"]=1
        self.df_label.loc[indx_all_alive,"outcomes"]=0 
    
    def normalise_numericals(self):
        non_binary_cols=list(self.df_input.columns[self.df_input.nunique()>2].values)
        non_binary_cols.remove('birth_date')

        scaler = StandardScaler()
        df_input_stnd_temp=scaler.fit_transform(self.df_input[non_binary_cols])
        df_input_stnd_temp = np.clip(df_input_stnd_temp, -5, 5)
        df_input_stnd=self.df_input.copy()
        df_input_stnd[non_binary_cols]=df_input_stnd_temp

        return df_input_stnd

