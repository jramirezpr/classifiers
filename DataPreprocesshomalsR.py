# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 09:53:45 2018

@author: Guest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import math
from sklearn.preprocessing import StandardScaler,scale 
from sklearn import model_selection
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn import covariance
import statsmodels.api as sm


#read data
dfconsCRM = pd.read_excel("zillowMergewConsumerData.xlsx",
                          "zillowConsMergeTotalData")
dfZiptoIncome = pd.read_excel("ACS_2016_with_median_income_excel.xlsx",
                              "zipcode_income")

dfHousing=pd.read_csv("ACSHousingReduced.csv")
dfZiptoIncome=dfZiptoIncome.merge(dfHousing,on="OBJECTID")
dfZiptoIncome['percHeatElec']=np.divide(dfZiptoIncome['B25040e4'],
             np.maximum(dfZiptoIncome['B25040e1'],1))
utilities=['APU',
# 'BEU',
 #'CEU',
# 'Imperial Irrigation District',
 'LADWP',
# 'LEU',
 'PGE',
 'RPU',
 'SCE',
 'SMUD',
# 'modesto irrigation district',
 'sdge']
#filter out outlier utility companies
dfconsCRM=dfconsCRM[dfconsCRM['Utility Company'].isin(utilities)]

av_price={}
av_Usage={}
stdbyComp={}
for utili in utilities:
    df_i=dfconsCRM[dfconsCRM['Utility Company']==utili]
    av_price[utili]=df_i['Annual Utility Payment'].mean()
    av_Usage[utili]=df_i['Annual Usage'].mean()
    stdbyComp[utili]=df_i['Annual Usage'].std()
dfconsCRM['meanPrice']=dfconsCRM['Utility Company'].map(av_price) 
dfconsCRM['logMeanPrice']=np.log(dfconsCRM['meanPrice'])   
dfconsCRM['centeredmeanPrice']=dfconsCRM['meanPrice']-1100
dfconsCRM['meanUsage']=dfconsCRM['Utility Company'].map(av_Usage)  
dfconsCRM['stdU']=dfconsCRM['Utility Company'].map(av_Usage)
dfconsCRM['isLADWP']=(dfconsCRM['Utility Company']=="LADWP")

dfconsCRM['logC']=np.log(dfconsCRM['Annual Usage'])

dfconsCRM['Tier1']=dfconsCRM['Tier 1 Rate']
dfconsCRM['logTier1']=np.log(dfconsCRM['Tier 1 Rate'])
dfconsCRM['Tier1Sq']=np.square(dfconsCRM['Tier 1 Rate'])

IncomeDict = dict(zip(dfZiptoIncome.ZCTA5CE10, dfZiptoIncome.Median_midpoint))
dfconsCRM['zipIncome'] = dfconsCRM['ZIP CODE'].map(IncomeDict)
#electricalHeatingDict,same
heatDict= dict(zip(dfZiptoIncome.ZCTA5CE10, dfZiptoIncome.percHeatElec))
dfconsCRM['zipheatElec'] = dfconsCRM['ZIP CODE'].map(heatDict)


dfconsCRM['logIncome']=np.log(dfconsCRM['zipIncome'])
dfconsCRM['incomeCutoff']=np.minimum(dfconsCRM['zipIncome'],112500)
print("size")
print(dfconsCRM.shape)
dfconsCRM = dfconsCRM[np.isfinite(dfconsCRM['finishedSqFt'])]
#outlier based on influence graph
#dfconsCRM.drop(300)
#outlier based on cooks distance/ddfits


#prepare data for PCA
namesPCA=['logMeanPrice','Year','Year_Built','YearH','May','SQFT','Oct',
          'Baseline','Amount','zipIncome','Tier1',
          'Bed','Bath',
          'logIncome']

dummies = pd.get_dummies(dfconsCRM['Utility Company'])
dfcopy= dfconsCRM[namesPCA].copy()
dfcopy=pd.concat([dfcopy,dummies],axis=1)
#get rid of outliers with Isolation Forest
outlierDet=covariance.EllipticEnvelope(contamination=.05,random_state=4059)
trained_outlier=outlierDet.fit(dfcopy)
y_vals=trained_outlier.predict(dfcopy)
isoutlier=pd.DataFrame(y_vals==-1,index=dfcopy.index)
dfcopy=dfcopy[y_vals==1]
dfconsCRM=dfconsCRM[y_vals==1]
replace_map = {'Utility Company': {'APU': 1, 'LADWP': 2, 'PGE': 3, 'RPU': 4,
                                  'SCE': 5, 'SMUD': 6, 'sdge': 7 }}
dfconsCRM=dfconsCRM.replace(replace_map)
dfconsCRM[['logMeanPrice','Year','Year_Built','YearH','May','SQFT','Oct',
          'Baseline','Amount','zipIncome','Tier1','logIncome',
           'Bed','Bath','Utility Company']].to_csv('consPreprocessed.csv')