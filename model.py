import pandas as pd
from xgboost import XGBClassifier
import pickle

df=pd.read_csv(r'C:\Users\DELL\Desktop\dev\models\insurance-fraud-claim-detection\insurance_fraud_claims.csv')

df.drop('_c39', axis=1, inplace=True)

df['policy_bind_year']=df['policy_bind_date'].str.extract('(\d{4})\-').astype('int32')
df['incident_month']=df['incident_date'].str.extract('\d{4}\-(\d{2})').astype('int32')
df['collision_type'] = df['collision_type'].replace("?", "undocumented")
df['police_report_available'] = df['police_report_available'].replace("?", "undocumented")
df['property_damage'] = df['property_damage'].replace("?", "undocumented")
df['auto_make'] = df['auto_make'].replace("Suburu", "Subaru")
df['incident_severity'] = df['incident_severity'].map({"Trivial Damage":0,
                                                       "Minor Damage":1,
                                                       "Major Damage":2,
                                                       "Total Loss":3
                                                      }).astype("int32")
df['umbrella_limit'].iloc[290] = 1000000
all_var=df.columns
cont_var =['age','incident_hour_of_the_day',
           'number_of_vehicles_involved','total_claim_amount',
           'injury_claim','property_claim','vehicle_claim',
           'months_as_customer','policy_annual_premium','policy_deductable',
           'umbrella_limit','capital-gains','capital-loss', 
           'auto_year','witnesses','bodily_injuries','policy_bind_year','incident_severity']
ord_var = ['policy_deductable','witnesses','bodily_injuries','incident_severity']
quan_var = (list(set(cont_var) - set(ord_var))) 
nom_var = (list(set(all_var) - set(cont_var)))
large_cat=['incident_date','auto_model','insured_zip','policy_bind_date','incident_location','policy_number']
df.drop(large_cat, axis=1, inplace=True)
nom_var.remove('fraud_reported')
nom_var = (list(set(nom_var) - set(large_cat))) 
df['loss_by_claims'] = df['total_claim_amount'] - (df['policy_annual_premium'] * (2015 - df['policy_bind_year']))
df['fraud_reported'] = df['fraud_reported'].map({"Y":1, "N":0})
df['pclaim_severity_int'] = df['property_claim']*df['incident_severity']
df['vclaim_severity_int'] = df['vehicle_claim']*df['incident_severity']
df['iclaim_severity_int'] = df['injury_claim']*df['incident_severity']
df['tclaim_severity_int'] = df['total_claim_amount']*df['incident_severity']

rem = ['collision_type',  'insured_sex',  'insured_occupation',  'incident_state',  'auto_make',  'incident_city',  'insured_education_level',  'policy_csl',  'insured_relationship',  'incident_month',  'policy_state']
dum_list = [e for e in nom_var if e not in rem]

dum = pd.get_dummies(df[dum_list], drop_first=True)

dum.reset_index(drop=True, inplace=True)
df.drop(['months_as_customer', 'age', 'policy_deductable', 'umbrella_limit', 'capital-gains', 'capital-loss', 'incident_hour_of_the_day', 'witnesses', 'vehicle_claim', 'auto_year'], axis=1, inplace=True)
df.reset_index(drop=True, inplace=True)
df_dummied = pd.concat([df, dum], axis=1)

df_dummied.drop(nom_var, axis=1, inplace=True)

x = df_dummied.drop('fraud_reported', axis=1)
y = df_dummied['fraud_reported']

xg = XGBClassifier(booster='gbtree', n_jobs= -1, scale_pos_weight = 3.054054054054054, reg_lambda=0.1, reg_alpha= 0.05, n_estimators= 550, max_depth= 6, gamma= 3, eta= 0.05)
xg.fit(x,y)

pickle.dump(xg,open(r'C:\Users\DELL\Desktop\dev\models\insurance-fraud-claim-detection\insurance_xgboost_pickle_model.pkl','wb'))
model=pickle.load(open(r'C:\Users\DELL\Desktop\dev\models\insurance-fraud-claim-detection\insurance_xgboost_pickle_model.pkl','rb'))