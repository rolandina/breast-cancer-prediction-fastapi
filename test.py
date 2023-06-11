from fastapi import FastAPI, Path, HTTPException
import psycopg2
from decouple import config
import json
import pandas as pd
import pickle

#preprocessing and model selection
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, FunctionTransformer
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, RepeatedKFold, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

#classification models metrics
from sklearn.metrics import classification_report, plot_precision_recall_curve

#classification models
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb


import uvicorn
import os

#from titanic.model import Model
from enum import Enum

class Race(Enum):
    white = "White"
    black = "Black"
    other = "Other"

class MaritalStatus(Enum):
    married = "Married"
    divorced = 'Divorced'
    single = "Single "
    widowed = "Widowed"
    separated = "Separated"


class TStage(str, Enum):
    t1 = "T1"
    t2 = "T2"
    t3 = "T3"
    t4 = "T4"

class NStage(str, Enum):
    n1 = "N1"
    n2 = "N2"
    n3 = "N3"

#may be produced from n and t stages
class SixthStage(str, Enum):
    IIA = "IIA"
    IIIA = "IIIA"
    IIIC = "IIIC"
    IIB = "IIB"
    IIIB = "IIIB"

class Grade(str, Enum):
    I = "1"
    II = "2"
    III = "3"
    IV = " anaplastic; Grade IV"

class AStage(str, Enum):
    regional = "Regional"
    distant = "Distant"

class EstrogenStatus(str, Enum):
    positive = "Positive"
    negative = "Negative"

class ProgesteroneStatus(str, Enum):
    positive = "Positive"
    negative = "Negative"

# load environment variables
DB_URL = config("DB_URL")

### open connection with db
def open_db_connection():
    try:
        connection = psycopg2.connect(DB_URL, sslmode='require')
        cursor = connection.cursor()
        return connection, cursor

    except (Exception, psycopg2.Error) as error:
        print("Error while fetching data from PostgreSQL", error)

# close connection with db
def close_db_connection(conn, cur):
    if conn:
        cur.close()
        conn.close()
        print("PostgreSQL connection is closed")
    else:
        print("There were no connection established")

def fetch_query(query, params=[0]):
    (conn, cur) = open_db_connection()

    print(f"Execute query: {query}")
    if len(params)!=0:
        cur.execute(query, tuple(params))
    else:
        cur.execute(query)
    
    print("Selecting rows from mobile table using cursor.fetchall")
    fetch_data = cur.fetchall()
    
    close_db_connection(conn,cur)
    
    return fetch_data

def commit_query(query, params=[0]):
    (conn, cur) = open_db_connection()

    print(f"Execute query: {query}")
    if len(params)!=0:
        cur.execute(query, tuple(params))
    else:
        cur.execute(query)
    

    conn.commit()
    print("Commit successfully!")
    
    close_db_connection(conn,cur)
    


##### help functions
def tumor_size_to_T_stage(ts):
    if ts < 20.:
        return 'T1'
    elif ts >=20. and ts < 50.:
        return 'T2'
    elif ts >=50.:
        return 'T3'

def tnm_to_stage_6th(t_stage, n_stage, m_stage = 'M0'):
    tnm_to_stage_dict = {
        "T1N1": "IIA",
        "T2N0": "IIA",
        "T2N1": "IIB",
        "T1N2": "IIIA",
        "T2N2": "IIIA",
        "T2N3": "IIIA",
        "T3N1": "IIIA",
        "T3N2": "IIIA",
        "T4N1": "IIIB",
        "T4N2": "IIIB",
        "T4N3": "IIIB",
        "T1N3": "IIIC",
        "T2N3": "IIIC",
        "T3N3": "IIIC",
        "T4N3": "IIIC",
    }
    tn_stage = t_stage + n_stage
    return tnm_to_stage_dict[tn_stage]

def positive_node_rate_to_N_stage(rate):
    if rate < 0.27:
        return 'N1'
    elif (rate >=0.27 and rate < 0.57):
        return 'N2'
    else:
        return 'N3'

def grade_to_differentiate(grade):
    grade_to_differentiate_dict = {
        '1': 'Well differentiated',
        '2': 'Moderately differentiated',
        '3': 'Poorly differentiated',
        ' anaplastic; Grade IV': 'Undifferentiated'
    }
     
    return grade_to_differentiate_dict[grade]

def prepare_X_pred(age, race, marital_status, tumor_size, tstage, nstage, grade, astage, estrogen_status, progesteron_status, node_examined, positive_node_rate):
    #features_range = { col: X[col].unique() if type(X[col].unique()[0])== str else [0, X[col].mean(), X[col].std()] for col in X.columns}
    X_pred = pd.DataFrame({ col : [0] for col in ['Age', 'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage',
       'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
       'Progesterone Status', 'Regional Node Examined',
       'Reginol Node Positive']})
    X_pred.loc[0, "Race"] = race
    X_pred.loc[0, "Marital Status"] = marital_status
    X_pred.loc[0, "Age"] = age
    X_pred.loc[0, "Tumor Size"] = tumor_size
    X_pred.loc[0, "Regional Node Examined"] = node_examined
    X_pred.loc[0, "T Stage "] = tstage
    X_pred.loc[0, "N Stage"] = nstage
    X_pred.loc[0, "Grade"] = grade
    X_pred.loc[0, "A Stage"] = astage
    X_pred.loc[0, "Estrogen Status"] = estrogen_status
    X_pred.loc[0, "Progesterone Status"] = progesteron_status
    X_pred.loc[0,"Reginol Node Positive"] = int(node_examined * (positive_node_rate/100.))
    X_pred.loc[0,'6th Stage'] = tnm_to_stage_6th(tstage, nstage)
    X_pred.loc[0,'differentiate'] = grade_to_differentiate(grade)
    return X_pred


def X_transformer(df, X_to_transform):
    
    ### add new feature
    print("X_to_transform  columns", X_to_transform.columns)
    
    X = df.drop(['Status','Survival Months'], axis=1)
    X['Positive Node Rate'] = X['Reginol Node Positive'] / X['Regional Node Examined']
    if 'Positive Node Rate' not in X_to_transform.columns:
        X_to_transform['Positive Node Rate'] = X_to_transform['Reginol Node Positive'] / X_to_transform['Regional Node Examined']
    print("X columns:", X.columns)

    ###
    drop_features = ['Marital Status', 'differentiate']

    num_features = ['Age', 'Tumor Size', 'Regional Node Examined',
       'Reginol Node Positive', 'Positive Node Rate']
    cat_features = ['Race', 'T Stage ', 'N Stage', 
                '6th Stage', 'Grade', 'A Stage', 
                'Estrogen Status','Progesterone Status']
    ordinal_features = []

    ### pipelines
    numeric_scale_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
               ('scaler', StandardScaler()),          
              ])

    ordinal_transformer = Pipeline(steps=[
        ('imputer1', SimpleImputer(strategy='constant', fill_value='absent')), 
        ('imputer2', SimpleImputer(missing_values = None, strategy='constant', fill_value='absent')),
        ('ordenc', OrdinalEncoder()),
        ('scaler', MinMaxScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer1', SimpleImputer(strategy='constant', fill_value='absent')), 
        ('imputer2', SimpleImputer(missing_values = None, strategy='constant', fill_value='absent')),
        ('onehotenc', OneHotEncoder())])


    #############################################################
    preprocessor = ColumnTransformer(transformers=[
        ('drop', 'drop', drop_features),
        ('num_scal', numeric_scale_transformer, num_features),
        ('cat', categorical_transformer, cat_features),
        ('ordinal', ordinal_transformer, ordinal_features)
    ])
    preprocessor.fit(X)

    return preprocessor.transform(X_to_transform)



app = FastAPI(title="Breast Cancer")





#### GET

## main info
@app.get("/",
 
    description="""BREAST CANCER PREDICTION MODEL API NAVIGATION INFO""",
)
async def info():
    
    return "BREAST CANCER PREDICTION MODEL API WORKS --> Check more details in /docs"
    



### get the list of models from data base (not all columns)

@app.get("/list_models/",
    response_description ="""{
        "rfb1": {'name': "Random Forest", 'test_score': 0.93, 'balanced': 'yes', 'time_created': '06/09/2022' },
        "rf": {'name': "Random Forest", 'test_score': 0.87, 'balanced': 'no', 'time_created': '06/09/2022'},
}""",
    description="""Get request without parameters return dictionary of models presented in database and ready to use
    to make prediction""",
)
async def list_models():

    get_models_query = """SELECT DISTINCT short_name, info FROM "public.models";"""
    models = fetch_query(get_models_query)

    response = { model[0]: {'description': model[1]} for model in models}


    #3 transform the list to dict
    #m = Model()
    return response
    



@app.get("/prediction/",
    response_description ="""
{
  "model": "lgb1",
  "prediction": "Dead",
  "pred_prob": 0.7632779676268621
}
""",
    description=""" /prediction/?m=lgb1&age=43&race=White&marital_status=Married&tstage=T2&nstage=N3&grade=3&astage=Regional&estrogen_status=Positive&progesteron_status=Negative&tumor_size=30&node_examined=23&positive_node_rate=90 
 Request takes multiple parameters and return prediction of survival for the patient with breast cancer.""",
)

async def prediction(m: str, age: int, race: Race, marital_status: MaritalStatus, tstage: TStage, nstage: NStage, grade: Grade, astage: AStage, estrogen_status: EstrogenStatus, progesteron_status: ProgesteroneStatus, tumor_size: int, node_examined: int, positive_node_rate: int):

    
    #1. prepare model
    # fetch model binary file from db
    get_model_query = """SELECT pickle_file FROM "public.models" WHERE short_name=%s;"""
    file = fetch_query(get_model_query, [m])[0][0]
    model = pickle.loads(file)

    #2. prepare X_pred

    # get dataset table and transform to pandas db
    conn, cur = open_db_connection()
    df = pd.read_sql_query('select * from "public.data_set"', con=conn)
    close_db_connection(conn, cur)

    sql_col_names = ['age', 'race', 'marital_status', 'T_stage', 'N_stage',
       'sixth_stage', 'differentiate', 'grade', 'A_stage', 'tumor_size',
       'estrogen_status', 'progesterone_status', 'regional_node_examined',
       'regional_node_positive', 'survival_months', 'status']
    excel_col_names = ['Age', 'Race', 'Marital Status', 'T Stage ', 'N Stage', '6th Stage',
       'differentiate', 'Grade', 'A Stage', 'Tumor Size', 'Estrogen Status',
       'Progesterone Status', 'Regional Node Examined',
       'Reginol Node Positive', 'Survival Months', 'Status']

    df.rename(columns = { old_name : new_name for old_name, new_name in zip(sql_col_names,excel_col_names)}, inplace = True)
    df.drop(['id'], axis=1, inplace=True)

    ## prepare y transformer
    y = df['Status']
    le = LabelEncoder()
    y_tr = le.fit_transform(y)
    
    # function to prepare X_pred
    X_pred = prepare_X_pred(age, race.value, marital_status.value, tumor_size, tstage.value, nstage.value, grade.value, astage.value, estrogen_status.value, progesteron_status.value, node_examined, positive_node_rate)
    print(X_pred)
    
    #3. transform X_pred
    X_pred_tr = X_transformer(df, X_pred)


    #4. predict
    y_pred = model.predict(X_pred_tr)
    y_pred_label = le.inverse_transform(y_pred)
    ###test print("PREDICTION LABEL!!! = ", y_pred_label)
    
    y_pred_proba = model.predict_proba(X_pred_tr) 
   
    y_pred_proba = y_pred_proba[0][0] if y_pred[0]==0 else y_pred_proba[0][1]

    ###test  print(f"\n Prediction probablility ={round(y_pred_proba*100,2)}% \n")
    

    # post test patient data to db
    sixth_stage=tnm_to_stage_6th(tstage, nstage)
    differentiate=grade_to_differentiate(grade)
    node_positive=int(node_examined*(positive_node_rate/100))
    params = [age, race.value, marital_status.value, 
          tstage.value, nstage.value, sixth_stage, 
          differentiate, grade.value, astage.value, 
          tumor_size, estrogen_status.value, progesteron_status.value, 
          node_examined, node_positive]

    params2 = params+params
    print("\nParameters: \n",params2)
    
    push_patient_query = """INSERT INTO "public.test_patients" (age, race, marital_status, "T_stage", "N_stage", "6th_stage", differentiate, grade, \
"A_stage", tumor_size, estrogen_status, progesterone_status, regional_node_examined, regional_node_positive) \
SELECT %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s WHERE NOT EXISTS (SELECT 1 FROM "public.test_patients" WHERE age=%s AND race=%s AND marital_status=%s AND "T_stage"=%s AND "N_stage"=%s AND "6th_stage"=%s AND differentiate=%s AND grade=%s AND "A_stage"=%s AND tumor_size=%s AND estrogen_status=%s AND  progesterone_status=%s AND regional_node_examined=%s AND regional_node_positive=%s);"""


    commit_query(push_patient_query, params+params)


    # post prediction data to db
    push_prediction_query = 'WITH inputvalues(model, status_predicted, proba_predicted, age, race, marital_status, tstage, nstage, sixth_stage, differentiate, grade, \
astage, tumor_size, estrogen_status, progesterone_status, node_examined, node_positive) \
AS (values \
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)\
)\
INSERT INTO "public.predictions" (model_id, patient_id, status_predicted, proba_predicted) \
SELECT mod.id, pat.id, d.status_predicted, d.proba_predicted \
FROM inputvalues AS d \
INNER JOIN "public.models" AS mod \
ON d.model = mod.short_name \
INNER JOIN "public.test_patients" AS pat \
ON d.age=pat.age AND \
d.race=pat.race AND \
d.marital_status=pat.marital_status AND \
d.tstage=pat."T_stage" AND \
d.nstage=pat."N_stage" AND \
d.sixth_stage=pat."6th_stage" AND \
d.grade=pat.grade AND \
d.astage=pat."A_stage" AND \
d.tumor_size=pat.tumor_size AND \
d.estrogen_status=pat.estrogen_status AND \
d.progesterone_status=pat.progesterone_status AND \
d.node_examined=pat.regional_node_examined AND \
d.node_positive=pat.regional_node_positive;'

    
    commit_query(push_prediction_query, [m, y_pred_label[0], y_pred_proba]+params)

        
    #4. prepare dict to return
    response = {'model': m, 'prediction': y_pred_label[0], "pred_prob": y_pred_proba}

    return response
    



@app.get("/get_dataset/",
        response_description = "pandas data frame in json format",
        description = "Returns breast cancer data set in form of data frame  in json format."
        )
async def get_dataset():

# connect to db
    conn, cur = open_db_connection()

    # get dataset table and transform to pandas db
    df = pd.read_sql_query('select * from "public.data_set"', con=conn)
    
    close_db_connection(conn, cur)
    

    print(df.head(2))

    return df.to_json()


if __name__=='__main__':
    uvicorn.run("test:app", host="0.0.0.0", reload=True)