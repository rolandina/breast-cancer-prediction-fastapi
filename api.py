from fastapi import FastAPI, Path, HTTPException
#import uvicorn
#import os

#from titanic.model import Model
from enum import Enum

# load environment variables
#port = os.environ["PORT"]


### here connect to db
def connect_to_db():
    conn = 1
    return conn

def get_model(m):

    #connect
    con = connect_to_db()
    model = con.get() 
    #  return model from db

    return model


app = FastAPI(title="Breast Cancer")

class Race(str, Enum):
    white = "White"
    black = "Black"
    other = "Other"

class MaritalStatus(str, Enum):
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



#### GET

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
    #1. connect to postegres db

    #2. get the list of models

    #3 transform the list to dict
    #m = Model()
    return "this is list model get request"
    

# get line from db with demanded model
@app.get("/get_model/",
    response_description ="""{
        "rfb1": {'name': "Random Forest", 'test_score': 0.93, 'balanced': 'yes', 'binary_pickle': "dfkw9er" , 'time_created': '06/09/2022' },
        
}""",
    description="""Request the model from models table of postegres breast cancer data base. 
    Get the dictionary with binary pickle model""",
)
async def get_model(m: str):
    #1. connect to postegres db

    #3 return line from db
    return f"model = {m} --> this should be line from db with pickle model"
    


@app.get("/prediction/",
    response_description ="""{
  "result": 1,
  "text_result": "survived",
  "probability": 69,
  "description": "The passager survived with probability 69%"
}""",
    description="/prediction/?model={m}&race={race}&marital_status={marital_status}&tstage={tstage}&nstage={nstage}&grade={grade}&astage={astage}&estrogen_status={estrogen_status}&progesteron_status={progesteron_status} takes 6 parameters and \
    return the answer if passager died or survived",
)

async def prediction(m: str, age: int, race: Race, marital_status: MaritalStatus, tstage: TStage, nstage: NStage, fare: float, embarked: Embarked):

    
    #1. prepare model
    model = m
    #2. prepare X_pred
    X_pred = []

    #3. predict
    pred = model.predict(X_pred)

    # post test patient data to db

    # post prediction data to db
        
    #4. prepare dict to return
    response = {'prediction': 'alive', "pred_prob": 0.99}

    return response
    



@app.get("/get_dataset/",
        response_description = "pandas data frame in json format",
        description = "Returns breast cancer data set in form of data frame  in json format."
        )
async def get_dataset():

    # connect to db
    # get dataset table and transform to pandas db
    # transform df to json
    return Model().titanic.to_json(orient= 'columns')


# if __name__=='__main__':
#     uvicorn.run("api:app", host="0.0.0.0", port = port, reload=False)