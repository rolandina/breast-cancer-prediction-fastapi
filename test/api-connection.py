import json
import requests
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


class TStage(Enum):
    t1 = "T1"
    t2 = "T2"
    t3 = "T3"
    t4 = "T4"

class NStage(Enum):
    n1 = "N1"
    n2 = "N2"
    n3 = "N3"

#may be produced from n and t stages
class SixthStage(Enum):
    IIA = "IIA"
    IIIA = "IIIA"
    IIIC = "IIIC"
    IIB = "IIB"
    IIIB = "IIIB"

class Grade(Enum):
    I = "1"
    II = "2"
    III = "3"
    IV = " anaplastic; Grade IV"

class AStage(Enum):
    regional = "Regional"
    distant = "Distant"

class EstrogenStatus(Enum):
    positive = "Positive"
    negative = "Negative"

class ProgesteroneStatus(Enum):
    positive = "Positive"
    negative = "Negative"



################### SET PARAMETERS #############################################

age = 30
positive_node_examined = 20
positive_node_rate = 30 # percentage
tumor_size = 12

race = Race.white.value
marital_status = MaritalStatus.divorced.value
tstage = TStage.t2.value
nstage = NStage.n2.value
astage =  AStage.regional.value
grade = Grade.III.value
progesterone_status = ProgesteroneStatus.positive.value
estrogen_status = EstrogenStatus.positive.value

params = [age, positive_node_examined, positive_node_rate, tumor_size,
race,marital_status,tstage,nstage,astage,grade,progesterone_status,estrogen_status]

print('\n *********** PRINT PARAMETERS ******************* \n')

for p in params:
    print(f"{p}\n")

print('\n *********** TEST list_models endpoint ******************* \n')
#choose model
model = 'lgb1' #by default 
url_list_models = 'https://bcp-fast-api.herokuapp.com/list_models/'
response_models = requests.get(url_list_models)
models = json.loads(response_models.text)
print(models)

print('\n ***************** TEST PREDICTION ENDPOINT ****************** \n')

url_prediction = f"https://bcp-fast-api.herokuapp.com/prediction/?m={model}&\
age={age}&race={race}&marital_status={marital_status}&tstage={tstage}&\
nstage={nstage}&grade={grade}&astage={astage}&estrogen_status={estrogen_status}&\
progesterone_status={progesterone_status}&tumor_size={tumor_size}&node_examined={positive_node_examined}&\
positive_node_rate={positive_node_rate}"

print(url_prediction)

response_prediction = requests.get(url_prediction)
print("STATUS CODE", response_prediction.status_code)
prediction = json.loads(response_prediction.content)
print(prediction)