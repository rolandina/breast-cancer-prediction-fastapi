# Breast Cancer Prediction Model API

This is a FastAPI application for a Breast Cancer Prediction Model. The API can be used to predict the survival of a patient with breast cancer based on several parameters such as age, race, marital status, estrogen status, progesterone status, tumor size, and more...

![](https://github.com/rolandina/breast-cancer-prediction-fastapi/blob/api/bcp-fast-api.png)

## Setup

First, clone the repository and navigate into the project directory.

Install the necessary dependencies with:

```shell
pip install -r requirements.txt
```

To run the API successfully you need to add .env file with DATABASE_URL parameter. You can ask me and I can give you the copy of the database, I used.

## Run the application

Start the server using:

```shell
uvicorn main:app --reload
```

You can then navigate to http://localhost:8000 in your web browser. The API documentation is available at http://localhost:8000/docs.

## Endpoints

Here are the main endpoints available in the application:

- GET / : Returns a greeting message.
- GET _/model_info/_ : Returns a dictionary with the parameters of the model used for predictions.
- GET _/prediction/_ : Returns a prediction of survival for a patient with breast cancer based on several parameters.

## Usage

To use the _/prediction_ endpoint, send a GET request with the following parameters:

- age: Integer
- race: String (Values: "White", "Black", "Other")
- marital_status: String (Values: "Married", "Divorced", "Single ", "Widowed", "Separated")
- tstage: String (Values: "T1", "T2", "T3", "T4")
- nstage: String (Values: "N1", "N2", "N3")
- grade: String (Values: "1", "2", "3", " anaplastic; Grade IV")
- astage: String (Values: "Regional", "Distant")
- estrogen_status: String (Values: "Positive", "Negative")
- progesterone_status: String (Values: "Positive", "Negative")
- tumor_size: Integer
- node_examined: Integer
- positive_node_rate: Integer

Example usage:

Start the server using:

```shell
curl -X GET "http://localhost:8000/prediction/?age=45&race=White&marital_status=Divorced&tstage=T2&nstage=N3&grade=%20anaplastic%3B%20Grade%20IV&astage=Distant&estrogen_status=Positive&progesterone_status=Positive&tumor_size=33&node_examined=1&positive_node_rate=1"
```

## Development

Make sure you have the correct environment variables set up for your database.

## License

This is a study project is which used open-source data.
Code in this project is available under the MIT License.
