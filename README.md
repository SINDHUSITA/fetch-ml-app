
# Fetch Machine Learning Assesment

A simple ML app to forcast the future receipt counts per day that fetch receives for 2022, given the data for 2021.

### Data:
* The given data represents time series information of receipts scanned for a year 2021. The data has 1 feature variable, ‘#Date’ and 1 dependent variable, ‘Receipt_Count’.
* There are 365 data points representing counts per each day.

### Application and its Components
* The application is backed by a pytorch model, where it uses deep learning, a neural network to predict the receipt counts for future year, 2022.
* **Preprocessing**: As part of preprocessing, the date feature is split to day, month and year features for training. This split is not mandatory for training. Alternatively, sine, cosine or rbf encodings can be done for datetime encoding.
* **Visualization**: Visualizations are made to display how the counts have been varying over 2021. Similarly, after training the model, results can be seen for the year 2022.
#### Components in the application
**The Model**: Data Preprocessing, Model training, Model Inference

**StreamLit Application**: Load the input file, Visualization and Results

* Here, initially the model is trained and the saved model is used for future inference/predictions. 
* Model training is part of a backend process and saves the trained model to the directory.
* To run the train the model alone initially, or retain the model, use:
`python train.py`

* However this step can be skipped even while running for the first time because, the model that has been trained for development purpose is pushed to the directory.
* The streamlit application allows one to upload the data in the required format and displays results for fututre.
* This Streamlit application is deployed on docker and docker image is shared in the readme.md
### How to run the application in local:
To run the streamlit application, run the following commands in the cloned directory: 

* `pip install -r requirements.txt`
* `streamlit run streamlit_main.py`

### Run the application with Dockerfile locally:
For this you will need docker installed.
* for docker build : `docker build -t username/imagename:v1 .`
* for docker run : `docker run -p 8501:8501 imagename:v1`

### Run with published docker-image 

* `docker pull pvvsdoc001/fetch-ml-app:v1`

* `docker run -p 8501:8501 pvvsdoc001/fetch-ml-app:v1`

** ***The Application will be accessible on http://localhost:8501/*** **

### How to use the application:
* Displayed images are results before and after prediction with the saved model
* Upload the file 'daily_data.csv' in the side bar, to test with new data
* Can download the final image with the button

### Future Scope:
* As this is a POC project which is at a rudimentary phase, exception handling and unit testing are not taken care of. Can be added later
* More visualizations for the data can be added, like month wise count distributions, weekly predictions
* The model and streamlit app can be added and deployed as seperate microservices.
