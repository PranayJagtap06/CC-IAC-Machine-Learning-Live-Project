# Cloud Counselage - Industry Academia Community (IAC) - Machine Learning Live Project

> This is a **Machine Learning Live Project** repository for the ML project I was assigned at ***Cloud Counselage-IAC***. This repo contains all the files required for training & deploying ML model.

### Task

----

Build a system/web application for your final year undergraduate students, which will suggest an appropriate job role and a course which help them secure the suggested job role by leveraging ML model. The model will utilize essential student's aptitude & career/subject preference data for predicting an appropriate job role for the student, and the app will suggest the required course. The model will be trained on dataset containing student's aptitude & career/subject preference records.

### Outcome: Job Role Recommender

----

***Job Role Recommender*** is a *Streamlit* based web app with leverages *Random Forest Machine Learning* model for predicting/recommending/suggesting a job role to students as per their aptitude scores & career preferences. The model was trained upon dataset provided by ***Cloud Counselage-IAC*** containing students aptitude scores & career preferences like, logical quotient rating, coding skills rating, management or technical, type of company they want to settle in, etc. Models performance was continuously tracked by logging experiments using *MLflow* & its tracking URI. The model with best performance was deployed in the *streamlit* web app.

