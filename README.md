# Cloud Counselage - Industry Academia Community (IAC) - Machine Learning Live Project - [IP-6281]

> This is a **Machine Learning Live Project** repository for the ML project I was assigned at ***Cloud Counselage-IAC***. This repo contains all the files required for training & deploying ML model.

[***Visit Streamlit Job Role Recommender Portal***](https://cc-iac-machine-learning-live-project.streamlit.app/)
[***Project Video Report***](https://drive.google.com/file/d/1bIJeomxCeRXxPMvV4vB_T7yw-fK6VTLC/view?usp=drive_link)

### Task

----

Build a system/web application for your final year undergraduate students, which will suggest an appropriate job role and a course which help them secure the suggested job role by leveraging ML model. The model will utilize essential student's aptitude & career/subject preference data for predicting an appropriate job role for the student, and the app will suggest the required course. The model will be trained on dataset containing student's aptitude & career/subject preference records.

### Outcome: Job Role Recommender

----

***Job Role Recommender*** is a *Streamlit* based web app with leverages *Random Forest Machine Learning* model for predicting/recommending/suggesting a job role to students as per their aptitude scores & career preferences. The model was trained upon dataset provided by ***Cloud Counselage-IAC*** containing students aptitude scores & career preferences like, logical quotient rating, coding skills rating, management or technical, type of company they want to settle in, etc. Models performance was continuously tracked by logging experiments using *MLflow* & its tracking URI. The model with best performance was deployed in the *streamlit* web app.

### Documentation

----

#### 1. Docker Environment Setup

 - Download and install `Docker` and `Docker Desktop` for your system/OS: [Docker Installation Docs](https://docs.docker.com/engine/install/), [Docker Desktop Installation Docs](https://docs.docker.com/desktop/)
 - (*Only for linux users*) Before running `docker` after installation, run this command first: `sudo usermod -aG docker $USER`. This command adds the user to the docker group enabling user to run docker commands without elevated privilages.
 - Copy & paste `Dockerfile`, `docker-compose.yaml`, `requirements-workspace.txt`, and `start` to your desired workspace folder/directory.
 - Create `.env` & `.dockerignore` files. Add your environment variables like `REPO_OWNER`, `REPO_NAME`, and `MLFLOW_TRACKING_URI`, these are needed to work with MLFlow. 
     > ***DO NOT ADD ENVIRONMENT VARIABLES & SECRETS DIRECTLY INTO YOUR Dockerfile IF YOU ARE COMMITTING IT TO VCS***. 

    Populate the `.dockerignore` file with this:
       
       .env
       *.env
       .env.*
       __pycache__
       *.pyc

    Feel free add as many as you feel necessary.
 - Now open the terminal or powershell/cmd in your workspace folder and run these docker commands: 
    
       # Initiates building docker services
       COMPOSE_MAKE=true DOCKER_BUILDKIT=1 docker compose build --parallel --no-cache

       # Check if images are created. You may find two new images: `ml-brain-container-v2` & `ml-brain-gpu-container-v2`
       docker images

       # Now let's start the containers
       docker compose up -d

       # Check the logs after few minutes
       docker logs ml-brain-container-v2
       docker logs ml-brain-gpu-container-v2

       # If the jupyter lab URL is visible in the logs, it means the container is up & running. You can confirm this too by running below command
       docker container ls

       # Exec into the container and check if workspace directories are mounted in container env
       docker exec -it ml-brain-container-v2 ls workspace
       docker exec -it ml-brain-gpu-container-v2 ls workspace

#### 2. Setup VSCode
 - Download & install [VSCode](https://code.visualstudio.com/) if not installed already.
 - Signup with your Microsoft or GitHub account, if required, and install necessary extensions. Python, Dev Containers & Docker extensions are must for our workflow.
 - Press `Ctrl + Shift + P` to open command palettle and type "Dev Containers" and look for "Attach to Running Container" and select it.
 - In the next pop-up connect to the container of your choice, and wait for it to complete setup.
 - Install your local extensions into the running container.

#### 3. Train Model OR Run Streamlit App
 - Open the jupyter notebook from the project folder and run subsequent code cells to train the models.
 - Edit the model aliases if needed in `main.py`.
 - Run `streamlit run main.py` from the root dir.
