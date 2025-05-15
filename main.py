# main.py

import os
import time
import mlflow
import base64
import pickle
import logging
import pandas as pd
import numpy as np
import streamlit as st
from typing import Optional, Dict, List


# Setup logging.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup page.
about = """***Job Role Recommender*** is a *Streamlit* based web app with leverages *Random Forest Machine Learning* model for predicting/recommending/suggesting a job role & required courses to achieve it to students as per their aptitude scores & career preferences. The model was trained upon dataset provided by ***Cloud Counselage-IAC*** containing students aptitude scores & career preferences like, logical quotient rating, coding skills rating, management or technical, type of company they want to settle in, etc. Models performance was continuously tracked by logging experiments using *MLflow* & its tracking URI. The model with best performance was deployed in the *streamlit* web app."""

st.set_page_config(page_title="Job Role Recommender",
                    page_icon="🎓", menu_items={"About": f"{about}"})
st.title(body="Still confused about your dream Job Role? Let us help you out! 🎓💼")
st.markdown(
    "*Our **Job Role Recommender** will suggest you most apt Job Role & courses to get started with preps for achieving it. **Good Luck! 🧿**...*")

# Initialize session state
if 'mlf_sk_model_v1' not in st.session_state:
    st.session_state.mlf_sk_model_v1 = None
if 'mlf_sk_model_v2' not in st.session_state:
    st.session_state.mlf_sk_model_v2 = None
if 'mlf_artifacts' not in st.session_state:
    st.session_state.mlf_artifacts = None

# Set mlflow tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))


# Model loading function
def load_model(model_alias: str, model_name: str="job_role_recommender") -> Optional[object]:
    """
    Load the model from the specified MLflow model URI.

    Args:
        model_name (str): The MLflow model name.

    Returns:
        Optional[object]: The loaded model or None if loading fails.
    """
    try:
        logger.info(f"Loading model from {model_name}")
        model_uri = f"models:/{model_name}@{model_alias}"
        sk_model = mlflow.sklearn.load_model(model_uri=model_uri)
        logger.info("Model loaded successfully.")
        st.success(f"{model_name}:{model_alias} model loaded successfully!", icon="✅")
        return sk_model
    except Exception as e:
        logger.critical(f"Error loading model: {e}")
        st.error(f"Error loading model: {e}", icon="🚨")
        return None


# Downloading model & artifacts
if "mlf_sk_model_v1" not in st.session_state or st.session_state.mlf_sk_model_v1 is None:
    st.session_state.mlf_sk_model_v1 = load_model(model_alias="challenger1")
if "mlf_sk_model_v2" not in st.session_state or st.session_state.mlf_sk_model_v2 is None:
    st.session_state.mlf_sk_model_v2 = load_model(model_alias="challenger2")
if "mlf_artifacts" not in st.session_state or st.session_state.mlf_artifacts is None:
    try:
        logger.info("Loading artifacts from MLflow.")
        st.session_state.mlf_artifacts = mlflow.artifacts.download_artifacts(run_id="48f0b164815a408fba4fd7c61e2966cf")
        logger.info("Artifacts downloaded successfully.")
        st.success("Artifacts downloaded successfully!", icon="✅")
    except Exception as e:
        logger.critical(f"Error downloading artifacts: {e}")
        st.error(f"Error downloading artifacts: {e}", icon="🚨")

# Loading artifacts
try:
    logger.info("Loading artifacts.")
    with open(os.path.join(st.session_state.mlf_artifacts, "le_crts/le_crts.sav"), "rb") as f1:
        le_crts = pickle.load(f1)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_intbks/le_intbks.sav"), "rb") as f2:
        le_intbks = pickle.load(f2)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_intca/le_intca.sav"), "rb") as f3:
        le_intca = pickle.load(f3)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_intsb/le_intsb.sav"), "rb") as f4:
        le_intsb = pickle.load(f4)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_memsc/le_memsc.sav"), "rb") as f5:
        le_memsc = pickle.load(f5)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_rwr/le_rwr.sav"), "rb") as f6:
        le_rwr = pickle.load(f6)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_tycompt/le_tycompt.sav"), "rb") as f7:
        le_tycompt = pickle.load(f7)
    with open(os.path.join(st.session_state.mlf_artifacts, "le_wkshp/le_wkshp.sav"), "rb") as f8:
        le_wkshp = pickle.load(f8)
    with open(os.path.join(st.session_state.mlf_artifacts, "ohe/one_hot_encoder.sav"), "rb") as f9:
        ohe = pickle.load(f9)
    with open(os.path.join(st.session_state.mlf_artifacts, "ohe_cols/ohe_cols.sav"), "rb") as f10:
        ohe_cols = pickle.load(f10)
    logger.info("Artifacts loaded successfully.")
    st.success("Artifacts loaded successfully!", icon="✅")
except Exception as e:
    logger.critical(f"Error loading artifacts: {e}")
    st.error(f"Error loading artifacts: {e}", icon="🚨")
    st.stop()

# User input section
st.markdown("----")
st.header("Please fill in the following details to get your Job Role & courses recommendation👇:")
st.markdown(":red[***Note:** All fields are mandatory.*]")
st.markdown("----")

# Taking user input
log_quot: int = st.slider("Logical Quotient Rating (0-9)", min_value=0, max_value=9, value=5, step=1, help="Logical Quotient Rating (0-9) - 0 being the lowest & 9 being the highest rating.")
cod_sk: int = st.slider("Coding Skills Rating (0-9)", min_value=0, max_value=9, value=5, step=1, help="Coding Skills Rating (0-9) - 0 being the lowest & 9 being the highest rating.")
hackathons: int = st.slider("Hackathons Participation (0-9)", min_value=0, max_value=9, value=5, step=1, help="Hackathons Participation (0-9) - 0 being the lowest & 9 being the highest rating.")
pub_speak: int = st.slider("Public Speaking Skills Rating (0-9)", min_value=0, max_value=9, value=5, step=1, help="Public Speaking Skills (0-9) - 0 being the lowest & 9 being the highest rating.")

SELE_LEARN_MAP: Dict[str, str] = {"yes": "Yes, I can learn new skills on my own. :thumbsup:", "no": "No, I need help learning new skills. :thumbsdown:"}
self_learn: str = st.pills("Can you learn new skills on your own?", options=SELE_LEARN_MAP, default="yes", format_func=lambda option: SELE_LEARN_MAP[option], help="Can you learn new skills on your own? - Yes or No")

XTRA_COURSE_MAP: Dict[str, str] = {"yes": "Yes, I did extra courses. 🎓📚", "no": "No, I did not take extra courses. 🎓"}
xtra_courses: str = st.pills("Did you take any extra courses?", options=XTRA_COURSE_MAP, default="yes", format_func=lambda option: XTRA_COURSE_MAP[option], help="Did you take any extra courses? - Yes or No")

TAKEN_GUIDANCE_MAP: Dict[str, str] = {"yes": "Yes, I took guidance from Elders/Seniors/Others. :thumbsup:", "no": "No, I did not take guidance from anyone. :thumbsdown:"}
taken_guidance: str = st.pills("Did you take guidance from Elders/Seniors/Others?", options=TAKEN_GUIDANCE_MAP, default="yes", format_func=lambda option: TAKEN_GUIDANCE_MAP[option], help="Did you take guidance from Elders/Seniors/Others? - Yes or No")

TEAM_WORK_MAP: Dict[str, str] = {"yes": "Yes, I am familiar to working in teams. 👥", "no": "No, I cannot work in a team. 👤"}
team_work: str = st.pills("Are you familiar to working in teams?", options=TEAM_WORK_MAP, default="yes", format_func=lambda option: TEAM_WORK_MAP[option], help="Are you familiar to working in teams? - Yes or No")

INTROVERT_MAP: Dict[str, str] = {"yes": "Yes, I am an introvert. 🫣", "no": "No, I am not an introvert. 😎"}
introvert: str = st.pills("Are you an introvert?", options=INTROVERT_MAP, default="yes", format_func=lambda option: INTROVERT_MAP[option], help="Are you an introvert? - Yes or No")

EMP_MAP: Dict[str, str] = {"excellent": "Excellent ✅", "medium": "Medium ☑️", "poor": "Poor ❌"}
rdwrskl: str = st.pills("Rate your reading-writing skills", options=EMP_MAP, default="excellent", format_func=lambda option: EMP_MAP[option], help="Rate your reading-writing skills - Excellent, Medium or Poor", key="rdwrskl_pills")

# mem_cap_map: Dict[str, str] = {"excellent": "Excellent ✅", "medium": "Medium ☑️", "poor": "Poor ❌"}
mem_cap: str = st.pills("Rate your memory capacity", options=EMP_MAP, default="excellent", format_func=lambda option: EMP_MAP[option], help="Rate your memory capacity - Excellent, Medium or Poor", key="memcap_pills")

MGMNT_TECH_MAP: Dict[str, str] = {"Management": "Management 💼", "Technical": "Technical 🚀"}
mgmnt_tech: str = st.pills("What is your job role preference?", options=MGMNT_TECH_MAP, default="Management", format_func=lambda option: MGMNT_TECH_MAP[option], help="What is your preference? - Management or Technical")

HARD_SMART_MAP: Dict[str, str] = {"hard worker": "Hard Worker 💪", "smart worker": "Smart Worker 🧠"}
hard_smart: str = st.pills("Are you a hard worker or a smart worker?", options=HARD_SMART_MAP, default="hard worker", format_func=lambda option: HARD_SMART_MAP[option], help="Are you a hard worker or a smart worker? - Hard Worker or Smart Worker")

CRTS_LIST: List[str] = ['information security', 'shell programming', 'r programming',
       'distro making', 'machine learning', 'full stack', 'hadoop',
       'app development', 'python']
crts: str = st.selectbox("Select the courses you have taken", options=CRTS_LIST, index=0, help="Select the courses you have taken - Select from the list")

WKSHP_LIST: List[str] = ['testing', 'database security', 'game development', 'data science',
       'system designing', 'hacking', 'cloud computing',
       'web technologies']
wkshps: str = st.selectbox("Select the workshops you have attended", options=WKSHP_LIST, index=0, help="Select the workshops you have attended - Select from the list")

INT_SUB_LIST: List[str] = ['programming', 'Management', 'data engineering', 'networks',
       'Software Engineering', 'cloud computing', 'parallel computing',
       'IOT', 'Computer Architecture', 'hacking']
int_subs: str = st.selectbox("Select the subjects you are interested in", options=INT_SUB_LIST, index=0, help="Select the subjects you are interested in - Select from the list")

INT_BKS_LIST: List[str] = ['Series', 'Autobiographies', 'Travel', 'Guide', 'Health',
       'Journals', 'Anthology', 'Dictionaries', 'Prayer books', 'Art',
       'Encyclopedias', 'Religion-Spirituality', 'Action and Adventure',
       'Comics', 'Horror', 'Satire', 'Self help', 'History', 'Cookbooks',
       'Math', 'Biographies', 'Drama', 'Diaries', 'Science fiction',
       'Poetry', 'Romance', 'Science', 'Trilogy', 'Fantasy', 'Childrens',
       'Mystery']
int_bks: str = st.selectbox("Select the type of books you read", options=INT_BKS_LIST, index=0, help="Select the type of books you read - Select from the list")

INT_CAREER_LIST: List[str] = ['testing', 'system developer', 'Business process analyst',
       'security', 'developer', 'cloud computing']
int_career: str = st.selectbox("Select your interested career", options=INT_CAREER_LIST, index=0, help="Select your interested career - Select from the list")

TYPE_COMP_LIST: List[str] = ['BPA', 'Cloud Services', 'product development',
       'Testing and Maintainance Services', 'SAaS services',
       'Web Services', 'Finance', 'Sales and Marketing', 'Product based',
       'Service Based']
type_comp: str = st.selectbox("Select the type of company you want to settle in", options=TYPE_COMP_LIST, index=0, help="Select the type of company you want to settle in - Select from the list")

# Select model version
MODEL_VRSN_MAP: Dict[str, str] = {"challenger1": "Version 1️⃣", "challenger2": "Version 2️⃣"}
model_version: str = st.pills("Select the model version to use for prediction", options=MODEL_VRSN_MAP, default="challenger2", format_func=lambda option: MODEL_VRSN_MAP[option], help="Select the model version to use for prediction - Version 1 or Version 2")

random_seed: int = st.number_input("Random Seed", min_value=0, max_value=100, value=42, step=1, help="Random Seed - Random seed for reproducibility")

# Button to get recommendation
pred_btn = st.button("***Get Job Role & Courses Recommendation***", help="Click to get Job Role & Courses recommendation")
st.markdown("----")


# Prepare input data for prediction
df: pd.DataFrame= pd.DataFrame({
    "Logical quotient rating": [log_quot],
    "coding skills rating": [cod_sk],
    "hackathons": [hackathons],
    "public speaking points": [pub_speak],
    "self-learning capability?": [self_learn],
    "Extra-courses did": [xtra_courses],
    "Taken inputs from seniors or elders": [taken_guidance],
    "worked in teams ever?": [team_work],
    "Introvert": [introvert],
    "reading and writing skills": [rdwrskl],
    "memory capability score": [mem_cap],
    "Management or Technical": [mgmnt_tech],
    "hard/smart worker": [hard_smart],
    "certifications": [crts],
    "workshops": [wkshps],
    "Interested subjects": [int_subs],
    "Interested Type of Books": [int_bks],
    "interested career area ": [int_career],
    "Type of company want to settle in?": [type_comp]
})

# Encoding categorical features
df['certifications'] = le_crts.transform(df['certifications'])
df['workshops'] = le_wkshp.transform(df['workshops'])
df['Interested subjects'] = le_intsb.transform(df['Interested subjects'])
df['Interested Type of Books'] = le_intbks.transform(df['Interested Type of Books'])
df['interested career area '] = le_intca.transform(df['interested career area '])
df['Type of company want to settle in?'] = le_tycompt.transform(df['Type of company want to settle in?'])
df['memory capability score'] = le_memsc.transform(df['memory capability score'])
df['reading and writing skills'] = le_rwr.transform(df['reading and writing skills'])

# OHE encoding
df_ohe = pd.DataFrame(ohe.transform(df[ohe_cols]), columns=ohe.get_feature_names_out(ohe_cols))
df = pd.concat([df.drop(ohe_cols, axis=1), df_ohe], axis=1)

# Creating interactive features
df['Technical_Inclination'] = (
    df['Logical quotient rating'] +
    df['coding skills rating'] +
    df['Interested subjects']
)
df['Teamwork_Comm_Aptitude'] = (
    df['public speaking points'] +
    df['reading and writing skills'] +
    (df['worked in teams ever?_yes'] * 2)
)
df['Proactive_Learning'] = (
    (df['self-learning capability?_yes'] * 2) +
    (df['Extra-courses did_yes'] * 1.5) +
    df['certifications'] +
    df['workshops']
)
df['Company_Fit'] = (
    df['Type of company want to settle in?']  +
    df['hard/smart worker_smart worker']
)
df['Experience_Exposure'] = (
    df['hackathons'] +
    df['coding skills rating']
)

# Confirming the model version
match model_version:
    case "challenger1":
        model_ = st.session_state.mlf_sk_model_v1
    case "challenger2":
        model_ = st.session_state.mlf_sk_model_v2

# Making prediction
st.subheader("Your Job Role & Courses Recommendation:")
np.random.seed(random_seed)
if pred_btn:
    with st.spinner("Making prediction..."):
        time.sleep(2)
        try:
            logger.info("Making prediction.")
            pred = model_.predict(df[model_.feature_names_in_])
            pred_proba = model_.predict_proba(df[model_.feature_names_in_])
            logger.info(f"Prediction made successfully. Predicted job role: {pred[0]}")
            st.success("Prediction made successfully!", icon="✅")
        except Exception as e:
            logger.critical(f"Error making prediction: {e}")
            st.error(f"Error making prediction: {e}", icon="🚨")
            st.stop()

    # Displaying the prediction
    # st.markdown("----")
    st.write(f"**Job Role:** :orange[{pred[0]}]")
    # st.write(f"**Courses to get started with:** {pred_proba[0]}")
    if np.max(pred_proba) > 0.79:
        txt2 = f":green[{np.max(pred_proba):.2%}]"
    elif np.max(pred_proba) > 0.49:
        txt2 = f":blue[{np.max(pred_proba):.2%}]"
    else:
        txt2 = f":red[{np.max(pred_proba):.2%}]"
    st.write(f":gray[*Model Confidence on Prediction:*] {txt2}")
    # st.markdown("----")


# Disclaimer
st.write("\n"*3)
st.markdown("""----""")
st.write("""*Disclaimer: Predictions made by the models may be inaccurate due to the nature of the models & data inadequacy. This is a simple demonstration of how machine learning can be used to recommend/suggest job roles based on aptitude scores and career preferences. For more accurate predictions, consider using more complex models, larger datasets, and field knowledge oriented feature engineering. Also, after getting your job role recommendation from above models consider visiting/consulting with a career coach/counselor for more detailed advise.*""")


def get_image_base64(image_path):
    """Read image file."""
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
st.markdown("""---""")
st.markdown("Created by [Pranay Jagtap](https://pranayjagtap.netlify.app)")

# Get the base64 string of the image
img_base64 = get_image_base64("assets/pranay_blusq.jpg")

# Create the HTML for the circular image
html_code = f"""
<style>
    .circular-image {{
        width: 125px;
        height: 125px;
        border-radius: 55%;
        overflow: hidden;
        display: inline-block;
    }}
    .circular-image img {{
        width: 100%;
        height: 100%;
        object-fit: cover;
    }}
</style>
<div class="circular-image">
    <img src="data:image/jpeg;base64,{img_base64}" alt="Pranay Jagtap">
</div>
"""

# Display the circular image
st.markdown(html_code, unsafe_allow_html=True)
# st.image("assets/pranay_sq.jpg", width=125)
st.markdown("Machine Learning Enthusiast | Electrical Engineer"\
            "<br>📍 Nagpur, Maharashtra, India", unsafe_allow_html=True)
