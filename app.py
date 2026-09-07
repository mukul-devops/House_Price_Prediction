import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from html import escape
from textwrap import dedent
from numbers import Integral, Real


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AmesValue AI | Property Valuation",
    page_icon=":material/real_estate_agent:",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "AMESVALUE AI"
APP_VERSION = "1.0"

HOLDOUT_R2 = 0.8497
CV_R2 = 0.8172
VALIDATION_MAE = 21293.29

MODEL_NAME = "XGBoost"
TARGET_TRANSFORMATION = "log1p(SalePrice)"
DATASET_NAME = "Ames Housing"


# ============================================================
# MODEL PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATHS = [
    BASE_DIR / "model" / "house_price_xgb_model.pkl",
    BASE_DIR / "house_price_xgb_model.pkl",
    Path.cwd() / "model" / "house_price_xgb_model.pkl",
    Path.cwd() / "house_price_xgb_model.pkl",
]


# ============================================================
# HTML RENDER HELPER
# ============================================================

def render_html(content):
    """
    Render custom HTML using Streamlit's native HTML renderer.
    Falls back to st.markdown for older Streamlit versions.
    """
    html = dedent(content).strip()

    if hasattr(st, "html"):
        st.html(html)
    else:
        st.markdown(
            html,
            unsafe_allow_html=True,
        )


# ============================================================
# MODEL LOADER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_model():
    errors = []

    for path in MODEL_PATHS:

        if not path.exists():
            continue

        try:
            return joblib.load(path)

        except Exception as exc:
            errors.append(
                f"{path}: {exc}"
            )

    searched = "\n".join(
        str(path)
        for path in MODEL_PATHS
    )

    error_details = (
        "\n".join(errors)
        if errors
        else "No model file was found."
    )

    raise FileNotFoundError(
        "Trained model could not be loaded.\n\n"
        f"Searched locations:\n{searched}\n\n"
        f"Loading errors:\n{error_details}"
    )


try:

    model = load_model()

except Exception as exc:

    st.error(
        "Unable to load the trained model."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# MODEL STRUCTURE
# ============================================================

try:

    preprocessor = model.named_steps["preprocessor"]

    xgb_model = model.named_steps["model"]

except Exception as exc:

    st.error(
        "The saved model does not match the expected "
        "preprocessing + XGBoost pipeline structure."
    )

    st.code(
        str(exc)
    )

    st.stop()


# ============================================================
# EXTRACT NUMERICAL FEATURES
# ============================================================

try:

    numeric_features = list(
        preprocessor.transformers_[0][2]
    )

except Exception:

    numeric_features = []


# ============================================================
# EXTRACT CATEGORICAL FEATURES
# ============================================================

try:

    categorical_features = list(
        preprocessor.transformers_[1][2]
    )

except Exception:

    categorical_features = []


# ============================================================
# EXTRACT CATEGORY OPTIONS
# ============================================================

category_options = {}

try:

    categorical_pipeline = (
        preprocessor
        .named_transformers_["cat"]
    )

    encoder = (
        categorical_pipeline
        .named_steps["onehot"]
    )

    category_options = {
        feature: [
            str(category)
            for category in categories
        ]
        for feature, categories
        in zip(
            categorical_features,
            encoder.categories_,
        )
    }

except Exception:

    category_options = {}


# ============================================================
# EXPECTED MODEL FEATURES
# ============================================================

try:

    expected_features = list(
        getattr(
            model,
            "feature_names_in_",
            preprocessor.feature_names_in_,
        )
    )

except Exception:

    expected_features = (
        numeric_features +
        categorical_features
    )


# ============================================================
# MODEL SCHEMA SAFETY CHECKS
# ============================================================

if len(expected_features) != len(
    set(expected_features)
):

    st.error(
        "The trained model contains duplicate feature names."
    )

    st.stop()


if not expected_features:

    st.error(
        "Unable to determine the trained model feature schema."
    )

    st.stop()


# ============================================================
# DEFAULT NUMERICAL VALUES
# ============================================================

DEFAULT_NUMERIC = {

    "Id": 1461,
    "MSSubClass": 20,
    "LotFrontage": 70.0,
    "LotArea": 8000,
    "OverallQual": 6,
    "OverallCond": 5,
    "YearBuilt": 1970,
    "YearRemodAdd": 1970,
    "MasVnrArea": 0.0,
    "BsmtFinSF1": 500.0,
    "BsmtFinSF2": 0.0,
    "BsmtUnfSF": 500.0,
    "TotalBsmtSF": 1000.0,
    "1stFlrSF": 1000,
    "2ndFlrSF": 500,
    "LowQualFinSF": 0,
    "GrLivArea": 1500,
    "BsmtFullBath": 1,
    "BsmtHalfBath": 0,
    "FullBath": 2,
    "HalfBath": 1,
    "BedroomAbvGr": 3,
    "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 6,
    "Fireplaces": 1,
    "GarageYrBlt": 1980.0,
    "GarageCars": 2,
    "GarageArea": 400.0,
    "WoodDeckSF": 50,
    "OpenPorchSF": 40,
    "EnclosedPorch": 0,
    "3SsnPorch": 0,
    "ScreenPorch": 0,
    "PoolArea": 0,
    "MiscVal": 0,
    "MoSold": 6,
    "YrSold": 2010,
}


# ============================================================
# DEFAULT CATEGORICAL VALUES
# ============================================================

DEFAULT_CATEGORICAL = {

    "MSZoning": "RL",
    "Street": "Pave",
    "Alley": "None",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "LotConfig": "Inside",
    "LandSlope": "Gtl",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "Condition2": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "RoofStyle": "Gable",
    "RoofMatl": "CompShg",
    "Exterior1st": "VinylSd",
    "Exterior2nd": "VinylSd",
    "MasVnrType": "None",
    "ExterQual": "TA",
    "ExterCond": "TA",
    "Foundation": "PConc",
    "BsmtQual": "TA",
    "BsmtCond": "TA",
    "BsmtExposure": "No",
    "BsmtFinType1": "GLQ",
    "BsmtFinType2": "Unf",
    "Heating": "GasA",
    "HeatingQC": "Ex",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "KitchenQual": "TA",
    "Functional": "Typ",
    "FireplaceQu": "None",
    "GarageType": "Attchd",
    "GarageFinish": "RFn",
    "GarageQual": "TA",
    "GarageCond": "TA",
    "PavedDrive": "Y",
    "PoolQC": "None",
    "Fence": "None",
    "MiscFeature": "None",
    "SaleType": "WD",
    "SaleCondition": "Normal",
}


# ============================================================
# DISPLAY NAMES
# ============================================================

DISPLAY_NAMES = {

    "Id": "Property ID",
    "MSSubClass": "Building Class",
    "LotFrontage": "Lot Frontage",
    "LotArea": "Lot Area",
    "OverallQual": "Overall Quality",
    "OverallCond": "Overall Condition",
    "YearBuilt": "Year Built",
    "YearRemodAdd": "Remodel Year",
    "MasVnrArea": "Masonry Area",
    "BsmtFinSF1": "Basement Finished Area 1",
    "BsmtFinSF2": "Basement Finished Area 2",
    "BsmtUnfSF": "Basement Unfinished Area",
    "TotalBsmtSF": "Total Basement Area",
    "1stFlrSF": "First Floor Area",
    "2ndFlrSF": "Second Floor Area",
    "LowQualFinSF": "Low Quality Finished Area",
    "GrLivArea": "Above-Ground Living Area",
    "BsmtFullBath": "Basement Full Bathrooms",
    "BsmtHalfBath": "Basement Half Bathrooms",
    "FullBath": "Full Bathrooms",
    "HalfBath": "Half Bathrooms",
    "BedroomAbvGr": "Bedrooms",
    "KitchenAbvGr": "Kitchens",
    "TotRmsAbvGrd": "Total Rooms",
    "Fireplaces": "Fireplaces",
    "GarageYrBlt": "Garage Year Built",
    "GarageCars": "Garage Capacity",
    "GarageArea": "Garage Area",
    "WoodDeckSF": "Wood Deck Area",
    "OpenPorchSF": "Open Porch Area",
    "EnclosedPorch": "Enclosed Porch",
    "3SsnPorch": "Three-Season Porch",
    "ScreenPorch": "Screen Porch",
    "PoolArea": "Pool Area",
    "MiscVal": "Miscellaneous Value",
    "MoSold": "Month Sold",
    "YrSold": "Year Sold",
}


# ============================================================
# OPTIONAL SHAP
# ============================================================

try:

    import shap

    SHAP_AVAILABLE = True

except ImportError:

    shap = None
    SHAP_AVAILABLE = False


@st.cache_resource(show_spinner=False)
def get_shap_explainer():

    if not SHAP_AVAILABLE:
        return None

    try:

        return shap.TreeExplainer(
            xgb_model
        )

    except Exception:

        return None


# ============================================================
# PREMIUM CSS
# ============================================================

render_html(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );


    /* ======================================================
       GLOBAL
       ====================================================== */

    :root {

        --bg: #070a10;
        --bg-soft: #0b1018;
        --panel: #101620;
        --panel-soft: #131a25;

        --border: rgba(255,255,255,0.075);

        --text: #f5f7fb;
        --text-soft: #c6cedb;
        --muted: #8994a7;

        --purple: #7c5cff;
        --purple-light: #9b85ff;

        --teal: #5eead4;
        --green: #69e6a5;

        --danger: #ff8797;

        --radius-lg: 26px;
        --radius-md: 18px;
        --radius-sm: 12px;
    }


    html,
    body,
    [class*="css"] {

        font-family:
            "Inter",
            sans-serif;
    }


    .stApp {

        background:

            radial-gradient(
                circle at 5% -5%,
                rgba(124,92,255,0.18),
                transparent 30%
            ),

            radial-gradient(
                circle at 100% 5%,
                rgba(94,234,212,0.075),
                transparent 24%
            ),

            radial-gradient(
                circle at 50% 100%,
                rgba(124,92,255,0.055),
                transparent 30%
            ),

            var(--bg);

        color: var(--text);
    }


    .block-container {

        max-width: 1450px;

        padding-top: 1.7rem;
        padding-bottom: 5rem;
    }


    #MainMenu {
        visibility: hidden;
    }


    footer {
        visibility: hidden;
    }


    header {
        background: transparent !important;
    }


    /* ======================================================
       TOP BAR
       ====================================================== */

    .topbar {

        display: flex;
        align-items: center;
        justify-content: space-between;

        margin-bottom: 18px;

        padding: 5px 2px;
    }


    .brand {

        display: flex;
        align-items: center;

        gap: 11px;

        font-size: 13px;
        font-weight: 800;

        letter-spacing: 1.8px;

        color: var(--text);
    }


    .brand-icon {

        width: 38px;
        height: 38px;

        display: flex;
        align-items: center;
        justify-content: center;

        border-radius: 12px;

        background:

            linear-gradient(
                135deg,
                rgba(124,92,255,0.24),
                rgba(94,234,212,0.10)
            );

        border:
            1px solid
            rgba(124,92,255,0.28);

        color: var(--purple-light);

        font-size: 18px;
    }


    .model-status {

        display: inline-flex;
        align-items: center;

        gap: 8px;

        padding: 8px 12px;

        border-radius: 999px;

        background:
            rgba(94,234,212,0.055);

        border:
            1px solid
            rgba(94,234,212,0.13);

        color: #9be8d9;

        font-size: 10px;
        font-weight: 700;

        letter-spacing: 1.1px;
    }


    .status-dot {

        width: 7px;
        height: 7px;

        border-radius: 50%;

        background: var(--teal);

        box-shadow:

            0 0 0 4px
            rgba(94,234,212,0.08),

            0 0 16px
            rgba(94,234,212,0.5);
    }


    /* ======================================================
       HERO
       ====================================================== */

    .hero {

        position: relative;
        overflow: hidden;

        padding: 44px;

        border-radius: 30px;

        border:
            1px solid
            var(--border);

        background:

            linear-gradient(
                135deg,
                rgba(124,92,255,0.135),
                rgba(16,22,32,0.92) 50%,
                rgba(10,15,23,0.98)
            );

        box-shadow:
            0 30px 90px
            rgba(0,0,0,0.30);

        margin-bottom: 22px;
    }


    .hero::before {

        content: "";

        position: absolute;

        width: 360px;
        height: 360px;

        right: -150px;
        top: -170px;

        border-radius: 50%;

        background:
            rgba(124,92,255,0.18);

        filter: blur(65px);
    }


    .hero::after {

        content: "";

        position: absolute;

        width: 240px;
        height: 240px;

        left: 45%;
        bottom: -190px;

        border-radius: 50%;

        background:
            rgba(94,234,212,0.06);

        filter: blur(55px);
    }


    .hero-content {

        position: relative;
        z-index: 2;
    }


    .hero-eyebrow {

        color: var(--teal);

        font-size: 11px;
        font-weight: 800;

        letter-spacing: 2.1px;

        text-transform: uppercase;

        margin-bottom: 14px;
    }


    .hero-title {

        font-size:
            clamp(40px, 5.5vw, 70px);

        line-height: 0.98;

        font-weight: 800;

        letter-spacing: -4px;

        margin: 0;
    }


    .hero-gradient {

        background:

            linear-gradient(
                110deg,
                #ffffff,
                #bdb3ff 48%,
                #79ead9
            );

        -webkit-background-clip: text;

        -webkit-text-fill-color:
            transparent;
    }


    .hero-subtitle {

        max-width: 720px;

        color: var(--muted);

        font-size: 15px;

        line-height: 1.75;

        margin-top: 20px;
    }


    .pill-row {

        display: flex;

        flex-wrap: wrap;

        gap: 8px;

        margin-top: 25px;
    }


    .pill {

        display: inline-flex;

        align-items: center;

        padding: 8px 12px;

        border-radius: 999px;

        background:
            rgba(255,255,255,0.045);

        border:
            1px solid
            var(--border);

        color: #cbd2df;

        font-size: 11px;

        font-weight: 650;
    }


    .hero-metrics {

        display: grid;

        grid-template-columns:
            repeat(3, minmax(0, 1fr));

        gap: 10px;

        margin-top: 30px;

        max-width: 900px;
    }


    .hero-metric {

        padding: 18px 19px;

        border-radius: 17px;

        background:
            rgba(255,255,255,0.035);

        border:
            1px solid
            rgba(255,255,255,0.065);

        backdrop-filter:
            blur(18px);
    }


    .hero-metric-label {

        color: var(--muted);

        font-size: 10px;
        font-weight: 700;

        letter-spacing: 1.2px;

        text-transform: uppercase;
    }


    .hero-metric-value {

        margin-top: 7px;

        font-size: 23px;

        font-weight: 800;

        color: var(--text);
    }


    .hero-metric-note {

        margin-top: 4px;

        color: #6f7b8e;

        font-size: 10px;
    }


    /* ======================================================
       INPUT INTRO
       ====================================================== */

    .input-intro {

        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 20px;

        padding: 18px 20px;

        margin-top: 24px;
        margin-bottom: 10px;

        border-radius: 18px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            var(--border);
    }


    .input-intro-title {

        font-size: 14px;

        font-weight: 750;
    }


    .input-intro-description {

        color: var(--muted);

        font-size: 11px;

        margin-top: 4px;
    }


    .input-count {

        white-space: nowrap;

        padding: 8px 11px;

        border-radius: 10px;

        background:
            rgba(124,92,255,0.10);

        border:
            1px solid
            rgba(124,92,255,0.16);

        color: #b9adff;

        font-size: 10px;

        font-weight: 700;
    }


    /* ======================================================
       SECTIONS
       ====================================================== */

    .section-header {

        display: flex;

        align-items: center;

        gap: 13px;

        margin-top: 29px;

        margin-bottom: 12px;
    }


    .section-number {

        width: 40px;
        height: 40px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 12px;

        background:

            linear-gradient(
                135deg,
                rgba(124,92,255,0.16),
                rgba(124,92,255,0.055)
            );

        border:
            1px solid
            rgba(124,92,255,0.18);

        color: #a99bff;

        font-size: 11px;

        font-weight: 800;
    }


    .section-title {

        font-size: 18px;

        font-weight: 750;
    }


    .section-description {

        margin-top: 3px;

        color: var(--muted);

        font-size: 11px;
    }


    /* ======================================================
    EXPANDERS
    ====================================================== */

    div[data-testid="stExpander"] {

        background:
            linear-gradient(
                180deg,
                rgba(255,255,255,0.035),
                rgba(255,255,255,0.018)
            ) !important;

        border:
            1px solid rgba(255,255,255,0.08) !important;

        border-radius: 18px !important;

        margin-bottom: 10px !important;

        overflow: hidden !important;
    }


    div[data-testid="stExpander"] details {

        background: transparent !important;
    }


    div[data-testid="stExpander"] summary {

        color: #f1f5f9 !important;

        background: transparent !important;

        padding: 14px 16px !important;

        font-size: 13px !important;

        font-weight: 700 !important;
    }


    div[data-testid="stExpander"] summary:hover {

        color: #ffffff !important;

        background:
            rgba(124,92,255,0.055) !important;
    }


    /* Expander arrow */
    div[data-testid="stExpander"] summary svg {

        fill: #a99bff !important;

        color: #a99bff !important;
    }

    /* ======================================================
    STREAMLIT INPUTS
    ====================================================== */

    /* Labels */
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label {

        color: #cbd5e1 !important;

        font-size: 11px !important;

        font-weight: 650 !important;
    }


    /* Number input container */
    div[data-testid="stNumberInput"] div[data-baseweb="input"],
    div[data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    div[data-testid="stNumberInput"] div[data-baseweb="base-input"] {

        background: #101620 !important;

        border: 1px solid rgba(255,255,255,0.12) !important;

        border-radius: 11px !important;
    }


    /* Selectbox main container */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {

        background: #101620 !important;

        border: 1px solid rgba(255,255,255,0.09) !important;

        border-radius: 11px !important;

        min-height: 42px;
    }


    /* Selected selectbox text */
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    div[role="button"] {

        color: #f5f7fb !important;
    }


    /* Selected value */
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    span {

        color: #f5f7fb !important;
    }


    /* Placeholder / selected value */
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"]
    input {

        color: #f5f7fb !important;
    }


    /* Hover */
    div[data-testid="stSelectbox"]
    div[data-baseweb="select"] > div:hover {

        border-color: rgba(124,92,255,0.45) !important;
    }


    /* ======================================================
    SELECTBOX DROPDOWN / POPOVER
    ====================================================== */

    /* Dropdown background */
    div[data-baseweb="popover"] {

        background: #111827 !important;

        border: 1px solid rgba(255,255,255,0.10) !important;

        border-radius: 13px !important;

        box-shadow:
            0 20px 60px rgba(0,0,0,0.55) !important;
    }


    /* Dropdown list */
    div[data-baseweb="popover"] ul {

        background: #111827 !important;
    }


    /* Dropdown options */
    div[data-baseweb="popover"]
    li[role="option"] {

        background: #111827 !important;

        color: #e5e7eb !important;

        font-size: 12px !important;

        font-weight: 500 !important;

        padding: 10px 13px !important;
    }


    /* Dropdown option hover */
    div[data-baseweb="popover"]
    li[role="option"]:hover {

        background: rgba(124,92,255,0.16) !important;

        color: #ffffff !important;
    }


    /* Selected dropdown option */
    div[data-baseweb="popover"]
    li[role="option"][aria-selected="true"] {

        background: rgba(124,92,255,0.22) !important;

        color: #ffffff !important;

        font-weight: 700 !important;
    }


    /* Dropdown text and spans */
    div[data-baseweb="popover"]
    li[role="option"] span {

        color: inherit !important;
    }


    /* Number input text and instructions (Fix invisibility) */
    div[data-testid="stNumberInput"] input {

        color: #rgb(49, 51, 63) !important;

        -webkit-text-fill-color: #rgb(49, 51, 63) !important;

        background: transparent !important;

        opacity: 1 !important;

        font-weight: 400 !important;
    }

    div[data-testid="stNumberInput"] div[data-testid="InputInstructions"] > span {

        color: #8994a7 !important;
    }


    /* Number input buttons */
    div[data-testid="stNumberInput"] button {

        color: #aeb8c8 !important;

        background: transparent !important;

        border: none !important;
    }


    div[data-testid="stNumberInput"] button:hover {

        color: #ffffff !important;

        background: rgba(124,92,255,0.12) !important;
    }

    /* ======================================================
       SUBMIT
       ====================================================== */

    .submit-area {

        margin-top: 25px;

        padding: 22px;

        border-radius: 20px;

        background:

            linear-gradient(
                135deg,
                rgba(124,92,255,0.08),
                rgba(94,234,212,0.025)
            );

        border:
            1px solid
            rgba(124,92,255,0.13);
    }


    .submit-title {

        font-size: 15px;

        font-weight: 750;

        margin-bottom: 4px;
    }


    .submit-description {

        color: var(--muted);

        font-size: 11px;

        margin-bottom: 16px;
    }


    div[data-testid="stFormSubmitButton"] button {

        width: 100%;

        min-height: 57px;

        border: 0 !important;

        border-radius:
            14px !important;

        color:
            #ffffff !important;

        background:

            linear-gradient(
                135deg,
                #8063ff,
                #6245df
            ) !important;

        font-size:
            13px !important;

        font-weight:
            800 !important;

        letter-spacing:
            0.8px;

        box-shadow:
            0 14px 35px
            rgba(124,92,255,0.22);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }


    div[data-testid="stFormSubmitButton"]
    button:hover {

        transform:
            translateY(-2px);

        box-shadow:
            0 18px 42px
            rgba(124,92,255,0.34);
    }


    /* ======================================================
       RESULT
       ====================================================== */

    .result-shell {

        position: relative;

        overflow: hidden;

        margin-top: 34px;

        padding: 32px;

        border-radius: 27px;

        border:
            1px solid
            rgba(94,234,212,0.16);

        background:

            radial-gradient(
                circle at 90% 0%,
                rgba(94,234,212,0.09),
                transparent 30%
            ),

            radial-gradient(
                circle at 5% 100%,
                rgba(124,92,255,0.10),
                transparent 30%
            ),

            linear-gradient(
                135deg,
                rgba(16,24,31,0.98),
                rgba(13,18,27,0.98)
            );

        box-shadow:
            0 25px 80px
            rgba(0,0,0,0.28);
    }


    .result-top {

        display: flex;

        align-items: center;

        justify-content: space-between;

        gap: 20px;
    }


    .result-kicker {

        color:
            var(--teal);

        font-size:
            10px;

        font-weight:
            800;

        letter-spacing:
            1.8px;
    }


    .result-heading {

        margin-top: 5px;

        font-size: 20px;

        font-weight: 750;
    }


    .result-ready {

        display: inline-flex;

        align-items: center;

        gap: 8px;

        padding: 8px 11px;

        border-radius: 999px;

        background:
            rgba(94,234,212,0.05);

        border:
            1px solid
            rgba(94,234,212,0.12);

        color:
            #9ae9da;

        font-size:
            10px;

        font-weight:
            700;
    }


    .result-price {

        margin-top: 28px;

        font-size:
            clamp(48px, 8vw, 82px);

        line-height: 0.95;

        font-weight: 850;

        letter-spacing: -4px;

        background:

            linear-gradient(
                100deg,
                #ffffff,
                #c7c0ff 48%,
                #8af0dd
            );

        -webkit-background-clip:
            text;

        -webkit-text-fill-color:
            transparent;
    }


    .result-subtitle {

        margin-top: 12px;

        color:
            var(--muted);

        font-size:
            11px;
    }


    .result-metrics {

        display: grid;

        grid-template-columns:
            repeat(3, minmax(0, 1fr));

        gap: 10px;

        margin-top: 27px;
    }


    .result-metric {

        padding: 17px;

        border-radius: 15px;

        background:
            rgba(255,255,255,0.03);

        border:
            1px solid
            rgba(255,255,255,0.065);
    }


    .result-metric-label {

        color:
            var(--muted);

        font-size:
            9px;

        font-weight:
            700;

        text-transform:
            uppercase;

        letter-spacing:
            1px;
    }


    .result-metric-value {

        margin-top: 6px;

        font-size:
            17px;

        font-weight:
            800;
    }


    /* ======================================================
       CONTENT CARDS
       ====================================================== */

    .content-card {

        margin-top: 15px;

        padding: 22px;

        border-radius: 19px;

        background:

            linear-gradient(
                180deg,
                rgba(255,255,255,0.028),
                rgba(255,255,255,0.016)
            );

        border:
            1px solid
            var(--border);
    }


    .card-kicker {

        color:
            var(--muted);

        font-size:
            9px;

        font-weight:
            800;

        letter-spacing:
            1.4px;

        text-transform:
            uppercase;
    }


    .card-title {

        margin-top: 5px;

        font-size:
            16px;

        font-weight:
            750;
    }


    .card-description {

        color:
            var(--muted);

        font-size:
            11px;

        line-height:
            1.65;

        margin-top: 5px;
    }


    /* ======================================================
       SNAPSHOT
       ====================================================== */

    .snapshot-grid {

        display: grid;

        grid-template-columns:
            repeat(4, minmax(0, 1fr));

        gap: 9px;

        margin-top: 16px;
    }


    .snapshot-item {

        padding: 15px;

        border-radius: 14px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.055);
    }


    .snapshot-label {

        color:
            #778295;

        font-size:
            9px;

        text-transform:
            uppercase;

        letter-spacing:
            0.8px;
    }


    .snapshot-value {

        margin-top: 6px;

        color:
            #edf0f5;

        font-size:
            14px;

        font-weight:
            750;

        overflow-wrap:
            anywhere;
    }


    /* ======================================================
       INSIGHTS
       ====================================================== */

    .insight-grid {

        display: grid;

        grid-template-columns:
            repeat(2, minmax(0, 1fr));

        gap: 9px;

        margin-top: 16px;
    }


    .insight-item {

        padding: 15px;

        border-radius: 14px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.055);
    }


    .insight-name {

        color:
            #e8ecf3;

        font-size:
            11px;

        font-weight:
            700;
    }


    .insight-value {

        margin-top: 6px;

        font-size:
            11px;

        font-weight:
            700;
    }


    .insight-positive {

        color:
            var(--teal);
    }


    .insight-negative {

        color:
            var(--danger);
    }


    .insight-note {

        margin-top: 13px;

        color:
            #687487;

        font-size:
            10px;

        line-height:
            1.6;
    }


    /* ======================================================
       PERFORMANCE
       ====================================================== */

    .performance-grid {

        display: grid;

        grid-template-columns:
            repeat(3, minmax(0, 1fr));

        gap: 9px;

        margin-top: 15px;
    }


    .performance-item {

        padding: 15px;

        border-radius: 14px;

        background:
            rgba(255,255,255,0.025);

        border:
            1px solid
            rgba(255,255,255,0.055);
    }


    .performance-value {

        font-size:
            18px;

        font-weight:
            800;
    }


    .performance-label {

        color:
            #788396;

        font-size:
            9px;

        margin-top: 4px;

        text-transform:
            uppercase;

        letter-spacing:
            0.7px;
    }


    .performance-note {

        color:
            #687487;

        font-size:
            10px;

        line-height:
            1.6;

        margin-top:
            14px;
    }


    /* ======================================================
       DISCLAIMER
       ====================================================== */

    .disclaimer {

        display: flex;

        gap: 14px;

        margin-top: 15px;

        padding: 18px;

        border-radius: 16px;

        background:
            rgba(255,255,255,0.02);

        border:
            1px solid
            var(--border);
    }


    .disclaimer-icon {

        flex: 0 0 auto;

        width: 32px;
        height: 32px;

        display: flex;

        align-items: center;

        justify-content: center;

        border-radius: 10px;

        background:
            rgba(124,92,255,0.09);

        border:
            1px solid
            rgba(124,92,255,0.13);

        color:
            #aa9dff;

        font-size:
            13px;
    }


    .disclaimer-title {

        color:
            #dce1e9;

        font-size:
            11px;

        font-weight:
            750;
    }


    .disclaimer-text {

        color:
            #788396;

        font-size:
            10px;

        line-height:
            1.65;

        margin-top:
            4px;
    }


    /* ======================================================
       RESET
       ====================================================== */

    div.stButton > button {

        border-radius:
            12px !important;

        background:
            rgba(255,255,255,0.035) !important;

        border:
            1px solid
            rgba(255,255,255,0.08) !important;

        color:
            #cbd2df !important;

        font-size:
            11px !important;

        font-weight:
            700 !important;

        min-height:
            43px;
    }


    div.stButton > button:hover {

        border-color:
            rgba(124,92,255,0.30) !important;

        color:
            #ffffff !important;
    }


    /* ======================================================
       ERROR
       ====================================================== */

    .error-card {

        margin-top:
            22px;

        padding:
            20px;

        border-radius:
            16px;

        background:
            rgba(255,90,110,0.045);

        border:
            1px solid
            rgba(255,90,110,0.13);

        color:
            #ffabb7;

        font-size:
            11px;

        line-height:
            1.6;
    }


    /* ======================================================
       FOOTER
       ====================================================== */

    .footer {

        margin-top:
            48px;

        padding-top:
            22px;

        border-top:
            1px solid
            rgba(255,255,255,0.05);

        text-align:
            center;

        color:
            #5f6a7c;

        font-size:
            10px;

        line-height:
            1.7;
    }


    .footer-brand {

        color:
            #8c96a7;

        font-weight:
            750;

        letter-spacing:
            0.8px;
    }


    /* ======================================================
       RESPONSIVE
       ====================================================== */

    @media (max-width: 900px) {

        .hero {
            padding: 30px;
        }

        .hero-metrics {
            grid-template-columns: 1fr;
        }

        .snapshot-grid {
            grid-template-columns:
                repeat(2, minmax(0, 1fr));
        }

        .performance-grid {
            grid-template-columns: 1fr;
        }
    }


    @media (max-width: 650px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .model-status {
            display: none;
        }

        .hero {
            padding: 25px;
            border-radius: 23px;
        }

        .hero-title {
            letter-spacing: -2.5px;
        }

        .hero-subtitle {
            font-size: 13px;
        }

        .result-shell {
            padding: 24px;
        }

        .result-top {
            align-items: flex-start;
        }

        .result-ready {
            display: none;
        }

        .result-price {
            letter-spacing: -2.5px;
        }

        .result-metrics {
            grid-template-columns: 1fr;
        }

        .snapshot-grid {
            grid-template-columns: 1fr;
        }

        .insight-grid {
            grid-template-columns: 1fr;
        }

        .input-intro {
            align-items: flex-start;
            flex-direction: column;
        }
    }

    /* ======================================================
    STREAMLIT TEXT VISIBILITY
    ====================================================== */

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown span {

        color: #d8dee9;
    }


    /* Don't allow Streamlit's default text color
    to destroy custom HTML */
    .stApp div {

        --text-color: #f5f7fb;
    }


    /* Custom HTML elements */
    .hero,
    .hero *,
    .topbar,
    .topbar *,
    .content-card,
    .content-card *,
    .result-shell,
    .result-shell *,
    .section-header,
    .section-header *,
    .input-intro,
    .input-intro *,
    .submit-area,
    .submit-area *,
    .disclaimer,
    .disclaimer *,
    .footer,
    .footer * {

        box-sizing: border-box;
    }
    
    </style>
    """
)


# ============================================================
# INPUT STATE
# ============================================================

input_data = {}
rendered_features = set()


# ============================================================
# FEATURE REGISTRATION
# ============================================================

def register_feature(feature):

    if feature in rendered_features:

        st.error(
            f"Internal UI error: '{feature}' "
            "is rendered more than once."
        )

        st.stop()

    rendered_features.add(feature)


# ============================================================
# ROBUST NUMERIC INPUT
# ============================================================

def numeric_input(
    feature,
    label=None,
    default=None,
    min_value=None,
    max_value=None,
    step=None,
    help_text=None,
):
    """
    Robust Streamlit number_input.

    IMPORTANT:
    Streamlit requires value, min_value,
    max_value and step to use compatible
    numeric types.

    This function automatically converts
    every numeric argument to the same type
    as the default value.
    """

    register_feature(feature)

    if default is None:

        default = DEFAULT_NUMERIC.get(
            feature,
            0
        )

    # --------------------------------------------------------
    # Determine numeric type
    # --------------------------------------------------------

    if isinstance(
        default,
        (float, np.floating)
    ):

        numeric_type = float

    elif isinstance(
        default,
        (Integral, np.integer)
    ):

        numeric_type = int

    else:

        numeric_type = float


    # --------------------------------------------------------
    # Convert default
    # --------------------------------------------------------

    default = numeric_type(
        default
    )


    # --------------------------------------------------------
    # Convert min/max
    # --------------------------------------------------------

    if min_value is not None:

        min_value = numeric_type(
            min_value
        )


    if max_value is not None:

        max_value = numeric_type(
            max_value
        )


    # --------------------------------------------------------
    # Convert step
    # --------------------------------------------------------

    if step is None:

        step = (
            1
            if numeric_type is int
            else 0.1
        )

    else:

        step = numeric_type(
            step
        )


    # --------------------------------------------------------
    # Ensure default is inside range
    # --------------------------------------------------------

    if min_value is not None:

        default = max(
            default,
            min_value
        )


    if max_value is not None:

        default = min(
            default,
            max_value
        )


    kwargs = {

        "label":
            label
            or DISPLAY_NAMES.get(
                feature,
                feature
            ),

        "value":
            default,

        "step":
            step,

        "key":
            f"num_{feature}",
    }


    if min_value is not None:

        kwargs["min_value"] = min_value


    if max_value is not None:

        kwargs["max_value"] = max_value


    if help_text:

        kwargs["help"] = help_text


    input_data[feature] = (
        st.number_input(**kwargs)
    )


# ============================================================
# CATEGORICAL INPUT
# ============================================================

def categorical_input(
    feature,
    label=None,
    help_text=None,
):

    register_feature(feature)

    options = category_options.get(
        feature,
        []
    )


    if not options:

        options = [
            str(
                DEFAULT_CATEGORICAL.get(
                    feature,
                    "None"
                )
            )
        ]


    options = [
        str(option)
        for option in options
    ]


    default = str(
        DEFAULT_CATEGORICAL.get(
            feature,
            options[0]
        )
    )


    if default in options:

        index = options.index(
            default
        )

    else:

        index = 0


    kwargs = {

        "label":
            label
            or DISPLAY_NAMES.get(
                feature,
                feature
            ),

        "options":
            options,

        "index":
            index,

        "key":
            f"cat_{feature}",
    }


    if help_text:

        kwargs["help"] = help_text


    input_data[feature] = (
        st.selectbox(**kwargs)
    )


# ============================================================
# SECTION HEADER
# ============================================================

def section_header(
    number,
    title,
    description,
):

    render_html(
        f"""
        <div class="section-header">

            <div class="section-number">
                {escape(str(number))}
            </div>

            <div>

                <div class="section-title">
                    {escape(str(title))}
                </div>

                <div class="section-description">
                    {escape(str(description))}
                </div>

            </div>

        </div>
        """
    )


# ============================================================
# MONEY FORMATTERS
# ============================================================

def money(value):

    return (
        f"${float(value):,.0f}"
    )


def compact_money(value):

    value = float(value)

    if abs(value) >= 1_000_000:

        return (
            f"${value / 1_000_000:.2f}M"
        )

    if abs(value) >= 1_000:

        return (
            f"${value / 1_000:.1f}K"
        )

    return money(value)


# ============================================================
# FEATURE NAME FORMATTER
# ============================================================

def pretty_feature_name(
    feature_name
):

    feature_name = str(
        feature_name
    )


    # Numerical feature

    if feature_name.startswith(
        "num__"
    ):

        raw_name = (
            feature_name[5:]
        )

        return DISPLAY_NAMES.get(
            raw_name,
            raw_name.replace(
                "_",
                " "
            )
        )


    # Categorical feature

    if feature_name.startswith(
        "cat__"
    ):

        encoded = (
            feature_name[5:]
        )


        for feature in categorical_features:

            prefix = (
                f"{feature}_"
            )


            if encoded.startswith(
                prefix
            ):

                category = encoded[
                    len(prefix):
                ]


                readable = (
                    DISPLAY_NAMES.get(
                        feature,
                        feature.replace(
                            "_",
                            " "
                        )
                    )
                )


                return (
                    f"{readable} · "
                    f"{category}"
                )


        return encoded.replace(
            "_",
            " "
        )


    return feature_name.replace(
        "_",
        " "
    )


# ============================================================
# AI INSIGHTS
# ============================================================

def get_prediction_insights(
    input_df
):

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    explainer = (
        get_shap_explainer()
    )


    if explainer is not None:

        try:

            transformed = (
                preprocessor.transform(
                    input_df
                )
            )


            shap_output = (
                explainer.shap_values(
                    transformed
                )
            )


            # Newer SHAP versions may
            # return an Explanation object.

            if hasattr(
                shap_output,
                "values"
            ):

                shap_values = (
                    shap_output.values
                )

            else:

                shap_values = (
                    shap_output
                )


            # Some versions return
            # a list of arrays.

            if isinstance(
                shap_values,
                list
            ):

                shap_values = (
                    shap_values[0]
                )


            shap_values = np.asarray(
                shap_values
            )


            if shap_values.ndim == 2:

                values = (
                    shap_values[0]
                )

            else:

                values = (
                    shap_values
                )


            feature_names = (
                preprocessor
                .get_feature_names_out()
            )


            if len(values) != len(
                feature_names
            ):

                raise ValueError(
                    "SHAP output does not "
                    "match transformed features."
                )


            top_indices = (
                np.argsort(
                    np.abs(values)
                )[::-1][:6]
            )


            insights = []


            for index in top_indices:

                contribution = float(
                    values[index]
                )


                if abs(
                    contribution
                ) < 1e-8:

                    continue


                insights.append(
                    (
                        pretty_feature_name(
                            feature_names[
                                index
                            ]
                        ),

                        contribution,
                    )
                )


            if insights:

                return {

                    "type":
                        "shap",

                    "items":
                        insights,
                }


        except Exception:

            pass


    # --------------------------------------------------------
    # GLOBAL XGBOOST IMPORTANCE FALLBACK
    # --------------------------------------------------------

    try:

        feature_names = (
            preprocessor
            .get_feature_names_out()
        )


        importances = np.asarray(
            xgb_model.feature_importances_
        )


        if len(importances) != len(
            feature_names
        ):

            return None


        top_indices = (
            np.argsort(
                importances
            )[::-1][:6]
        )


        insights = []


        for index in top_indices:

            importance = float(
                importances[index]
            )


            insights.append(
                (
                    pretty_feature_name(
                        feature_names[
                            index
                        ]
                    ),

                    importance,
                )
            )


        return {

            "type":
                "importance",

            "items":
                insights,
        }


    except Exception:

        return None


# ============================================================
# CLEAR FORM
# ============================================================

def clear_form_state():

    keys_to_delete = [

        key
        for key in list(
            st.session_state.keys()
        )

        if (
            key.startswith("num_")
            or key.startswith("cat_")
        )
    ]


    for key in keys_to_delete:

        del st.session_state[key]


    st.session_state.pop(
        "prediction",
        None
    )


# ============================================================
# TOP BAR
# ============================================================

render_html(
    """
    <div class="topbar">

        <div class="brand">

            <div class="brand-icon">
                ◇
            </div>

            AMESVALUE AI

        </div>

        <div class="model-status">

            <span class="status-dot"></span>

            MODEL READY

        </div>

    </div>
    """
)


# ============================================================
# HERO
# ============================================================

render_html(
    f"""
    <div class="hero">

        <div class="hero-content">

            <div class="hero-eyebrow">
                AI PROPERTY VALUATION ENGINE
            </div>

            <div class="hero-title">

                Intelligent Property<br>

                <span class="hero-gradient">
                    Valuation.
                </span>

            </div>

            <div class="hero-subtitle">

                Estimate property value using a trained
                XGBoost regression pipeline built on the
                Ames Housing dataset. Configure the property,
                run the model, and inspect the signals behind
                the prediction.

            </div>

            <div class="pill-row">

                <span class="pill">
                    XGBoost
                </span>

                <span class="pill">
                    Log Target
                </span>

                <span class="pill">
                    {len(expected_features)} Features
                </span>

                <span class="pill">
                    Ames Housing
                </span>

                <span class="pill">
                    Regression
                </span>

            </div>

            <div class="hero-metrics">

                <div class="hero-metric">

                    <div class="hero-metric-label">
                        Holdout R²
                    </div>

                    <div class="hero-metric-value">
                        {HOLDOUT_R2:.2%}
                    </div>

                    <div class="hero-metric-note">
                        Held-out test performance
                    </div>

                </div>

                <div class="hero-metric">

                    <div class="hero-metric-label">
                        5-Fold CV R²
                    </div>

                    <div class="hero-metric-value">
                        {CV_R2:.2%}
                    </div>

                    <div class="hero-metric-note">
                        Cross-validation mean
                    </div>

                </div>

                <div class="hero-metric">

                    <div class="hero-metric-label">
                        Validation MAE
                    </div>

                    <div class="hero-metric-value">
                        {compact_money(VALIDATION_MAE)}
                    </div>

                    <div class="hero-metric-note">
                        Typical absolute error
                    </div>

                </div>

            </div>

        </div>

    </div>
    """
)


# ============================================================
# INPUT INTRO
# ============================================================

render_html(
    f"""
    <div class="input-intro">

        <div>

            <div class="input-intro-title">
                Configure the property
            </div>

            <div class="input-intro-description">
                Fields are pre-populated with representative
                Ames Housing values. Replace them with the
                property's actual characteristics.
            </div>

        </div>

        <div class="input-count">
            {len(expected_features)} MODEL INPUTS
        </div>

    </div>
    """
)


# ============================================================
# PROPERTY FORM
# ============================================================

with st.form(
    "property_valuation_form"
):

    # ========================================================
    # 01 — PROPERTY & LOCATION
    # ========================================================

    section_header(
        "01",
        "Property & Location",
        "Identity, zoning, neighborhood and building profile.",
    )


    with st.expander(
        "Property identity & neighborhood",
        expanded=True,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            numeric_input(
                "Id",
                "Property ID",
                min_value=1,
                help_text=(
                    "Reference identifier retained "
                    "for compatibility with the "
                    "trained model."
                ),
            )

            categorical_input(
                "MSZoning",
                "Zoning",
            )

            categorical_input(
                "Neighborhood",
                "Neighborhood",
            )


        with col2:

            numeric_input(
                "MSSubClass",
                "Building Class",
            )

            categorical_input(
                "BldgType",
                "Building Type",
            )

            categorical_input(
                "HouseStyle",
                "House Style",
            )


        with col3:

            categorical_input(
                "LotConfig",
                "Lot Configuration",
            )

            categorical_input(
                "LandContour",
                "Land Contour",
            )

            categorical_input(
                "LandSlope",
                "Land Slope",
            )


    # ========================================================
    # 02 — LAND & STRUCTURE
    # ========================================================

    section_header(
        "02",
        "Land & Structure",
        "Lot dimensions, construction history and structural characteristics.",
    )


    with st.expander(
        "Land & construction",
        expanded=True,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            numeric_input(
                "LotFrontage",
                "Lot Frontage",
                min_value=0.0,
            )

            numeric_input(
                "LotArea",
                "Lot Area",
                min_value=0,
            )

            categorical_input(
                "LotShape",
                "Lot Shape",
            )

            categorical_input(
                "Street",
                "Street",
            )


        with col2:

            categorical_input(
                "Utilities",
                "Utilities",
            )

            categorical_input(
                "Condition1",
                "Nearby Condition",
            )

            categorical_input(
                "Condition2",
                "Secondary Condition",
            )

            categorical_input(
                "Foundation",
                "Foundation",
            )


        with col3:

            numeric_input(
                "YearBuilt",
                "Year Built",
                min_value=1800,
                max_value=2010,
            )

            numeric_input(
                "YearRemodAdd",
                "Remodel Year",
                min_value=1800,
                max_value=2010,
            )

            categorical_input(
                "PavedDrive",
                "Paved Drive",
            )

            categorical_input(
                "Alley",
                "Alley",
            )


    # ========================================================
    # 03 — QUALITY
    # ========================================================

    section_header(
        "03",
        "Quality & Condition",
        "Overall construction quality, systems and property condition.",
    )


    with st.expander(
        "Quality & condition",
        expanded=True,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            numeric_input(
                "OverallQual",
                "Overall Quality",
                min_value=1,
                max_value=10,
            )

            numeric_input(
                "OverallCond",
                "Overall Condition",
                min_value=1,
                max_value=9,
            )

            categorical_input(
                "ExterQual",
                "Exterior Quality",
            )

            categorical_input(
                "ExterCond",
                "Exterior Condition",
            )


        with col2:

            categorical_input(
                "HeatingQC",
                "Heating Quality",
            )

            categorical_input(
                "KitchenQual",
                "Kitchen Quality",
            )

            categorical_input(
                "Functional",
                "Home Functionality",
            )

            categorical_input(
                "CentralAir",
                "Central Air",
            )


        with col3:

            categorical_input(
                "Electrical",
                "Electrical System",
            )

            categorical_input(
                "Heating",
                "Heating System",
            )

            categorical_input(
                "RoofStyle",
                "Roof Style",
            )

            categorical_input(
                "RoofMatl",
                "Roof Material",
            )


    # ========================================================
    # 04 — EXTERIOR
    # ========================================================

    section_header(
        "04",
        "Exterior & Appearance",
        "Exterior materials, masonry and transaction characteristics.",
    )


    with st.expander(
        "Exterior materials",
        expanded=False,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            categorical_input(
                "Exterior1st",
                "Primary Exterior",
            )

            categorical_input(
                "Exterior2nd",
                "Secondary Exterior",
            )

            categorical_input(
                "MasVnrType",
                "Masonry Type",
            )


        with col2:

            numeric_input(
                "MasVnrArea",
                "Masonry Area",
                min_value=0.0,
            )

            categorical_input(
                "SaleType",
                "Sale Type",
            )


        with col3:

            categorical_input(
                "SaleCondition",
                "Sale Condition",
            )


    # ========================================================
    # 05 — BASEMENT
    # ========================================================

    section_header(
        "05",
        "Basement",
        "Basement quality, finish, exposure and usable area.",
    )


    with st.expander(
        "Basement configuration",
        expanded=False,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            categorical_input(
                "BsmtQual",
                "Basement Quality",
            )

            categorical_input(
                "BsmtCond",
                "Basement Condition",
            )

            categorical_input(
                "BsmtExposure",
                "Basement Exposure",
            )

            categorical_input(
                "BsmtFinType1",
                "Primary Finish",
            )


        with col2:

            categorical_input(
                "BsmtFinType2",
                "Secondary Finish",
            )

            numeric_input(
                "BsmtFinSF1",
                "Finished Area 1",
                min_value=0.0,
            )

            numeric_input(
                "BsmtFinSF2",
                "Finished Area 2",
                min_value=0.0,
            )

            numeric_input(
                "BsmtUnfSF",
                "Unfinished Area",
                min_value=0.0,
            )


        with col3:

            numeric_input(
                "TotalBsmtSF",
                "Total Basement Area",
                min_value=0.0,
            )

            numeric_input(
                "BsmtFullBath",
                "Basement Full Bath",
                min_value=0,
            )

            numeric_input(
                "BsmtHalfBath",
                "Basement Half Bath",
                min_value=0,
            )


    # ========================================================
    # 06 — LIVING
    # ========================================================

    section_header(
        "06",
        "Living Space & Layout",
        "Living area, floors, rooms, bedrooms and bathrooms.",
    )


    with st.expander(
        "Living space",
        expanded=False,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            numeric_input(
                "GrLivArea",
                "Above-Ground Living Area",
                min_value=0,
            )

            numeric_input(
                "1stFlrSF",
                "First Floor Area",
                min_value=0,
            )

            numeric_input(
                "2ndFlrSF",
                "Second Floor Area",
                min_value=0,
            )


        with col2:

            numeric_input(
                "LowQualFinSF",
                "Low Quality Finished Area",
                min_value=0,
            )

            numeric_input(
                "TotRmsAbvGrd",
                "Total Rooms",
                min_value=0,
            )

            numeric_input(
                "BedroomAbvGr",
                "Bedrooms",
                min_value=0,
            )


        with col3:

            numeric_input(
                "FullBath",
                "Full Bathrooms",
                min_value=0,
            )

            numeric_input(
                "HalfBath",
                "Half Bathrooms",
                min_value=0,
            )

            numeric_input(
                "KitchenAbvGr",
                "Kitchens",
                min_value=0,
            )


    # ========================================================
    # 07 — GARAGE
    # ========================================================

    section_header(
        "07",
        "Garage",
        "Garage capacity, construction year, finish and condition.",
    )


    with st.expander(
        "Garage configuration",
        expanded=False,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            categorical_input(
                "GarageType",
                "Garage Type",
            )

            categorical_input(
                "GarageFinish",
                "Garage Finish",
            )


        with col2:

            numeric_input(
                "GarageYrBlt",
                "Garage Year Built",
                min_value=1800,
                max_value=2010,
            )

            numeric_input(
                "GarageCars",
                "Garage Capacity",
                min_value=0,
            )


        with col3:

            numeric_input(
                "GarageArea",
                "Garage Area",
                min_value=0.0,
            )

            categorical_input(
                "GarageQual",
                "Garage Quality",
            )

            categorical_input(
                "GarageCond",
                "Garage Condition",
            )


    # ========================================================
    # 08 — OUTDOOR
    # ========================================================

    section_header(
        "08",
        "Outdoor & Amenities",
        "Fireplaces, decks, porches, pool and additional amenities.",
    )


    with st.expander(
        "Outdoor features & amenities",
        expanded=False,
    ):

        col1, col2, col3 = (
            st.columns(3)
        )


        with col1:

            numeric_input(
                "Fireplaces",
                "Fireplaces",
                min_value=0,
            )

            categorical_input(
                "FireplaceQu",
                "Fireplace Quality",
            )

            numeric_input(
                "WoodDeckSF",
                "Wood Deck Area",
                min_value=0,
            )

            numeric_input(
                "OpenPorchSF",
                "Open Porch Area",
                min_value=0,
            )


        with col2:

            numeric_input(
                "EnclosedPorch",
                "Enclosed Porch",
                min_value=0,
            )

            numeric_input(
                "3SsnPorch",
                "Three-Season Porch",
                min_value=0,
            )

            numeric_input(
                "ScreenPorch",
                "Screen Porch",
                min_value=0,
            )


        with col3:

            numeric_input(
                "PoolArea",
                "Pool Area",
                min_value=0,
            )

            categorical_input(
                "PoolQC",
                "Pool Quality",
            )

            categorical_input(
                "Fence",
                "Fence",
            )

            categorical_input(
                "MiscFeature",
                "Miscellaneous Feature",
            )

            numeric_input(
                "MiscVal",
                "Miscellaneous Value",
                min_value=0,
            )


    # ========================================================
    # 09 — SALE
    # ========================================================

    section_header(
        "09",
        "Sale Information",
        "Transaction timing retained from the original training schema.",
    )


    with st.expander(
        "Sale timing",
        expanded=False,
    ):

        col1, col2 = (
            st.columns(2)
        )


        with col1:

            numeric_input(
                "MoSold",
                "Month Sold",
                min_value=1,
                max_value=12,
            )


        with col2:

            numeric_input(
                "YrSold",
                "Year Sold",
                min_value=2006,
                max_value=2010,
            )


    # ========================================================
    # FEATURE COVERAGE SAFETY
    # ========================================================

    missing_ui_features = [

        feature
        for feature in expected_features
        if feature not in input_data
    ]


    if missing_ui_features:

        for feature in missing_ui_features:

            if feature in DEFAULT_NUMERIC:

                input_data[feature] = (
                    DEFAULT_NUMERIC[feature]
                )

            elif feature in DEFAULT_CATEGORICAL:

                input_data[feature] = (
                    DEFAULT_CATEGORICAL[feature]
                )

            else:

                raise ValueError(
                    f"No default value exists for "
                    f"model feature '{feature}'."
                )


    # ========================================================
    # SUBMIT AREA
    # ========================================================

    render_html(
        """
        <div class="submit-area">

            <div class="submit-title">
                Ready to estimate?
            </div>

            <div class="submit-description">
                Your property profile will pass through
                the trained preprocessing pipeline and
                XGBoost valuation model.
            </div>

        </div>
        """
    )


    submitted = (
        st.form_submit_button(
            "GENERATE PROPERTY VALUATION"
        )
    )


# ============================================================
# PREDICTION
# ============================================================

if submitted:

    st.session_state.pop(
        "prediction",
        None
    )


    try:

        # ----------------------------------------------------
        # Validate UI coverage
        # ----------------------------------------------------

        missing_features = [

            feature
            for feature in expected_features
            if feature not in input_data
        ]


        if missing_features:

            raise ValueError(
                "Missing model features: "
                + ", ".join(
                    missing_features
                )
            )


        # ----------------------------------------------------
        # Build input DataFrame
        # ----------------------------------------------------

        input_df = pd.DataFrame(
            [input_data]
        )


        # Exact training order

        input_df = input_df.reindex(
            columns=expected_features
        )


        # ----------------------------------------------------
        # Validate schema
        # ----------------------------------------------------

        if list(
            input_df.columns
        ) != expected_features:

            raise ValueError(
                "Input feature order does not "
                "match the trained model."
            )


        if input_df.shape != (
            1,
            len(expected_features)
        ):

            raise ValueError(
                "Input shape does not match "
                "the trained model."
            )


        # ----------------------------------------------------
        # Validate missing values
        # ----------------------------------------------------

        if input_df.isnull().any().any():

            missing = (
                input_df.columns[
                    input_df.isnull().any()
                ].tolist()
            )


            raise ValueError(
                "Missing values detected in: "
                + ", ".join(missing)
            )


        # ----------------------------------------------------
        # Prediction
        # ----------------------------------------------------

        predicted_log_price = (
            model.predict(
                input_df
            )[0]
        )


        predicted_log_price = float(
            predicted_log_price
        )


        if not np.isfinite(
            predicted_log_price
        ):

            raise ValueError(
                "The model returned "
                "an invalid prediction."
            )


        # ----------------------------------------------------
        # Convert log1p prediction
        # back to original SalePrice
        # ----------------------------------------------------

        predicted_price = float(
            np.expm1(
                predicted_log_price
            )
        )


        if not np.isfinite(
            predicted_price
        ):

            raise ValueError(
                "The converted prediction "
                "is invalid."
            )


        predicted_price = max(
            predicted_price,
            0.0
        )


        # ----------------------------------------------------
        # Save prediction
        # ----------------------------------------------------

        st.session_state[
            "prediction"
        ] = {

            "price":
                predicted_price,

            "input_data":
                dict(input_data),

            "input_df":
                input_df.copy(),
        }


    except Exception as exc:

        render_html(
            f"""
            <div class="error-card">

                <strong>
                    Prediction could not be generated.
                </strong>

                <br><br>

                Please verify the property inputs
                and try again.

                <br><br>

                <code>
                    {escape(str(exc))}
                </code>

            </div>
            """
        )


# ============================================================
# RESULT
# ============================================================

prediction = (
    st.session_state.get(
        "prediction"
    )
)


if prediction:

    predicted_price = (
        prediction["price"]
    )

    result_data = (
        prediction["input_data"]
    )

    result_df = (
        prediction["input_df"]
    )


    # ========================================================
    # MAIN RESULT
    # ========================================================

    render_html(
        f"""
        <div class="result-shell">

            <div class="result-top">

                <div>

                    <div class="result-kicker">
                        AI VALUATION ENGINE
                    </div>

                    <div class="result-heading">
                        Estimated Property Value
                    </div>

                </div>

                <div class="result-ready">

                    <span class="status-dot"></span>

                    Prediction ready

                </div>

            </div>


            <div class="result-price">
                {money(predicted_price)}
            </div>


            <div class="result-subtitle">
                Estimated SalePrice generated by the trained
                XGBoost regression pipeline after converting
                the log1p target prediction back to its original
                dollar scale.
            </div>


            <div class="result-metrics">

                <div class="result-metric">

                    <div class="result-metric-label">
                        Model
                    </div>

                    <div class="result-metric-value">
                        XGBoost
                    </div>

                </div>


                <div class="result-metric">

                    <div class="result-metric-label">
                        Feature Inputs
                    </div>

                    <div class="result-metric-value">
                        {len(expected_features)}
                    </div>

                </div>


                <div class="result-metric">

                    <div class="result-metric-label">
                        Target
                    </div>

                    <div class="result-metric-value">
                        log1p(SalePrice)
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # PROPERTY SNAPSHOT
    # ========================================================

    area = float(
        result_data.get(
            "GrLivArea",
            0
        )
    )


    bedrooms = int(
        result_data.get(
            "BedroomAbvGr",
            0
        )
    )


    full_baths = float(
        result_data.get(
            "FullBath",
            0
        )
    )


    half_baths = float(
        result_data.get(
            "HalfBath",
            0
        )
    )


    total_baths = (
        full_baths
        +
        (half_baths * 0.5)
    )


    quality = result_data.get(
        "OverallQual",
        "-"
    )


    year_built = result_data.get(
        "YearBuilt",
        "-"
    )


    neighborhood = result_data.get(
        "Neighborhood",
        "-"
    )


    garage = result_data.get(
        "GarageCars",
        0
    )


    basement = float(
        result_data.get(
            "TotalBsmtSF",
            0
        )
    )


    render_html(
        f"""
        <div class="content-card">

            <div class="card-kicker">
                PROPERTY PROFILE
            </div>

            <div class="card-title">
                Valuation snapshot
            </div>

            <div class="card-description">
                Key characteristics supplied to the model
                for this valuation.
            </div>


            <div class="snapshot-grid">

                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Neighborhood
                    </div>

                    <div class="snapshot-value">
                        {escape(str(neighborhood))}
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Living Area
                    </div>

                    <div class="snapshot-value">
                        {area:,.0f} sq ft
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Bedrooms
                    </div>

                    <div class="snapshot-value">
                        {bedrooms}
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Bathrooms
                    </div>

                    <div class="snapshot-value">
                        {total_baths:g}
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Overall Quality
                    </div>

                    <div class="snapshot-value">
                        {escape(str(quality))} / 10
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Year Built
                    </div>

                    <div class="snapshot-value">
                        {escape(str(year_built))}
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Garage
                    </div>

                    <div class="snapshot-value">
                        {escape(str(garage))} cars
                    </div>

                </div>


                <div class="snapshot-item">

                    <div class="snapshot-label">
                        Basement
                    </div>

                    <div class="snapshot-value">
                        {basement:,.0f} sq ft
                    </div>

                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # AI MODEL INSIGHTS
    # ========================================================

    insights = (
        get_prediction_insights(
            result_df
        )
    )


    if insights:

        if insights["type"] == "shap":

            kicker = (
                "PROPERTY-SPECIFIC AI SIGNALS"
            )

            title = (
                "What influenced this prediction?"
            )

            description = (
                "SHAP identifies which encoded model "
                "features pushed this property's prediction "
                "higher or lower. These are model contributions, "
                "not causal explanations."
            )

        else:

            kicker = (
                "GLOBAL MODEL SIGNALS"
            )

            title = (
                "What the model learned"
            )

            description = (
                "Property-specific SHAP analysis was unavailable "
                "in this environment, so the application is "
                "showing the strongest global XGBoost feature "
                "importance signals."
            )


        cards = ""


        for name, value in insights["items"]:

            if insights["type"] == "shap":

                if value >= 0:

                    class_name = (
                        "insight-positive"
                    )

                    direction = (
                        "+ Higher model output"
                    )

                else:

                    class_name = (
                        "insight-negative"
                    )

                    direction = (
                        "− Lower model output"
                    )

            else:

                class_name = (
                    "insight-positive"
                )

                direction = (
                    f"{value:.1%} global importance"
                )


            cards += dedent(
                f"""
                <div class="insight-item">

                    <div class="insight-name">
                        {escape(str(name))}
                    </div>

                    <div class="insight-value {class_name}">
                        {escape(direction)}
                    </div>

                </div>
                """
            )


        if insights["type"] == "shap":

            note = (
                "Positive SHAP contributions increase the "
                "model's log-price output; negative contributions "
                "decrease it. SHAP contribution values are model "
                "output units and are not dollar amounts."
            )

        else:

            note = (
                "Global feature importance describes how strongly "
                "features were used by the trained XGBoost model. "
                "It does not describe this individual property's "
                "prediction."
            )


        render_html(
            f"""
            <div class="content-card">

                <div class="card-kicker">
                    {escape(kicker)}
                </div>

                <div class="card-title">
                    {escape(title)}
                </div>

                <div class="card-description">
                    {escape(description)}
                </div>

                <div class="insight-grid">
                    {cards}
                </div>

                <div class="insight-note">
                    {escape(note)}
                </div>

            </div>
            """
        )


    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    render_html(
        f"""
        <div class="content-card">

            <div class="card-kicker">
                MODEL PERFORMANCE
            </div>

            <div class="card-title">
                How the valuation engine was evaluated
            </div>

            <div class="card-description">
                These metrics describe model performance during
                evaluation. They are not a confidence score for
                this individual property.
            </div>


            <div class="performance-grid">

                <div class="performance-item">

                    <div class="performance-value">
                        {HOLDOUT_R2:.2%}
                    </div>

                    <div class="performance-label">
                        Holdout R²
                    </div>

                </div>


                <div class="performance-item">

                    <div class="performance-value">
                        {CV_R2:.2%}
                    </div>

                    <div class="performance-label">
                        5-Fold CV R²
                    </div>

                </div>


                <div class="performance-item">

                    <div class="performance-value">
                        {money(VALIDATION_MAE)}
                    </div>

                    <div class="performance-label">
                        Validation MAE
                    </div>

                </div>

            </div>


            <div class="performance-note">
                The final model was selected after comparing
                raw-target and log-target XGBoost approaches.
                The log-target baseline produced the strongest
                combination of holdout performance and
                cross-validation stability in this project.
            </div>

        </div>
        """
    )


    # ========================================================
    # DISCLAIMER
    # ========================================================

    render_html(
        """
        <div class="disclaimer">

            <div class="disclaimer-icon">
                i
            </div>

            <div>

                <div class="disclaimer-title">
                    Important model context
                </div>

                <div class="disclaimer-text">
                    This is a machine-learning estimate, not a
                    professional appraisal, guaranteed market
                    price or financial recommendation. The model
                    was trained on the Ames Housing dataset, so
                    its output is expressed in US dollars and
                    reflects patterns represented in that
                    historical dataset. It should not be
                    interpreted as a live real-estate market
                    valuation.
                </div>

            </div>

        </div>
        """
    )


    # ========================================================
    # NEW VALUATION
    # ========================================================

    render_html(
        "<div style='height:12px'></div>"
    )


    reset_col, spacer = (
        st.columns([1, 4])
    )


    with reset_col:

        if st.button(
            "START NEW VALUATION",
            key="new_valuation",
        ):

            clear_form_state()

            st.rerun()


# ============================================================
# FOOTER
# ============================================================

render_html(
    f"""
    <div class="footer">

        <div class="footer-brand">
            {escape(APP_NAME)}
        </div>

        <div>
            Machine Learning Property Valuation ·
            {escape(MODEL_NAME)} Regression ·
            {escape(DATASET_NAME)}
        </div>

        <div style="margin-top:5px;">
            v{escape(APP_VERSION)} ·
            {len(expected_features)} model features ·
            {escape(TARGET_TRANSFORMATION)}
        </div>

    </div>
    """
)