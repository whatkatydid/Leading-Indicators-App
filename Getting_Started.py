import numpy as np
import pandas as pd
import streamlit as st
import pickle

from lightgbm import LGBMRegressor, plot_importance, plot_tree as plot_tree_lgbm, create_tree_digraph
from custom_pdp import partial_dependence

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
from pdpbox import pdp
from PIL import Image

from streamlit_extras.app_logo import add_logo

import os
path = os.getcwd()
os.chdir(os.path.join(path))


st.set_page_config(
    page_title="Leading Indicators Console",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded"
)


# Custom CSS for sidebar
st.markdown("""
  <style>  
      .css-17lntkn {
        font-weight: normal;
        font-size: 16px;
      }
      
      .css-pkbazv {
        font-weight: bold;
        font-size: 16px;
        color: red;
      }
  </style>""", unsafe_allow_html=True)


# Add sidebar logo
add_logo("Assets/logo.png", height=225)


# import sklearn; sklearn.__version__

hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)



for k, v in st.session_state.items():
  if k not in st.session_state:
      st.session_state[k] = v

import streamlit
# st.write(streamlit.__version__)

st.write("# :red[What is Leading Indicators?]")

st.write(":red[Leading Indicators] is a customized machine learning solution that quantifies the effect that Media KPIs have on measured Business KPIs (e.g., Sales, New Accounts, Brand Health Lift, Traffic).  \n  \n  Leading Indicators modeling uses :red[Gradient Boosting Regression Trees (GBRT)] to guide in-flight optimization of digital campaigns to gain insights and drive higher performance uplifts of future campaigns.")

col1, col2, col3 = st.columns(3)
with col2:
  image = Image.open('Assets/gbrt.png')
  st.image(image, width=150, use_column_width='never')


st.write("")
st.write("**Select a model to get started!**")

st.markdown(
    """
    <style>
    [data-baseweb="select"] {
        margin-top: -50px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# @st.cache_data
def upload_model():

  if model_select != "":

    dependent_var = model_dict[model.index(model_select)]["dependent"]
    loaded_model = pickle.load(open(model_dict[model.index(model_select)]["filename"], "rb"))

    X = pd.read_pickle(model_dict[model.index(model_select)]["data"])
    data = pd.read_excel(model_dict[model.index(model_select)]["output"], sheet_name="Modeled Data")
    rf_results = pd.read_excel(model_dict[model.index(model_select)]["output"], sheet_name="Relative Importance")
    feature_variables = loaded_model.steps[1][1].booster_.feature_name()


    st.session_state["model_select"] = model_select
    st.session_state["dependent_var"] = dependent_var
    st.session_state["loaded_model"] = loaded_model
    st.session_state["X"] = X
    st.session_state["data"] = data
    st.session_state["rf_results"] = rf_results
    st.session_state["feature_variables"] = feature_variables

    return [loaded_model, X, data, rf_results, feature_variables]


# Set up model upload specifications, indexed by model
model = ["American Express Social Consideration", "American Express Linear Consideration"]
dependent = ["Consideration_MarketPlatform", "combined_consideration_mc1"]
filename = ["Models/Amex Social/final_model_1221_Platforms.pkl", "Models/Amex Linear/final_model_Linear_TV_053023.pkl"]
data = ["Models/Amex Social/final_df_1221_Platforms.pkl", "Models/Amex Linear/final_model_Linear_TV_data.pkl"]
output = ["Models/Amex Social/AMEX_LIC_Summary_Social_Platforms_1221.xlsx", "Models/Amex Linear/AMEX_LIC_Summary_Global_05.31.23.xlsx"]


model_dict = [{'model': model, 'dependent': dependent, 'filename': filename, 'data': data, 'output': output} for model, dependent, filename, data, output in zip(model, dependent, filename, data, output)]


# st.write(st.session_state["model_select"])
# st.write(model.index(st.session_state["model_select"]))


if "model_select" not in st.session_state:
  st.session_state["model_select"] = "American Express Social Consideration"

model_select = st.selectbox(label="", key='model', options=model, 
                            index=model.index(st.session_state["model_select"]),
                            on_change=upload_model
                            )

# st.write(model_select)
# st.write(model_dict[model.index(model_select)]["output"])

loaded_model, X, data, rf_results, feature_variables = upload_model()

if "dependent_var" not in st.session_state:
  st.session_state["dependent_var"] = dependent_var
if "loaded_model" not in st.session_state:
  st.session_state["loaded_model"] = loaded_model
if "X" not in st.session_state:
  st.session_state["X"] = X
if "data" not in st.session_state:
  st.session_state["data"] = data
if "feature_variables" not in st.session_state:
  st.session_state["feature_variables"] = feature_variables
if "rf_results" not in st.session_state:
  st.session_state["rf_results"] = rf_results





# filename = "../../../Social/Modeling/Saved Models/final_model_1221_Platforms.pkl"
# data = "../../../Social/Modeling/Saved Models/final_df_1221_Platforms.pkl"
# results = "../../../Social/Modeling/Results/Global/AMEX_LIC_Summary_Social_Platforms_1221.xlsx"

# loaded_model = pickle.load(open(filename, "rb"))
# X = pd.read_pickle(data)
# rf_results = pd.read_excel(results, sheet_name="Relative Importance")
# data = pd.read_excel(results, sheet_name="Modeled Data")

# feature_variables = loaded_model.steps[1][1].booster_.feature_name()





# filename = "../../../Social/Modeling/Saved Models/final_model_1221_Platforms.pkl"
# X = pd.read_pickle("../../../Social/Modeling/Saved Models/final_df_1221_Platforms.pkl")

# loaded_model = pickle.load(open(filename, "rb"))


# if st.checkbox("Show raw data"):
#     st.subheader("Raw data")
#     st.write(X)

# feature_variables = loaded_model.steps[1][1].booster_.feature_name()
# # st.write(feature_variables)

# importances_gini = pd.Series(loaded_model.steps[1][1].feature_importances_, index = X.columns)
# sorted_importances = importances_gini.sort_values()

# sorted_importances


# plot = px.bar(sorted_importances, 
# 				title="Relative Influence",
# 				labels={"index": "Metric",
# 						"value": "% Influence"},
# 				height=1000, 
# 				orientation="h",
# 				text_auto=True)
# plot.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
# plot.update_layout(showlegend=False)
# st.plotly_chart(plot, use_container_width=True)


# # st.write(loaded_model.predict(X))



# appended_pdps = pd.DataFrame()


# for feature, n in zip(feature_variables, range(0, len(feature_variables))):

#   pdp_data = partial_dependence(loaded_model, features = [(n)], X=X.fillna(0), percentiles = (0, 1), grid_resolution = 100)

#   pdp_df = pd.DataFrame(data = pdp_data["average"].transpose())
#   pdp_append = pd.DataFrame(data = np.array(pdp_data["values"]).transpose(), columns = [feature])
#   pdp_df = pd.concat([pdp_append, pdp_df], axis = 1).reset_index(drop = True)

#   pdp_df["Metric"] = pdp_df.columns[0]  
#   pdp_df = pdp_df.rename(columns={pdp_df.columns[0]: "Value", pdp_df.columns[1]: "Predicted Response"})
#   pdp_df = pdp_df[["Metric", "Value", "Predicted Response"]]

#   appended_pdps = appended_pdps.append(pdp_df)

# st.write(appended_pdps)


# pdp_selection = st.selectbox(
#     "# Select A Metric",
#     key='pdp',
#     options=feature_variables)


# pdp_select = appended_pdps[appended_pdps["Metric"] == pdp_selection]

# plot = px.line(pdp_select, 
# 					x="Value", 
# 					y="Predicted Response", 
# 					labels={"Value": pdp_selection,
# 						"Predicted Response": "Predicted Response"},
# 					title=(f'Predicted Response vs. {pdp_selection}'))


# st.plotly_chart(plot, use_container_width=True)




# interact_selection_a = st.selectbox(
#     "# Select A Metric",
#     key = "interact1", 
#     options=feature_variables)

# interact_selection_b = st.selectbox(
#     "# Select A Metric",
#     key = "interact2",
#     options=feature_variables)


# features_to_plot = [interact_selection_a, interact_selection_b]
# percentilesx = [0, 1]
# percentilesy = [0, 0.67]

# plot_params = {
     
#     'title': features_to_plot[0] + ' and ' + features_to_plot[1], # plot title and subtitle      
#     'subtitle': 'X Percentiles: ' +  str(percentilesx) +  '\n' + 'Y Percentiles: ' + str(percentilesy),
#     'title_fontsize': 15,     
#     'subtitle_fontsize': 12,       
#     'contour_color':  'white', # color for contour line#       
#     'font_family': 'Century Gothic',  
#     'cmap': matplotlib.cm.get_cmap(name='viridis', lut = None), # matplotlib color map for interact plot       
#     'inter_fill_alpha': 1, # fill alpha for interact plot#       
#     'inter_fontsize': 1, # fontsize for interact plot text# 
# }




# df = X.copy()
# model = loaded_model

# inputdata = df[(df[features_to_plot[0]] > np.quantile(df[features_to_plot[0]].dropna(), percentilesx[0])) & (df[features_to_plot[0]] < np.quantile(df[features_to_plot[0]].dropna(), percentilesx[1])) & (df[features_to_plot[1]] > np.quantile(df[features_to_plot[1]].dropna(), percentilesy[0])) & (df[features_to_plot[1]] < np.quantile(df[features_to_plot[1]].dropna(), percentilesy[1]))]
# inter1  =  pdp.pdp_interact(model=model, dataset= inputdata, model_features=df.columns, features=features_to_plot, num_grid_points=[16, 16])
# pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid', plot_params = plot_params)
# plt.savefig('Outputs/Interaction_Plot.png')


# image = Image.open('Outputs/Interaction_Plot.png')
# st.image(image, caption='Interaction Plot')