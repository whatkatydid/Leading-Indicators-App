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
    layout="wide",
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


hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


# filename = "../../../Social/Modeling/Saved Models/final_model_1221_Platforms.pkl"
# data = "../../../Social/Modeling/Saved Models/final_df_1221_Platforms.pkl"
# results = "../../../Social/Modeling/Results/Global/AMEX_LIC_Summary_Social_Platforms_1221.xlsx"

# loaded_model = pickle.load(open(filename, "rb"))
# X = pd.read_pickle(data)
# rf_results = pd.read_excel(results, sheet_name="Relative Importance")
# data = pd.read_excel(results, sheet_name="Modeled Data")

# feature_variables = loaded_model.steps[1][1].booster_.feature_name()


model_select = st.session_state["model_select"]
loaded_model = st.session_state["loaded_model"]
X = st.session_state["X"]
data = st.session_state["data"]
feature_variables = st.session_state["feature_variables"]
rf_results = st.session_state["rf_results"]

st.write("# :red[Features of Relative Influence:]" + f" :blue[***{model_select}***]")


importances_gini = pd.Series(loaded_model.steps[1][1].feature_importances_, index = X.columns)
sorted_importances = importances_gini.sort_values()
sorted_importances.index = sorted_importances.index.str.replace('_bu', '') # For Amex models

rf_all = pd.DataFrame(sorted_importances, columns = ['Importance'])
rf_all["GINI IMPORTANCE"] = rf_all['Importance'] / rf_all['Importance'].sum()
rf_all = pd.merge(rf_results[["VARIABLE", "TYPE", "SUBTYPE"]], rf_all[["GINI IMPORTANCE"]], left_on = 'VARIABLE', right_index = True)

rf_all = rf_all.reset_index()
rf_all = rf_all.sort_values(by=["TYPE", "GINI IMPORTANCE"], ascending=[False, True])

plot_all = px.bar(rf_all, 
				x='GINI IMPORTANCE',
				y='VARIABLE',
				color='TYPE',
				title="Relative Influence: All Metrics",
				labels={"VARIABLE": "Metric",
						"TYPE": "Category",
						"GINI IMPORTANCE": "% Influence"},
				height=850, 
				orientation="h",
				text_auto='.1%')
plot_all.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
plot_all.update_layout(showlegend=True, legend_traceorder="reversed", legend=dict(y=1,  x=1.1))
plot_all.update_xaxes(tickformat = ',.0%')
# st.plotly_chart(plot_all, use_container_width=True)


rf_subtype = pd.DataFrame(rf_results.groupby(["TYPE", "SUBTYPE"])["GINI IMPORTANCE", "PERMUTATION IMPORTANCE"].sum())
rf_subtype = rf_subtype.reset_index()

rf_subtype = rf_subtype.sort_values(by=["TYPE","GINI IMPORTANCE"], ascending=[False, True])

# st.write(rf_subtype)


plot_subtype = px.bar(rf_subtype, 
				x='GINI IMPORTANCE',
				y='SUBTYPE',
				color='TYPE',
				title="Relative Influence: Metrics",
				labels={"TYPE": "Category",
						"SUBTYPE": "Metric",
						"GINI IMPORTANCE": "% Influence"},
				height=850, 
				orientation="h",
				text_auto='.1%')
plot_subtype.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
plot_subtype.update_layout(showlegend=True, legend_traceorder="reversed", legend=dict(y=1,  x=1.1))
plot_subtype.update_xaxes(tickformat = ',.0%')


# st.plotly_chart(plot_subtype, use_container_width=True)


rf_type = pd.DataFrame(rf_results.groupby(["TYPE"])["GINI IMPORTANCE", "PERMUTATION IMPORTANCE"].sum())
rf_type = rf_type.reset_index()
rf_type = rf_type.sort_values(by=["TYPE","GINI IMPORTANCE"], ascending=[True, False])

# st.write(rf_type)


plot_type = px.bar(rf_type, 
				x='GINI IMPORTANCE',
				y='TYPE',
				color='TYPE',
				title="Relative Influence: Categories",
				labels={"TYPE": "Category",
						"GINI IMPORTANCE": "% Influence"},
				height=850, 
				orientation="h",
				text_auto='.1%',
				color_discrete_sequence=["red", "lightskyblue","royalblue"]
				#color_discrete_map={'Structural': 'royalblue', 
            #                    'Media: KPI': 'lightskyblue', 
            #                    'Media: Campaign': 'red'}
            ) # For Amex models
plot_type.update_traces(textfont_size=12, textangle=0, textposition="outside", cliponaxis=False)
plot_type.update_layout(showlegend=True, legend=dict(y=1,  x=1.1))
plot_type.update_xaxes(tickformat = ',.0%')
# st.plotly_chart(plot_type, use_container_width=True)


col1, col2 = st.columns(2)

with col1:
	tab1, tab2 = st.tabs(["View Categories", " "])
	with tab1:
		st.plotly_chart(plot_type, use_container_width=True)
with col2:
	tab3, tab4 = st.tabs(["View Metrics", "View All"])
	with tab3:
		st.plotly_chart(plot_subtype, use_container_width=True)
	with tab4:
		st.plotly_chart(plot_all, use_container_width=True)







# st.write(rf_all["GINI IMPORTANCE"].sum())
# st.write(rf_subtype["GINI IMPORTANCE"].sum())
# st.write(rf_type["GINI IMPORTANCE"].sum())


# if "loaded_model" not in st.session_state:
# 	st.session_state["loaded_model"] = loaded_model
# if "X" not in st.session_state:
# 	st.session_state["X"] = X
# if "feature_variables" not in st.session_state:
# 	st.session_state["feature_variables"] = feature_variables
# if "data" not in st.session_state:
# 	st.session_state["data"] = data