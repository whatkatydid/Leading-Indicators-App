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

# for k, v in st.session_state.items():
# 	if k not in st.session_state:
# 	    st.session_state[k] = v


# for k, v in st.session_state.items():
# 	st.session_state[k] = v

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


model_select = st.session_state["model_select"]
loaded_model = st.session_state["loaded_model"]
X = st.session_state["X"]
data = st.session_state["data"]
feature_variables = st.session_state["feature_variables"]
rf_results = st.session_state["rf_results"]

st.write("# :red[Partial Dependence Plots:]" + f" :blue[***{model_select}***]")

@st.cache_data(show_spinner=False)
def append_pdps(feature_variables, _loaded_model):
	appended_pdps = pd.DataFrame()

	with st.spinner("Please wait..."):
		for feature, n in zip(feature_variables, range(0, len(feature_variables))):
		  pdp_data = partial_dependence(loaded_model, features = [(n)], X=X.fillna(0), percentiles = (0, 1), grid_resolution = 100)

		  pdp_df = pd.DataFrame(data = pdp_data["average"].transpose())
		  pdp_append = pd.DataFrame(data = np.array(pdp_data["values"]).transpose(), columns = [feature])
		  pdp_df = pd.concat([pdp_append, pdp_df], axis = 1).reset_index(drop = True)

		  pdp_df["Metric"] = pdp_df.columns[0]  
		  pdp_df = pdp_df.rename(columns={pdp_df.columns[0]: "Value", pdp_df.columns[1]: "Predicted Response"})
		  pdp_df = pdp_df[["Metric", "Value", "Predicted Response"]]

		  appended_pdps = appended_pdps.append(pdp_df)
	return appended_pdps
appended_pdps = append_pdps(feature_variables, loaded_model)


tab1, tab2 = st.tabs(["View All", "Compare"])
with tab1:
	plot = px.line(appended_pdps, 
					x="Value", 
					y="Predicted Response", 
					color="Metric",
					facet_col="Metric",
					facet_col_wrap=3,
					facet_row_spacing=0.025,
					facet_col_spacing=0.1,
					height=4000, 
					render_mode="svg",
					line_shape="spline",
					labels={"Value": "",
								"Predicted Response": "Predicted Response"})

	plot.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
	plot.update_annotations(font_size=16)
	plot.update_yaxes(matches=None, showticklabels=True)
	plot.update_xaxes(matches=None, showticklabels=True)
	plot.update_layout(showlegend=False)

	st.plotly_chart(plot, use_container_width=True)

with tab2:
	col1, col2 = st.columns(2, gap="large")

	with col1:

		pdp_selection = st.selectbox(
	    "# Select A Metric",
	    key='pdp_a',
	    options=feature_variables)

		pdp_select = appended_pdps[appended_pdps["Metric"] == pdp_selection]

		plot = px.line(pdp_select, 
						x="Value", 
						y="Predicted Response", 
						render_mode="svg",
						line_shape="spline",
						height=550,
						color_discrete_sequence=["royalblue"],
						labels={"Value": pdp_selection,
							"Predicted Response": "Predicted Response"},
						title=(f'Predicted Response vs. {pdp_selection}'))

		st.plotly_chart(plot, use_container_width=True)

	with col2:
		pdp_selection = st.selectbox(
	    "# Select Another Metric",
	    index=3,
	    key='pdp_b',
	    options=feature_variables)

		pdp_select = appended_pdps[appended_pdps["Metric"] == pdp_selection]

		plot = px.line(pdp_select, 
						x="Value", 
						y="Predicted Response", 
						render_mode="svg",
						line_shape="spline",
						height=550,
						color_discrete_sequence=["lightskyblue"],
						labels={"Value": pdp_selection,
							"Predicted Response": "Predicted Response"},
						title=(f'Predicted Response vs. {pdp_selection}'))

		st.plotly_chart(plot, use_container_width=True)


# Refresh pdps with new selected model
st.session_state["appended_pdps"] = appended_pdps