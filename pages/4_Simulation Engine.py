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
#     st.session_state[k] = v


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
dependent_var = st.session_state["dependent_var"]
loaded_model = st.session_state["loaded_model"]
X = st.session_state["X"]
data = st.session_state["data"]
feature_variables = st.session_state["feature_variables"]


st.write("# :red[Simulation Engine:]" + f" :blue[***{model_select}***]")


X.columns = X.columns.str.replace('_bu', '') # For Amex models
act_y = data[dependent_var]
X = data[feature_variables]


rows = ["a", "b", "c", "d", "e"]
sim_features = []
sim_values = []


# Function to clear simulation values with button click
def reset_value():
	for r in rows:
		st.session_state["select_" + r] = " "
		st.session_state["sim_" + r] = -1

st.write("***")
col1, col2 = st.columns([7,1], gap="large")
with col1:
		st.button("Reset Simulation", on_click=reset_value)
st.write("")


col1, col2, col3 = st.columns([1, 1, 2], gap="large")

with col1:
	st.write("Select Predictors")
with col2:
	st.write("Select Test Values")
with col3:
	st.write("Forecasted Results")


c1, c2 = st.columns([1, 1], gap="large")

with c1:
	for row in rows:
		col1, col2 = st.columns([1, 1], gap="large")
		with col1:
			sim_selection = st.selectbox("# Select A Predictor",
		    			key = "select_" + row, 
		    			options = [" "] + feature_variables,
		    			label_visibility="collapsed"
		    			)

			sim_features.append(sim_selection)

			st.write("")
			st.write("")

		with col2:
			if sim_selection != " ":
				sim_value = st.slider(label='Select Test Values', 
											min_value=X[sim_selection].min()*1.0, 
											max_value=X[sim_selection].max()*1.0, 
											value=X[sim_selection].mean(),
											step=(X[sim_selection].max()-X[sim_selection].min())/200,
											format="%0.4f",
											key="sim_" + row,
											label_visibility="collapsed")

				
			else:
					st.select_slider(label='Clear Test Values', 
											options=[' ', "None Selected"],
											key="sim_clear" + row,
											label_visibility="collapsed")

					sim_value = -1
			sim_values.append(sim_value)


	with c2:


		# st.write(sim_features)
		# st.write(sim_values)

		sim_X = X.copy()

		for feature, n in zip(sim_features, sim_values):
			if feature != " ":
				sim_X[feature] = n
	
		sim_y = loaded_model.predict(sim_X)


		# If no predictors selected, reset simulation results
		if all(x == " " for x in sim_features):
			sim_y = act_y


		sim_results = pd.DataFrame({"Current": sim_y, "Simulated": sim_y})
		sim_agg = pd.DataFrame({"Type": ["Current", "Simulated"], "Response":[act_y.mean(), sim_y.mean()]})


		sim_plot = px.bar(sim_agg,
								title="Expected Lift: " + str('{:,.2%}'.format(sim_y.mean()/act_y.mean()-1)),
								x="Type",
								y="Response",
								color="Type",
								labels={"Type": "",
								"Response": "Reponse"},
		 						text_auto='0.3'
		 						)
		sim_plot.update_traces(textfont_size=12, 
								textangle=0, 
								textposition="outside", 
								cliponaxis=False)
		sim_plot.update_yaxes(
								range=(min(act_y.mean(), sim_y.mean())*.9, max(act_y.mean(), sim_y.mean())*1.03),
    							constrain='domain')
		sim_plot.update_layout(title_x=0.4, showlegend=False)
		st.plotly_chart(sim_plot, use_container_width=True)


# st.write(sim_features)
#st.write(sim_values)

# st.button("Reset Simulation", on_click=reset_value)

# if st.button('Reset Simulation'):
# 		sim_features = [" " for i in sim_features]
# 		sim_values = [" " for i in sim_values]






# for col in columns:

# 	if "select_" + col not in st.session_state:
# 		st.session_state["select_" + col] = "select_" + col




# st.write(st.session_state.sim_a)



