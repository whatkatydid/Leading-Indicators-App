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
loaded_model = st.session_state["loaded_model"]
X = st.session_state["X"]
data = st.session_state["data"]
feature_variables = st.session_state["feature_variables"]
rf_results = st.session_state["rf_results"]

st.write("# :red[Two-Feature Interaction Plots:]" + f" :blue[***{model_select}***]")

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

#try:
#	appended_pdps = st.session_state["appended_pdps"]
#except KeyError:
#	st.error('Please run Partial Dependence Plots tab to continue!', icon="🛑")
#	quit()


with st.form(key='input_columns'):
	c1, c2 = st.columns([1, 1], gap="large")
	with c1:
		interact_selection_a = st.selectbox(
	    	"# Select A Metric",
	    	key = "interact_a", 
	    	options=feature_variables,
	    	index=0)
	with c2:
		interact_selection_b = st.selectbox(
	    	"# Select Another Metric",
	    	key = "interact_b",
	    	options=feature_variables,
	    	index=2)
	submitButton = st.form_submit_button(label = 'Refresh Plots')


if interact_selection_a == interact_selection_b:
	st.error('Please select two different metrics to continue!', icon="🛑")
	quit()


features_to_plot = [interact_selection_a, interact_selection_b]
percentilesx = [0, 0.95]
percentilesy = [0, 0.95]

plot_params = {
    'title': features_to_plot[0] + ' and ' + features_to_plot[1], # plot title and subtitle      
    'subtitle': 'X Percentiles: ' +  str(percentilesx) +  '\n' + 'Y Percentiles: ' + str(percentilesy),
    'title_fontsize': 15,     
    'subtitle_fontsize': 12,       
    'contour_color':  'white', # color for contour line#       
    'font_family': 'Century Gothic',  
    'cmap': matplotlib.cm.get_cmap(name='viridis', lut = None), # matplotlib color map for interact plot       
    'inter_fill_alpha': 1, # fill alpha for interact plot#       
    'inter_fontsize': 1, # fontsize for interact plot text# 
}


df = X.copy()
model = loaded_model

df.columns = df.columns.str.replace('_bu', '') # For Amex models
inputdata = df[(df[features_to_plot[0]] > np.quantile(df[features_to_plot[0]].dropna(), percentilesx[0])) & (df[features_to_plot[0]] < np.quantile(df[features_to_plot[0]].dropna(), percentilesx[1])) & (df[features_to_plot[1]] > np.quantile(df[features_to_plot[1]].dropna(), percentilesy[0])) & (df[features_to_plot[1]] < np.quantile(df[features_to_plot[1]].dropna(), percentilesy[1]))]


try:
	inter1  =  pdp.pdp_interact(model=model, dataset= inputdata, model_features=df.columns, features=features_to_plot, num_grid_points=[12, 12])
	pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='grid', plot_params = plot_params)
	plt.savefig('Outputs/Interaction_Plot.png')
except ValueError:
	st.error('Plot cannot be created due to insufficient data. Please select two different metrics to continue!', icon="🛑")
	quit()
except IndexError:
	st.error('Plot cannot be created due to insufficient data. Please select two different metrics to continue!', icon="🛑")
	quit()


image = Image.open('Outputs/Interaction_Plot.png')
image_res = image.crop((0, 280, 725, 900))

col1, col2 = st.columns([1, 1], gap="large")

with col1:

	# st.markdown("<h1 style='text-align: center") 

	if st.form_submit_button:

		st.write("")
		st.write(f'##### **Interaction Plot: {interact_selection_a} vs. {interact_selection_b}**')
		st.image(image_res, caption='Interaction Plot', width=400, use_column_width='auto')

with col2:

	pdp_select = appended_pdps[(appended_pdps["Metric"] == interact_selection_a) | (appended_pdps["Metric"] == interact_selection_b)]

	plot = px.line(pdp_select, 
					x="Value", 
					y="Predicted Response", 
					color="Metric",
					facet_col="Metric",
					facet_col_wrap=1,
					facet_row_spacing=0.15,
					facet_col_spacing=0.1,
					height=700, 
					render_mode="svg",
					line_shape="spline",
					labels={"Value": "",
								"Predicted Response": "Predicted Response"})


	plot.for_each_annotation(lambda a: a.update(text=" ".join(("Predicted Response vs. ", a.text.split("=")[-1]))))
	# plot.for_each_annotation(lambda a: a.update(text=f'<b>{a.text}</b>'))
	plot.update_annotations(font_size=16)
	plot.update_yaxes(matches=None, showticklabels=True)
	plot.update_xaxes(matches=None, showticklabels=True)
	plot.update_layout(showlegend=False)

	st.plotly_chart(plot, use_container_width=True)
	

# Refresh pdps with new selected model
st.session_state["appended_pdps"] = appended_pdps