import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import pickle
from model import Normal 

# Load both ML models
with open("./code/model/car_selling_price_old.model", "rb") as f:
    model_old = pickle.load(f)

with open("./code/model/car_selling_price_new.model", "rb") as f:
    new_loaded_model = pickle.load(f)
    new_model = new_loaded_model['model']  # Extract model
    new_scaler = new_loaded_model['scaler']  # Extract scaler
    new_year_default = new_loaded_model.get('year', 2014)  # Default value if missing
    new_mileage_default = new_loaded_model.get('mileage', 21.24)
    new_max_power_default = new_loaded_model.get('max_power', 103.52)

# Initialize Dash app with Bootstrap styling
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 🚀 **Navigation Header**
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Old Model", href="/old")),
        dbc.NavItem(dbc.NavLink("New Model", href="/new")),
    ],
    brand="Car Price Prediction",
    brand_href="/",
    color="primary",
    dark=True,
    className="mb-4"
)

# 🎯 **Home Page**
home_page = html.Div([
    navbar,
    html.H1("Welcome to Car Price Prediction", className="text-center text-primary mb-4"),
    html.P("This app allows you to predict the selling price of a car based on its year, mileage, and max power.",
           className="text-center text-secondary"),
    
    html.Ul([
        html.Li("Old Model: Uses a traditional ML approach."),
        html.Li("New Model: Uses an upgraded, more accurate ML model with scaling."),
    ], className="text-center"),
    
    html.Div(
        dcc.Link("Go to Prediction →", href="/old", className="btn btn-primary mt-3"),
        className="text-center"
    )
])

# 📌 **Common Input Fields (for both models)**
input_form = html.Div([
    html.Label("Year of Manufacture:", style={"marginTop": "10px"}),
    dcc.Input(id="input-year", type="number", placeholder="Enter year", required=True, style={"marginBottom": "15px"}),

    html.Label("Mileage (in KM):", style={"marginTop": "10px"}),
    dcc.Input(id="input-mileage", type="number", placeholder="Enter mileage", required=True, style={"marginBottom": "15px"}),

    html.Label("Max Power (in HP):", style={"marginTop": "10px"}),
    dcc.Input(id="input-maxpower", type="number", placeholder="Enter max power", required=True, style={"marginBottom": "15px"}),

    html.Button("Calculate", id="predict-button", n_clicks=0, style={"marginTop": "20px"}),
], style={
    "width": "50%", 
    "margin": "auto", 
    "padding": "20px", 
    "border": "1px solid black", 
    "borderRadius": "10px", 
    "display": "flex", 
    "flexDirection": "column"
})

# ✅ **Old Model Page**
old_model_layout = html.Div([
    navbar,
    html.H1("Car Selling Price Prediction (Old Model)", style={"textAlign": "center"}),

    html.P("To predict the car price, enter the details below.", 
           style={"textAlign": "center", "marginBottom": "20px"}),

    input_form,
    html.Div(id="output-prediction-old", style={"textAlign": "center", "marginTop": "20px", "fontSize": "20px"})
])

# 🎨 **New Model Page (Modern & Stylish)**
new_model_layout = html.Div([
    navbar,
    html.Div(
        [
            html.H1("🚗 Car Price Prediction (New Model)", className="text-center text-primary mb-4"),
            html.P("Experience an upgraded ML model for better accuracy!", className="text-center text-secondary"),
            
            dbc.Container([
                dbc.Row([
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody([
                                html.Label("Year of Manufacture:"),
                                dcc.Input(id="input-year-new", type="number", placeholder="Enter year", className="form-control mb-3", value=new_year_default),

                                html.Label("Mileage (in KM):"),
                                dcc.Input(id="input-mileage-new", type="number", placeholder="Enter mileage", className="form-control mb-3", value=new_mileage_default),

                                html.Label("Max Power (in HP):"),
                                dcc.Input(id="input-maxpower-new", type="number", placeholder="Enter max power", className="form-control mb-3", value=new_max_power_default),

                                dbc.Button("Calculate", id="predict-button-new", color="primary", className="mt-3 w-100"),
                            ])
                        ), width=6
                    )
                ], className="justify-content-center mt-4"),
                
                dbc.Row([
                    dbc.Col(html.Div(id="output-prediction-new", className="text-center mt-3 text-success fw-bold"), width=6)
                ], className="justify-content-center")
            ])
        ], className="container"
    )
])

# 🌍 **Routing (Switch Between Pages)**
app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content")
])

@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname")
)
def display_page(pathname):
    if pathname == "/new":
        return new_model_layout
    elif pathname == "/old":
        return old_model_layout
    else:
        return home_page  # Default Home Page

# 🚀 **Old Model Prediction**
@app.callback(
    Output("output-prediction-old", "children"),
    Input("predict-button", "n_clicks"),
    State("input-year", "value"),
    State("input-mileage", "value"),
    State("input-maxpower", "value")
)
def predict_price_old(n_clicks, year, mileage, max_power):
    if not n_clicks:
        return ""

    if year is None or mileage is None or max_power is None:
        return "Please fill in all fields to calculate the price."

    input_features = pd.DataFrame({"year": [year], "mileage": [mileage], "max_power": [max_power]})
    predicted_price = model_old.predict(input_features)[0]
    predicted_price = np.exp(predicted_price)
    return f"The predicted selling price (Old Model) is: ${predicted_price:,.2f}"

# 🎯 **New Model Prediction (with Scaling)**
@app.callback(
    Output("output-prediction-new", "children"),
    Input("predict-button-new", "n_clicks"),
    State("input-year-new", "value"),
    State("input-mileage-new", "value"),
    State("input-maxpower-new", "value")
)
def predict_price_new(n_clicks, year, mileage, max_power):
    if not n_clicks:
        return ""

    if year is None or mileage is None or max_power is None:
        return "Please fill in all fields to calculate the price."

    sample        = np.array([[year, mileage, max_power]])
    sample_scaled = new_scaler.transform(sample)
    intercept     = np.ones((sample_scaled.shape[0], 1))
    sample_scaled = np.concatenate((intercept, sample_scaled), axis=1)
    predicted_price = np.exp(new_model.predict(sample_scaled))
    return predicted_price

# ✅ **Run the app**
if __name__ == "__main__":
    app.run_server(host = '0.0.0.0', port=9009)
