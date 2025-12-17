"""
Superconductor Dashboard with SQL Playground
==============================================
Author: Ankita Biswas
Project: Public Dashboard for Superconductors
Date: December 2025

Interactive dashboard with SQL query interface for exploring superconductivity data
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

PLOTLY_CONFIG = {
    'title': {'font': {'size': 24}},
    'xaxis': {'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},
    'yaxis': {'title': {'font': {'size': 18}}, 'tickfont': {'size': 16}},
    'legend': {'font': {'size': 16}},
    'font': {'size': 16}
}


app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Import Garamond font */
            @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:wght@400;500;600;700&display=swap');
            
            /* UVA Brand Colors */
            :root {
                --uva-navy: #232D4B;
                --uva-orange: #E57200;
            }
            
            /* Apply Garamond to everything */
            * {
                font-family: 'EB Garamond', 'Garamond', 'Times New Roman', serif !important;
            }
            
            body {
                font-family: 'EB Garamond', 'Garamond', 'Times New Roman', serif;
                color: #232D4B;
            }
            
            h1, h2, h3, h4, h5, h6 {
                font-family: 'EB Garamond', 'Garamond', 'Times New Roman', serif;
                font-weight: 600;
                color: #232D4B;
            }
            
            /* Table styling */
            table {
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 2px 4px rgba(35, 45, 75, 0.1);
            }
            
            th {
                background-color: #232D4B;
                color: #FFFFFF;
                padding: 14px;
                text-align: left;
                font-weight: 600;
                border-bottom: 3px solid #E57200;
                font-size: 17px;
            }
            
            td {
                padding: 12px;
                border-bottom: 1px solid #ddd;
                font-size: 16px;
            }
            
            tr:nth-child(even) {
                background-color: #f8f9fa;
            }
            
            tr:hover {
                background-color: #FFF5E6;
            }
            
            .tab--selected {
                border-bottom: 3px solid #E57200 !important;
                color: #232D4B !important;
            }
            
            code {
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Monaco', 'Courier New', monospace;
                color: #E57200;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# DATABASE CONNECTION

def get_db_connection():
    """Create database connection using environment variables"""
    db_user = os.getenv('POSTGRES_USER', 'postgres')
    db_password = os.getenv('POSTGRES_PASSWORD', 'postgres')
    db_host = os.getenv('POSTGRES_HOST', 'db')
    db_port = os.getenv('POSTGRES_PORT', '5432')
    db_name = os.getenv('POSTGRES_DB', 'superconductors')
    
    connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    return create_engine(connection_string)


# DATA LOADING

def load_data():
    """Load data from 3NF PostgreSQL database"""
    engine = get_db_connection()
    
    # Load main dataset with JOINs
    query = """
    SELECT 
        m.data_number,
        m.chemical_formula,
        m.tc_kelvin,
        m.n_elements,
        c.category_name as category_detailed,
        f.family_name as material_family,
        q.tier_name as quality_tier,
        m.has_oxygen_var,
        m.is_high_tc,
        m.publication_year,
        m.tc_std,
        m.n_measurements
    FROM materials m
    LEFT JOIN categories c ON m.category_id = c.category_id
    LEFT JOIN families f ON m.family_id = f.family_id
    LEFT JOIN quality_tiers q ON m.tier_id = q.tier_id
    """
    df = pd.read_sql(query, engine)
    
    # Load element statistics
    elem_stats_query = """
    SELECT 
        e.element_symbol as element,
        es.count,
        es.mean_tc,
        es.median_tc
    FROM element_stats es
    JOIN elements e ON es.element_symbol = e.element_symbol
    ORDER BY es.mean_tc DESC
    """
    elem_stats = pd.read_sql(elem_stats_query, engine)
    
    # Load top features
    features_query = """
    SELECT 
        feature,
        composite_score
    FROM feature_importance
    ORDER BY composite_score DESC
    LIMIT 20
    """
    top_features = pd.read_sql(features_query, engine)
    
    return df, elem_stats, top_features

# Load data
df, elem_stats, top_features = load_data()

# TAB 1: OVERVIEW
def create_overview_tab(df):
    """Overview with Tc distribution"""
    
    stats = df['tc_kelvin'].describe()
    
    # Histogram
    fig = px.histogram(df, x='tc_kelvin', nbins=50,
                      title='Critical Temperature Distribution',
                      labels={'tc_kelvin': 'Tc (K)', 'count': 'Count'},
                      color_discrete_sequence=['steelblue'])
    fig.add_vline(x=77, line_dash="dash", line_color="green",
                 annotation_text="LN₂ (77K)",
                 annotation_font_size=18)
    fig.update_layout(
        height=400,
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        font=dict(size=16)
    )
    
    return html.Div([
        html.H3("Dataset Overview"),
        html.P(f"Total materials: {len(df):,}"),
        
        # Stats cards
        html.Div([
            html.Div([
                html.H3(f"{stats['mean']:.1f} K", style={'color': '#3498db', 'margin': '0'}),
                html.P("Mean Tc", style={'margin': '0'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
            
            html.Div([
                html.H3(f"{stats['50%']:.1f} K", style={'color': '#e74c3c', 'margin': '0'}),
                html.P("Median Tc", style={'margin': '0'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
            
            html.Div([
                html.H3(f"{stats['max']:.1f} K", style={'color': '#2ecc71', 'margin': '0'}),
                html.P("Maximum Tc", style={'margin': '0'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
            
            html.Div([
                html.H3(f"{(df['tc_kelvin'] > 77).sum():,}", style={'color': '#9b59b6', 'margin': '0'}),
                html.P("High-Tc (>77K)", style={'margin': '0'})
            ], style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#ecf0f1',
                     'borderRadius': '5px', 'flex': '1', 'margin': '5px'}),
        ], style={'display': 'flex', 'marginBottom': '20px'}),
        
        dcc.Graph(figure=fig)
    ])


# TAB 2: ELEMENT EXPLORER
def create_elements_tab(df):
    """Element prevalence analysis"""
    
    return html.Div([
        html.H3("Element Prevalence and Tc Statistics"),
        
        html.Div([
            html.Label("Select Elements to Compare:"),
            dcc.Dropdown(
                id='element-dropdown',
                options=[{'label': elem, 'value': elem} for elem in elem_stats['element'].head(15)],
                value=elem_stats['element'].head(5).tolist(),
                multi=True
            )
        ], style={'marginBottom': '20px'}),
        
        html.Div(id='element-graphs')
    ])

@app.callback(
    Output('element-graphs', 'children'),
    Input('element-dropdown', 'value')
)
def update_element_graphs(selected_elements):
    if not selected_elements:
        return html.P("Please select at least one element")
    
    filtered_stats = elem_stats[elem_stats['element'].isin(selected_elements)]
    
    fig = px.bar(filtered_stats, x='element', y='mean_tc',
                 title='Mean Tc by Element',
                 labels={'mean_tc': 'Mean Tc (K)', 'element': 'Element'},
                 color='mean_tc', color_continuous_scale='RdYlBu_r')
    fig.add_hline(y=77, line_dash="dash", line_color="green",
                 annotation_text="LN₂ (77K)",
                 annotation_font_size=18)
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        font=dict(size=16)
    )
    
    return dcc.Graph(figure=fig)

# TAB 3: COMPOSITION TRENDS

def create_trends_tab(df):
    """Composition complexity and temporal trends"""
    
    complexity_stats = df.groupby('n_elements')['tc_kelvin'].agg(['mean', 'std', 'count']).reset_index()
    complexity_stats = complexity_stats[complexity_stats['count'] >= 5]
    
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        x=complexity_stats['n_elements'],
        y=complexity_stats['mean'],
        error_y=dict(type='data', array=complexity_stats['std']),
        marker_color='steelblue'
    ))
    fig1.add_hline(y=77, line_dash="dash", line_color="green",
                  annotation_text="LN₂ (77K)",
                  annotation_font_size=18)
    fig1.update_layout(
        title='Tc vs Compositional Complexity',
        xaxis_title='Number of Elements',
        yaxis_title='Mean Tc (K)',
        height=400,
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        font=dict(size=16))
    
    # Temporal trends
    if 'publication_year' in df.columns:
        year_stats = df.groupby('publication_year').agg({
            'tc_kelvin': ['count', 'mean', 'max']
        }).reset_index()
        year_stats.columns = ['year', 'count', 'mean_tc', 'max_tc']
        year_stats = year_stats[year_stats['count'] >= 5]
        
        fig2 = make_subplots(rows=2, cols=1, subplot_titles=('Discoveries Over Time', 'Tc Evolution'))
        
        fig2.add_trace(go.Bar(x=year_stats['year'], y=year_stats['count'],
                             marker_color='steelblue'), row=1, col=1)
        
        fig2.add_trace(go.Scatter(x=year_stats['year'], y=year_stats['max_tc'],
                                 mode='lines+markers', name='Max Tc',
                                 line=dict(color='red', width=2)), row=2, col=1)
        fig2.add_trace(go.Scatter(x=year_stats['year'], y=year_stats['mean_tc'],
                                 mode='lines+markers', name='Mean Tc',
                                 line=dict(color='blue', width=2)), row=2, col=1)
        
        fig2.update_xaxes(title_text="Year", row=2, col=1, title_font_size=18, tickfont_size=16)
        fig2.update_yaxes(title_text="Count", row=1, col=1, title_font_size=18, tickfont_size=16)
        fig2.update_yaxes(title_text="Tc (K)", row=2, col=1, title_font_size=18, tickfont_size=16)
        fig2.update_layout(height=700, font=dict(size=16))
        
        return html.Div([
            dcc.Graph(figure=fig1),
            dcc.Graph(figure=fig2)
        ])
    else:
        return html.Div([dcc.Graph(figure=fig1)])


# TAB 4: DATA QUALITY

def create_quality_tab(df):
    """Data quality metrics"""
    
    # Calculate metrics
    high_tc_count = df['is_high_tc'].sum() if 'is_high_tc' in df.columns else 0
    oxygen_var_count = df['has_oxygen_var'].sum() if 'has_oxygen_var' in df.columns else 0
    multi_meas_count = (df['n_measurements'] > 1).sum() if 'n_measurements' in df.columns else 0
    low_unc_count = (df['tc_std'] < 1).sum() if 'tc_std' in df.columns else 0
    
    high_tc_pct = 100 * high_tc_count / len(df)
    oxygen_var_pct = 100 * oxygen_var_count / len(df)
    multi_meas_pct = 100 * multi_meas_count / len(df)
    low_unc_pct = 100 * low_unc_count / len(df)
    
    metrics_df = pd.DataFrame({
        'Metric': ['High-Tc (>77K)', 'Oxygen Variability', 'Multiple Measurements', 'Low Uncertainty (<1K)'],
        'Count': [high_tc_count, oxygen_var_count, multi_meas_count, low_unc_count],
        'Percentage': [high_tc_pct, oxygen_var_pct, multi_meas_pct, low_unc_pct]
    })
    
    fig = px.bar(metrics_df, x='Metric', y='Percentage',
                title='Data Quality Indicators',
                labels={'Percentage': 'Percentage of Materials (%)'},
                color='Percentage', color_continuous_scale='RdYlGn')
    fig.update_layout(
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        font=dict(size=16)
    )
    
    return html.Div([
        html.H3("Data Quality Dashboard"),
        dcc.Graph(figure=fig),
        
        html.Div([
            html.H4("Quality Summary", style={'marginBottom': '15px'}),
            dash_table.DataTable(
                data=[
                    {'Metric': 'High-Tc (>77K)', 'Count': f"{high_tc_count:,}", 'Percentage': f"{high_tc_pct:.1f}%"},
                    {'Metric': 'Oxygen Variability', 'Count': f"{oxygen_var_count:,}", 'Percentage': f"{oxygen_var_pct:.1f}%"},
                    {'Metric': 'Multiple Measurements', 'Count': f"{multi_meas_count:,}", 'Percentage': f"{multi_meas_pct:.1f}%"},
                    {'Metric': 'Low Uncertainty (<1K)', 'Count': f"{low_unc_count:,}", 'Percentage': f"{low_unc_pct:.1f}%"}
                ],
                columns=[
                    {'name': 'Metric', 'id': 'Metric'},
                    {'name': 'Count', 'id': 'Count'},
                    {'name': 'Percentage', 'id': 'Percentage'}
                ],
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontFamily': 'EB Garamond, Garamond, serif',
                    'fontSize': '16px'
                },
                style_header={
                    'backgroundColor': '#232D4B',
                    'color': 'white',
                    'fontWeight': '600',
                    'borderBottom': '3px solid #E57200',
                    'fontFamily': 'EB Garamond, Garamond, serif',
                    'fontSize': '17px'
                },
                style_data={
                    'backgroundColor': 'white',
                    'color': '#232D4B',
                    'borderBottom': '1px solid #ddd'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
                    {'if': {'column_id': 'Percentage'}, 'color': '#E57200', 'fontWeight': 'bold', 'textAlign': 'right'},
                    {'if': {'column_id': 'Count'}, 'fontWeight': 'bold', 'textAlign': 'right'}
                ]
            )
        ], style={'marginTop': '30px'})
    ])


# TAB 5: FEATURE IMPORTANCE

def create_features_tab(df, top_features):
    """Feature importance visualization"""
    
    fig = px.bar(top_features.head(15),
                x='composite_score',
                y='feature',
                orientation='h',
                title='Top 15 Features by Composite Importance Score',
                labels={'composite_score': 'Composite Score', 'feature': 'Feature'},
                color='composite_score',
                color_continuous_scale='Viridis')
    fig.update_layout(
        height=600,
        yaxis={'categoryorder': 'total ascending'},
        title_font_size=24,
        xaxis_title_font_size=18,
        yaxis_title_font_size=18,
        xaxis_tickfont_size=16,
        yaxis_tickfont_size=16,
        font=dict(size=16)
    )
    
    return html.Div([
        html.H3("Feature Importance for Tc Prediction"),
        html.P("Composite score combines F-test, mutual information, and random forest importance"),
        dcc.Graph(figure=fig),
        
        html.Div([
            html.H4("Top 10 Features", style={'marginBottom': '15px'}),
            dash_table.DataTable(
                data=[
                    {
                        'Rank': i+1,
                        'Feature': row['feature'],
                        'Score': f"{row['composite_score']:.3f}"
                    }
                    for i, (_, row) in enumerate(top_features.head(10).iterrows())
                ],
                columns=[
                    {'name': 'Rank', 'id': 'Rank'},
                    {'name': 'Feature', 'id': 'Feature'},
                    {'name': 'Score', 'id': 'Score'}
                ],
                style_cell={
                    'textAlign': 'left',
                    'padding': '12px',
                    'fontFamily': 'EB Garamond, Garamond, serif',
                    'fontSize': '16px'
                },
                style_header={
                    'backgroundColor': '#232D4B',
                    'color': 'white',
                    'fontWeight': '600',
                    'borderBottom': '3px solid #E57200',
                    'fontFamily': 'EB Garamond, Garamond, serif',
                    'fontSize': '17px'
                },
                style_data_conditional=[
                    {'if': {'row_index': 'odd'}, 'backgroundColor': '#f8f9fa'},
                    {'if': {'column_id': 'Score'}, 'color': '#E57200', 'fontWeight': 'bold', 'textAlign': 'right'},
                    {'if': {'column_id': 'Rank'}, 'fontWeight': 'bold', 'textAlign': 'center'}
                ]
            )
        ], style={'backgroundColor': '#ecf0f1', 'padding': '20px', 'borderRadius': '5px', 'marginTop': '20px'})
    ])


# TAB 6: SQL QUERY PLAYGROUND

def create_sql_tab():
    """Interactive SQL query interface"""
    
    # Example queries
    example_queries = {
        'Top 10 Highest Tc Materials': """SELECT 
    chemical_formula, 
    tc_kelvin, 
    material_family,
    publication_year
FROM materials m
LEFT JOIN families f ON m.family_id = f.family_id
ORDER BY tc_kelvin DESC
LIMIT 10;""",
        
        'Materials by Category Count': """SELECT 
    c.category_name,
    COUNT(*) as material_count,
    AVG(m.tc_kelvin) as avg_tc,
    MAX(m.tc_kelvin) as max_tc
FROM materials m
JOIN categories c ON m.category_id = c.category_id
GROUP BY c.category_name
ORDER BY material_count DESC;""",
        
        'Elements in High-Tc Materials': """SELECT 
    e.element_symbol,
    COUNT(DISTINCT me.material_id) as material_count,
    AVG(m.tc_kelvin) as avg_tc
FROM materials_elements me
JOIN elements e ON me.element_symbol = e.element_symbol
JOIN materials m ON me.material_id = m.data_number
WHERE m.tc_kelvin > 77
GROUP BY e.element_symbol
ORDER BY material_count DESC
LIMIT 15;""",
        
        'Recent Discoveries (Post-2000)': """SELECT 
    chemical_formula,
    tc_kelvin,
    publication_year,
    c.category_name
FROM materials m
JOIN categories c ON m.category_id = c.category_id
WHERE publication_year > 2000
ORDER BY publication_year DESC, tc_kelvin DESC
LIMIT 20;""",
        
        'Materials with Multiple Measurements': """SELECT 
    chemical_formula,
    tc_kelvin,
    tc_std,
    n_measurements,
    quality_tier
FROM materials m
LEFT JOIN quality_tiers q ON m.tier_id = q.tier_id
WHERE n_measurements > 5
ORDER BY n_measurements DESC
LIMIT 15;""",
        
        'Complex Materials (5+ Elements)': """SELECT 
    chemical_formula,
    tc_kelvin,
    n_elements,
    material_family
FROM materials m
LEFT JOIN families f ON m.family_id = f.family_id
WHERE n_elements >= 5
ORDER BY tc_kelvin DESC
LIMIT 20;"""
    }
    
    return html.Div([
        html.H2("SQL Query Playground 🎮", style={'color': '#232D4B', 'marginBottom': '20px'}),
        
        html.P([
            "Query the 3NF database directly! Explore 8 tables: ",
            html.Code("materials"), ", ",
            html.Code("categories"), ", ",
            html.Code("families"), ", ",
            html.Code("quality_tiers"), ", ",
            html.Code("elements"), ", ",
            html.Code("materials_elements"), ", ",
            html.Code("element_stats"), ", ",
            html.Code("feature_importance"),
        ], style={'fontSize': '16px', 'marginBottom': '20px'}),
        
        # Example queries dropdown
        html.Div([
            html.Label("Example Queries:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
            dcc.Dropdown(
                id='example-query-dropdown',
                options=[{'label': name, 'value': query} for name, query in example_queries.items()],
                placeholder="Select an example query...",
                style={'marginBottom': '15px'}
            )
        ]),
        
        # SQL textarea
        html.Div([
            html.Label("Your SQL Query:", style={'fontWeight': 'bold', 'fontSize': '16px', 'marginBottom': '10px'}),
            dcc.Textarea(
                id='sql-query-input',
                placeholder='Enter your SQL query here...\n\nExample:\nSELECT * FROM materials LIMIT 5;',
                style={
                    'width': '100%',
                    'height': '200px',
                    'fontFamily': 'Monaco, Courier, monospace',
                    'fontSize': '14px',
                    'padding': '15px',
                    'border': '2px solid #232D4B',
                    'borderRadius': '4px',
                    'backgroundColor': '#f8f9fa'
                }
            )
        ], style={'marginBottom': '20px'}),
        
        # Execute button
        html.Div([
            html.Button(
                'Execute Query',
                id='execute-query-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#E57200',
                    'color': 'white',
                    'padding': '12px 30px',
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer',
                    'marginRight': '10px'
                }
            ),
            html.Button(
                'Clear',
                id='clear-query-btn',
                n_clicks=0,
                style={
                    'backgroundColor': '#95a5a6',
                    'color': 'white',
                    'padding': '12px 30px',
                    'fontSize': '16px',
                    'fontWeight': '600',
                    'border': 'none',
                    'borderRadius': '4px',
                    'cursor': 'pointer'
                }
            )
        ], style={'marginBottom': '20px'}),
        
        # Results section
        html.Div(id='query-results', style={'marginTop': '30px'}),
        
        # Database schema reference
        html.Div([
            html.Hr(style={'margin': '40px 0'}),
            html.H4("Database Schema Reference", style={'color': '#232D4B'}),
            
            html.Div([
                html.Div([
                    html.H5("Tables:", style={'color': '#E57200'}),
                    html.Ul([
                        html.Li([html.Strong("materials"), " - Main table with 15,845 materials"]),
                        html.Li([html.Strong("categories"), " - 18 material categories"]),
                        html.Li([html.Strong("families"), " - 5 material families"]),
                        html.Li([html.Strong("quality_tiers"), " - 2 quality levels"]),
                        html.Li([html.Strong("elements"), " - 90 unique elements"]),
                        html.Li([html.Strong("materials_elements"), " - Junction table (65,781 relationships)"]),
                        html.Li([html.Strong("element_stats"), " - Aggregated element statistics"]),
                        html.Li([html.Strong("feature_importance"), " - Top features for Tc prediction"])
                    ])
                ], style={'flex': '1'}),
                
                html.Div([
                    html.H5("Key Columns:", style={'color': '#E57200'}),
                    html.Ul([
                        html.Li([html.Code("materials.tc_kelvin"), " - Critical temperature"]),
                        html.Li([html.Code("materials.chemical_formula"), " - Material formula"]),
                        html.Li([html.Code("materials.n_elements"), " - Number of elements"]),
                        html.Li([html.Code("materials.is_high_tc"), " - High-Tc flag (>77K)"]),
                        html.Li([html.Code("categories.category_name"), " - Material category"]),
                        html.Li([html.Code("families.family_name"), " - Material family"]),
                        html.Li([html.Code("elements.element_symbol"), " - Element symbol"])
                    ])
                ], style={'flex': '1'})
            ], style={'display': 'flex', 'gap': '40px'})
            
        ], style={
            'backgroundColor': '#f8f9fa',
            'padding': '20px',
            'borderRadius': '8px',
            'marginTop': '30px'
        })
        
    ], style={'padding': '20px'})



# CALLBACKS FOR SQL TAB

@app.callback(
    Output('sql-query-input', 'value'),
    [Input('example-query-dropdown', 'value'),
     Input('clear-query-btn', 'n_clicks')],
    [State('sql-query-input', 'value')]
)
def update_query_input(example_query, clear_clicks, current_query):
    """Update SQL input from example or clear it"""
    ctx = callback_context
    
    if not ctx.triggered:
        return current_query or ''
    
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    if trigger_id == 'example-query-dropdown' and example_query:
        return example_query
    elif trigger_id == 'clear-query-btn':
        return ''
    
    return current_query or ''


@app.callback(
    Output('query-results', 'children'),
    Input('execute-query-btn', 'n_clicks'),
    State('sql-query-input', 'value')
)
def execute_sql_query(n_clicks, query):
    """Execute SQL query and display results"""
    
    if not n_clicks or not query:
        return html.Div([
            html.P("👆 Enter a SQL query above and click 'Execute Query' to see results",
                  style={'textAlign': 'center', 'color': '#7f8c8d', 'fontSize': '16px', 'padding': '40px'})
        ])
    
    try:
        # Get database connection
        engine = get_db_connection()
        
        # Security: Only allow SELECT queries
        query_upper = query.strip().upper()
        if not query_upper.startswith('SELECT'):
            return html.Div([
                html.Div([
                    html.H4("Error: Only SELECT queries allowed", style={'color': '#e74c3c'}),
                    html.P("For security reasons, only SELECT statements are permitted. No INSERT, UPDATE, DELETE, DROP, etc.")
                ], style={
                    'backgroundColor': '#ffe6e6',
                    'padding': '20px',
                    'borderRadius': '4px',
                    'border': '2px solid #e74c3c'
                })
            ])
        
        # Execute query
        result_df = pd.read_sql(query, engine)
        
        # Success message
        success_msg = html.Div([
            html.Span("Query executed successfully! ", style={'color': '#27ae60', 'fontWeight': 'bold'}),
            html.Span(f"Returned {len(result_df)} rows, {len(result_df.columns)} columns")
        ], style={
            'backgroundColor': '#d5f4e6',
            'padding': '15px',
            'borderRadius': '4px',
            'marginBottom': '20px',
            'border': '2px solid #27ae60'
        })
        
        # Create DataTable
        if len(result_df) == 0:
            data_table = html.P("No results returned.", style={'textAlign': 'center', 'color': '#7f8c8d', 'padding': '20px'})
        else:
            data_table = dash_table.DataTable(
                data=result_df.to_dict('records'),
                columns=[{'name': col, 'id': col} for col in result_df.columns],
                page_size=20,
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'left',
                    'padding': '10px',
                    'fontFamily': 'EB Garamond, Garamond, serif',
                    'fontSize': '14px',
                    'minWidth': '100px',
                    'maxWidth': '300px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis'
                },
                style_header={
                    'backgroundColor': '#232D4B',
                    'color': 'white',
                    'fontWeight': '600',
                    'borderBottom': '3px solid #E57200',
                    'fontSize': '15px'
                },
                style_data={
                    'backgroundColor': 'white',
                    'color': '#232D4B',
                    'borderBottom': '1px solid #ddd'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ],
                export_format='csv',
                export_headers='display'
            )
        
        return html.Div([
            success_msg,
            data_table
        ])
        
    except Exception as e:
        # Error handling
        error_msg = str(e)
        return html.Div([
            html.Div([
                html.H4("Query Error", style={'color': '#e74c3c'}),
                html.P(error_msg, style={'fontFamily': 'Monaco, Courier, monospace', 'fontSize': '14px'}),
                html.Hr(),
                html.P("Tips:", style={'fontWeight': 'bold'}),
                html.Ul([
                    html.Li("Check your SQL syntax"),
                    html.Li("Verify table and column names"),
                    html.Li("Use the schema reference below"),
                    html.Li("Try one of the example queries")
                ])
            ], style={
                'backgroundColor': '#ffe6e6',
                'padding': '20px',
                'borderRadius': '4px',
                'border': '2px solid #e74c3c'
            })
        ])

# TAB 7: DATABASE SCHEMA
def create_schema_tab():
    """Database schema information"""
    
    return html.Div([
        html.H2("3NF Database Schema",
               style={'textAlign': 'center', 'marginTop': '30px', 'color': '#232D4B'}),
        
        html.Div([
            html.H3("View Interactive ER Diagram", style={'color': '#232D4B', 'marginBottom': '20px'}),
            html.P("Our database follows strict Third Normal Form (3NF) with 8 normalized tables and proper foreign key relationships.",
                  style={'fontSize': '16px', 'marginBottom': '30px'}),
            
            html.A(
                html.Button(
                    "Open dbdiagram.io Visualizer",
                    style={
                        'backgroundColor': '#232D4B',
                        'color': 'white',
                        'padding': '15px 30px',
                        'fontSize': '18px',
                        'fontWeight': '600',
                        'border': 'none',
                        'borderRadius': '4px',
                        'cursor': 'pointer',
                        'fontFamily': 'EB Garamond, Garamond, serif',
                        'boxShadow': '0 4px 6px rgba(0,0,0,0.2)',
                        'marginTop': '20px',
                        'display': 'inline-block'
                    }
                ),
                href="https://dbdiagram.io/d",
                target="_blank"
            ),
            
            html.Div([
                html.H4("How to View:", style={'marginTop': '30px', 'color': '#232D4B'}),
                html.Ol([
                    html.Li("Click the button above to open dbdiagram.io"),
                    html.Li("Click 'Import' → 'From DBML'"),
                    html.Li("Upload the file: database_schema_3NF.dbml"),
                    html.Li("Explore the interactive ER diagram!")
                ], style={'textAlign': 'left', 'maxWidth': '600px', 'margin': '0 auto', 'fontSize': '16px'})
            ]),
            
            html.Div([
                html.H4("Database Structure:", style={'marginTop': '30px', 'color': '#232D4B'}),
                html.Ul([
                    html.Li("8 normalized tables in 3NF"),
                    html.Li("Foreign key relationships"),
                    html.Li("15,845 materials with 65,781 material-element relationships"),
                    html.Li("90 unique elements, 18 categories, 5 material families"),
                    html.Li("Junction table for many-to-many relationships")
                ], style={'textAlign': 'left', 'maxWidth': '600px', 'margin': '20px auto', 'fontSize': '16px'})
            ])
            
        ], style={
            'maxWidth': '800px',
            'margin': '0 auto',
            'padding': '40px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '8px',
            'boxShadow': '0 2px 8px rgba(0,0,0,0.1)'
        })
        
    ], style={'padding': '20px'})


# APP LAYOUT
app.layout = html.Div([
    html.H1("Superconductor Materials Database Dashboard",
            style={'textAlign': 'center', 'padding': '20px', 'backgroundColor': '#232D4B',
                   'color': 'white', 'marginBottom': '0'}),
    
    dcc.Tabs([
        dcc.Tab(label='Overview', children=[create_overview_tab(df)]),
        dcc.Tab(label='Element Explorer', children=[create_elements_tab(df)]),
        dcc.Tab(label='Composition Trends', children=[create_trends_tab(df)]),
        dcc.Tab(label='Data Quality', children=[create_quality_tab(df)]),
        dcc.Tab(label='Feature Importance', children=[create_features_tab(df, top_features)]),
        dcc.Tab(label='SQL Playground', children=[create_sql_tab()]),
        dcc.Tab(label='Database Schema', children=[create_schema_tab()])
    ]),
    
    # Footer
    html.Footer([
        html.Hr(style={'margin': '40px 0 20px 0'}),
        html.Div([
            html.P([
                "Data source: ",
                html.A("NIMS Superconducting Materials Database",
                      href="https://mdr.nims.go.jp/collections/4c428a0c-d209-4990-ad1f-656d05d1cfe2",
                      target="_blank",
                      style={'color': '#E57200'}),
                " | Features: matminer elemental properties"
            ], style={'textAlign': 'center', 'color': '#666', 'marginBottom': '10px'}),
            html.P("Dashboard by Ankita Biswas | University of Virginia | 2025",
                  style={'textAlign': 'center', 'color': '#999', 'fontSize': '14px'})
        ], style={'padding': '20px'})
    ], style={'backgroundColor': '#f8f9fa', 'marginTop': '40px'})
])

if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8050)
