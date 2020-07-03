import pandas as pd
import numpy as np
import pickle
from sklearn import linear_model
import scipy.stats as stat
import model_def
pd.options.display.max_columns = None
#import requests

import dash
import dash_core_components as dcc
from textwrap import dedent
#import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
import dash_html_components as html

ENV = 'dev'
external_stylesheets = ['https://codepen.io/davifoga/pen/jOWYyyG.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)



if ENV == 'dev':
    debug = True # True ??
    server = app.server # ??
    # loan_data_inputs_pd_temp = pd.read_csv('D:\Davide\loan_data_inputs_test.csv')
    file_pd = 'pd_model.sav'
    file_st_1 = 'lgd_model_stage_1.sav'
    file_st_2 = 'lgd_model_stage_2.sav'
    file_ead = 'reg_ead.sav'
    reg_pd = pickle.load(open(file_pd, 'rb'))
    reg_lgd_st_1 = pickle.load(open(file_st_1, 'rb'))
    reg_lgd_st_2 = pickle.load(open(file_st_2, 'rb'))
    reg_ead = pickle.load(open(file_ead, 'rb'))

    """file_pd = 'pd_model.sav'
    file_st_1 = 'lgd_model_stage_1.sav'
    file_st_2 = 'lgd_model_stage_2.sav'
    file_ead = 'reg_ead.sav'
    with open(file_pd, 'rb') as file:
        reg_pd = pickle.load(file)
    with open(file_st_1, 'rb') as file:
        reg_lgd_st_1 = pickle.load(file)
    with open(file_st_2, 'rb') as file:
        reg_lgd_st_2 = pickle.load(file)
    with open(file_ead, 'rb') as file:
        reg_ead = pickle.load(file)"""



else:
    """server = app.server
    file_pd = 'https://credit-df.s3.eu-north-1.amazonaws.com/pd_model.sav'
    file_st_1 = 'https://credit-df.s3.eu-north-1.amazonaws.com/lgd_model_stage_1.sav'
    file_st_2 = 'https://credit-df.s3.eu-north-1.amazonaws.com/lgd_model_stage_2.sav'
    file_ead = 'https://credit-df.s3.eu-north-1.amazonaws.com/reg_ead.sav'
    reg_pd = requests.get(file_pd)
    reg_lgd_st_1 = requests.get(file_st_1)
    reg_lgd_st_2 = requests.get(file_st_2)
    reg_ead = requests.get(file_ead)"""
    debug = False



features_all_pd = ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86']

ref_categories_pd = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'inq_last_6mths:>6',
'acc_now_delinq:0',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

features_all = ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:MORTGAGE',
'home_ownership:NONE',
'home_ownership:OTHER',
'home_ownership:OWN',
'home_ownership:RENT',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:car',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:educational',
'purpose:home_improvement',
'purpose:house',
'purpose:major_purchase',
'purpose:medical',
'purpose:moving',
'purpose:other',
'purpose:renewable_energy',
'purpose:small_business',
'purpose:vacation',
'purpose:wedding',
'initial_list_status:f',
'initial_list_status:w',
'term_int',
'emp_length_int',
'mths_since_issue_d',
'mths_since_earliest_cr_line',
'funded_amnt',
'int_rate',
'installment',
'annual_inc',
'dti',
'delinq_2yrs',
'inq_last_6mths',
'mths_since_last_delinq',
'mths_since_last_record',
'open_acc',
'pub_rec',
'total_acc',
'acc_now_delinq',
'total_rev_hi_lim']
# List of all independent variables for the models.

features_reference_cat = ['grade:G',
'home_ownership:RENT',
'verification_status:Verified',
'purpose:credit_card',
'initial_list_status:f']
# List of the dummy variable reference categories.

features = features_all + features_all_pd
features = np.array(features)
features = list(np.unique(features))

feature_ownership = ['Mortgage', 'Own', 'Rent', 'Other', 'None']
feature_purpose = ['Credit card', 'Car', 'Debt consolidation', 'Educational', 'Home improvement', 'House', 'Major purchase', 'Medical', 'Moving',
                   'Other', 'Renewable energy', 'Small business', 'Vacation', 'Wedding']
feature_term = [36, 60]
feature_address = ['ND', 'NE', 'IA', 'NV', 'FL', 'HI', 'AL', 'NM', 'VA', 'NY', 'OK', 'TN', 'MO', 'LA', 'MD', 'NC', 'CA', 'UT', 'KY', 'AZ', 'NJ', 'AR', 'MI', 'PA', 'OH', 'MN',
                   'RI', 'MA', 'DE', 'SD', 'IN', 'GA', 'WA', 'OR', 'WI', 'MT', 'TX', 'IL', 'CT', 'KS', 'SC', 'CO', 'VT', 'AK', 'MS', 'WV', 'NH', 'WY', 'DC', 'ME', 'ID']
feature_verification = ['Not Verified', 'Source Verified', 'Verified']
feature_initial_status = ['Fractional', 'Whole']
feature_grade = ['A', 'B', 'C', 'D', 'E', 'F', 'G']


style_data_applicant = {'backgroundColor':'#efefed',
                        'margin':'1%',
                        'margin-left':'0%',
                        'textAlign': 'left',
                        'fontSize':11,
                        #'width':'90%'
                         }

style_data_applicant_text = {'backgroundColor':'#00005e',
                             'margin':'1%',
                             'margin-left':'0%',
                             'textAlign': 'left',
                             'color':'#efefed',
                             'fontSize':14,
                             #'width':'90%'
                         }

style_output = {'backgroundColor':'#dcedff',
                 'margin':'0%',
                 'margin-left':'0%',
                 'textAlign': 'left',
                 'color':'#00005e',
                 'fontSize':16,
                         }

style_output_text = {'backgroundColor':'#dcedff',
                 'margin':'.5%',
                 'margin-left':'2%',
                 'textAlign': 'left',
                 'color':'#00005e',
                 'fontSize':16,
                         }

style_output_subtitles = {'backgroundColor':'#dcedff',
                         'margin':'.5%',
                         'margin-left':'1%',
                         'textAlign': 'left',
                         'color':'#00005e',
                         'fontSize':22,
                                 }

style_name = {'backgroundColor':'#efefed',
                 'margin':'.5%',
                 'margin-left':'0%',
                 'textAlign': 'left',
                 'color':'#00005e',
                 'fontSize':14,
                         }

style_subtitles = {'backgroundColor':'#00005e',
                   'textAlign': 'center',
                   'color':'#a6d1ff',
                   'fontSize':22,
                   }

style_assumptions = style_data_applicant
style_assumptions_text = style_data_applicant_text

#'backgroundColor':'#00019f'

app.title = 'DF Credit'
app.layout = html.Div(children=[
    html.Header(children='Credit Request Evaluation',
            style={'textAlign': 'center',
                  'color': 'white',
                  'fontSize':40,
                  'backgroundColor':'#00005e'}),

    dcc.Markdown(dedent('''developed by [Davide Fogarolo](https://www.linkedin.com/in/davide-fogarolo)'''),
            style={'textAlign': 'center',
                  'color': 'white',
                  'fontSize':12,
                  'backgroundColor':'#00005e'}),

    html.Div([
        html.P("Please fill out all the requested fields below in order to have an average estimate of the applicant's EAD, PD, LGD, and EL", id="intro", style=style_output_subtitles,
               className='one row'),

        html.Div([html.P("Applicant's full name:", style=style_output_text, className = 'two columns'),
                 dcc.Input(id="input_0", type="text", placeholder="Type full name", style=style_name, className='two columns')],
                 className = 'one row'),

        html.P('The Expected Loss for {} is {} USD.'.format(' [Full Name] ', ' [Complete the form] '), id="expected_loss", style=style_output_text,
               className='one row'),

        html.P('The Probability of Default is {}.'.format(' [Complete the form]'), id="pd", style=style_output_text,
               className='one row'),

        html.P('The Loss Given Default is {}.'.format(' [Complete the form]'), id="lgd", style=style_output_text,
               className='one row'),

        html.P('The Exposure at Default is USD {}.'.format( ' [Complete the form]'), id="ead", style=style_output_text,
               className='one row'),


    ], style=style_output, className='one row'),

    html.Div([
        html.H1(children='Information provided by the applicant', style=style_subtitles, className='one row'),

        html.Div([html.P("Purpose of the credit request:", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Dropdown(id="input_1", placeholder="Choose purpose", options=[{"label": i, "value": i} for i in feature_purpose],
                              multi=False, style= style_data_applicant, className = 'five columns')],
                 className = 'one row'),

        html.Div([html.P("Amount requested for the loan (USD):", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Input(id="input_2", type="number", placeholder="Type the amount", style= style_data_applicant, className = 'five columns')],
                 className = 'one row'),

        html.Div([html.P("Length of the loan term (months):", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Dropdown(id="input_3", placeholder="Choose the length", options=[{"label": i, "value": i} for i in feature_term],
                              multi=False, style= style_data_applicant, className = 'five columns')],
                 className = 'one row'),

        html.Div([html.P("Applicant's annual income (USD):", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Input(id="input_4", type="number", placeholder="Type annual income", style= style_data_applicant, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Employment length (years):", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Input(id="input_5", type="number", placeholder="Type the n° of years", style= style_data_applicant, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Applicant's home contract:", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Dropdown(id="input_6", placeholder="Choose the home contract", options=[{"label": i, "value": i} for i in feature_ownership],
                              multi=False, style= style_data_applicant, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Applicant's State Address:", style=style_data_applicant_text, className = 'six columns'),
                 dcc.Dropdown(id="input_7", placeholder="Choose the State", options=[{"label": i, "value": i} for i in feature_address],
                              multi=False, style= style_data_applicant, className = 'five columns')],
                 className='one row'),

    ], style={'margin':'3%'}, className='five columns'),


    html.Div([
        html.H1(children='Assumptions regarding information in proprietary databases', style=style_subtitles, className='one row'),

        html.Div([html.P("Credit Grade of the applicant:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Dropdown(id="assumption_1", placeholder="Choose the credit grade", options=[{"label": i, "value": i} for i in feature_grade],
                              multi=False, style=style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Verification status of the applicant:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Dropdown(id="assumption_2", placeholder="Choose the status", options=[{"label": i, "value": i} for i in feature_verification],
                              multi=False, style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("List status of the applicant:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Dropdown(id="assumption_3", placeholder="Choose the status", options=[{"label": i, "value": i} for i in feature_initial_status],
                              multi=False, style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("N° of months since the loan has been issued:", style=style_assumptions_text, className = 'six columns'), # delete, and leave 0 per default ??
                 dcc.Input(id="assumption_4", type="number", placeholder="Type n° months", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Interest rate (%):", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_5", type="number", placeholder="Type the IR", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Period since the earliest credit line (months):", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_6", type="number", placeholder="Type n° months", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of inquiries in the last 6 months:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_7", type="number", placeholder="Type n° months", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of  accounts on which the applicant is now delinquent:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_8", type="number", placeholder="Type n° delinquent accounts ", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of months since last delinquency:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_9", type="number", placeholder="Type n° months", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("N° of 30+ days past-due incidences of delinquency for the applicant (past 2 years):", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_10", type="number", placeholder="Type n°", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of months since the last public record:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_11", type="number", placeholder="Type n° months", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of open credit lines in the applicant's credit file:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_12", type="number", placeholder="Type n° credit lines", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Number of the applicant's derogatory public records:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_13", type="number", placeholder="Type n° records", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Total number of credit lines currently in the applicant's credit file:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_14", type="number", placeholder="Type n° credit lines", style= style_assumptions, className = 'five columns')],
                 className='one row'),

        html.Div([html.P("Total revolving high credit/credit limit:", style=style_assumptions_text, className = 'six columns'),
                 dcc.Input(id="assumption_15", type="number", placeholder="Type limit", style= style_assumptions, className = 'five columns')],
                 className='one row'),

    ], style={'margin':'3%'}, className='five columns'),



    ])


""" CALLBACKS """
@app.callback([Output('expected_loss','children'),
               Output('pd','children'),
               Output('lgd','children'),
               Output('ead','children'),
               ],
        [Input('input_0','value'),
         Input('input_1', 'value'),
         Input('input_2', 'value'),
         Input('input_3', 'value'),
         Input('input_4', 'value'),
         Input('input_5', 'value'),
         Input('input_6', 'value'),
         Input('input_7', 'value'),

         Input('assumption_1', 'value'),
         Input('assumption_2', 'value'),
         Input('assumption_3', 'value'),
         Input('assumption_4', 'value'),
         Input('assumption_5', 'value'),
         Input('assumption_6', 'value'),
         Input('assumption_7', 'value'),
         Input('assumption_8', 'value'),
         Input('assumption_9', 'value'),
         Input('assumption_10', 'value'),
         Input('assumption_11', 'value'),
         Input('assumption_12', 'value'),
         Input('assumption_13', 'value'),
         Input('assumption_14', 'value'),
         Input('assumption_15', 'value')])
def update_time_slider(value_0, value_1, value_2, value_3, value_4, value_5, value_6, value_7,
                       assumption_1, assumption_2, assumption_3, assumption_4, assumption_5, assumption_6, assumption_7, assumption_8, assumption_9,
                       assumption_10, assumption_11, assumption_12, assumption_13, assumption_14, assumption_15
                       ):

    data = [[0]*134]
    df = pd.DataFrame(columns=features, data=data)

    try:
        if value_1 == 'Car':
            df['purpose:car'] = np.where((df['purpose:car'] == 0), 1, 0)
            df['purpose:major_purch__car__home_impr'] = np.where((df['purpose:major_purch__car__home_impr'] == 0), 1, 0)
        elif value_1 == 'Debt consolidation':
            df['purpose:debt_consolidation'] = np.where((df['purpose:debt_consolidation'] == 0), 1, 0)
            #df['purpose:debt_consolidation'] = np.where((df['purpose:debt_consolidation'] == 0), 1, 0)
        elif value_1 == 'Educational':
            df['purpose:educational'] = np.where((df['purpose:educational'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Home improvement':
            df['purpose:home_improvement'] = np.where((df['purpose:home_improvement'] == 0), 1, 0)
            df['purpose:major_purch__car__home_impr'] = np.where((df['purpose:major_purch__car__home_impr'] == 0), 1, 0)
        elif value_1 == 'House':
            df['purpose:house'] = np.where((df['purpose:house'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Major purchase':
            df['purpose:major_purchase'] = np.where((df['purpose:major_purchase'] == 0), 1, 0)
            df['purpose:major_purch__car__home_impr'] = np.where((df['purpose:major_purch__car__home_impr'] == 0), 1, 0)
        elif value_1 == 'Medical':
            df['purpose:medical'] = np.where((df['purpose:medical'] == 0), 1, 0)
            df['purpose:oth__med__vacation'] = np.where((df['purpose:oth__med__vacation'] == 0), 1, 0)
        elif value_1 == 'Moving':
            df['purpose:moving'] = np.where((df['purpose:moving'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Other':
            df['purpose:other'] = np.where((df['purpose:other'] == 0), 1, 0)
            df['purpose:oth__med__vacation'] = np.where((df['purpose:oth__med__vacation'] == 0), 1, 0)
        elif value_1 == 'Renewable energy':
            df['purpose:renewable_energy'] = np.where((df['purpose:renewable_energy'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Small business':
            df['purpose:small_business'] = np.where((df['purpose:small_business'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Vacation':
            df['purpose:vacation'] = np.where((df['purpose:vacation'] == 0), 1, 0)
            df['purpose:oth__med__vacation'] = np.where((df['purpose:oth__med__vacation'] == 0), 1, 0)
        elif value_1 == 'Wedding':
            df['purpose:wedding'] = np.where((df['purpose:wedding'] == 0), 1, 0)
            df['purpose:educ__sm_b__wedd__ren_en__mov__house'] = np.where((df['purpose:educ__sm_b__wedd__ren_en__mov__house'] == 0), 1, 0)
        elif value_1 == 'Credit card':
            pass
    except TypeError:
        pass

    try:
        df['funded_amnt'] = np.where((df['funded_amnt'] == 0), value_2, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        if value_3 == '36':
            df['term:36'] = np.where((df['term:36'] == 0), 1, 0)
        elif value_3 == '60':
            df['term:60'] = np.where((df['term:60'] == 0), 1, 0)
        df['term_int'] = np.where((df['term_int'] == 0), value_3, 0) # ??
    except TypeError:
        pass

    try:
        if value_4 <= 20000:
            df['annual_inc:<20K'] = np.where((df['annual_inc:<20K'] == 0), 1, 0)
        elif (value_4 > 20000) & (value_4 <=30000):
            df['annual_inc:20K-30K'] = np.where((df['annual_inc:20K-30K'] == 0), 1, 0)
        elif (value_4 > 30000) & (value_4 <=40000):
            df['annual_inc:30K-40K'] = np.where((df['annual_inc:30K-40K'] == 0), 1, 0)
        elif (value_4 > 40000) & (value_4 <=50000):
            df['annual_inc:40K-50K'] = np.where((df['annual_inc:40K-50K'] == 0), 1, 0)
        elif (value_4 > 50000) & (value_4 <=60000):
            df['annual_inc:50K-60K'] = np.where((df['annual_inc:50K-60K'] == 0), 1, 0)
        elif (value_4 > 60000) & (value_4 <=70000):
            df['annual_inc:60K-70K'] = np.where((df['annual_inc:60K-70K'] == 0), 1, 0)
        elif (value_4 > 70000) & (value_4 <=80000):
            df['annual_inc:70K-80K'] = np.where((df['annual_inc:70K-80K'] == 0), 1, 0)
        elif (value_4 > 80000) & (value_4 <=90000):
            df['annual_inc:80K-90K'] = np.where((df['annual_inc:80K-90K'] == 0), 1, 0)
        elif (value_4 > 90000) & (value_4 <=100000):
            df['annual_inc:90K-100K'] = np.where((df['annual_inc:90K-100K'] == 0), 1, 0)
        elif (value_4 > 100000) & (value_4 <=120000):
            df['annual_inc:100K-120K'] = np.where((df['annual_inc:100K-120K'] == 0), 1, 0)
        elif (value_4 > 120000) & (value_4 <=140000):
            df['annual_inc:120K-140K'] = np.where((df['annual_inc:120K-140K'] == 0), 1, 0)
        elif value_4 > 140000:
            df['annual_inc:>140K'] = np.where((df['annual_inc:>140K'] == 0), 1, 0)
        df['annual_inc'] = np.where((df['annual_inc'] == 0), value_4, 0) # ??
    except TypeError:
        pass

    try:
        if value_5 == 0:
            df['emp_length:0'] = np.where((df['emp_length:0'] == 0), 1, 0)
        elif value_5 == 1:
            df['emp_length:1'] = np.where((df['emp_length:1'] == 0), 1, 0)
        elif value_5 >= 2 & value_5 <= 4:
            df['emp_length:2-4'] = np.where((df['emp_length:2-4'] == 0), 1, 0)
        elif value_5 >= 5 & value_5 <= 6:
            df['emp_length:5-6'] = np.where((df['emp_length:5-6'] == 0), 1, 0)
        elif value_5 >= 7 & value_5 <= 9:
            df['emp_length:7-9'] = np.where((df['emp_length:7-9'] == 0), 1, 0)
        elif value_5 >= 10:
            df['emp_length:10'] = np.where((df['emp_length:10'] == 0), 1, 0)
        df['emp_length_int'] = np.where((df['emp_length_int'] == 0), value_5, 0) # ??
    except TypeError:
        pass

    try:
        if value_6 == 'Mortgage':
            df['home_ownership:MORTGAGE'] = np.where((df['home_ownership:MORTGAGE'] == 0), 1, 0)
            #df['home_ownership:MORTGAGE'] = np.where((df['home_ownership:MORTGAGE'] == 0), 1, 0)
        elif value_6 == 'Own':
            df['home_ownership:OWN'] = np.where((df['home_ownership:OWN'] == 0), 1, 0)
            #df['home_ownership:OWN'] = np.where((df['home_ownership:OWN'] == 0), 1, 0)
        elif value_6 == 'Other':
            df['home_ownership:OTHER'] = np.where((df['home_ownership:OTHER'] == 0), 1, 0)
            df['home_ownership:RENT_OTHER_NONE_ANY'] = np.where((df['home_ownership:RENT_OTHER_NONE_ANY'] == 0), 1, 0)
        elif value_6 == 'None':
            df['home_ownership:NONE'] = np.where((df['home_ownership:NONE'] == 0), 1, 0)
            df['home_ownership:RENT_OTHER_NONE_ANY'] = np.where((df['home_ownership:RENT_OTHER_NONE_ANY'] == 0), 1, 0)
        elif value_6 == 'Rent':
            pass
    except TypeError:
        pass

    try:
        if (value_7 == 'ND') or (value_7 == 'NE') or (value_7 == 'IA') or (value_7 == 'NV') or (value_7 == 'FL') or (value_7 == 'HI') or (value_7 == 'AL'):
            df['addr_state:ND_NE_IA_NV_FL_HI_AL'] = np.where((df['addr_state:ND_NE_IA_NV_FL_HI_AL'] == 0), 1, 0)
        elif (value_7 == 'NM') or (value_7 == 'VA'):
            df['addr_state:NM_VA'] = np.where((df['addr_state:NM_VA'] == 0), 1, 0)
        elif (value_7 == 'NY'):
            df['addr_state:NY'] = np.where((df['addr_state:NY'] == 0), 1, 0)
        elif (value_7 == 'OK') or (value_7 == 'TN') or (value_7 == 'MO') or (value_7 == 'LA') or (value_7 == 'MD') or (value_7 == 'NC'):
            df['addr_state:OK_TN_MO_LA_MD_NC'] = np.where((df['addr_state:OK_TN_MO_LA_MD_NC'] == 0), 1, 0)
        elif (value_7 == 'CA'):
            df['addr_state:CA'] = np.where((df['addr_state:CA'] == 0), 1, 0)
        elif (value_7 == 'UT') or (value_7 == 'KY') or (value_7 == 'AZ') or (value_7 == 'NJ'):
            df['addr_state:UT_KY_AZ_NJ'] = np.where((df['addr_state:UT_KY_AZ_NJ'] == 0), 1, 0)
        elif (value_7 == 'AR') or (value_7 == 'MI') or (value_7 == 'PA') or (value_7 == 'OH') or (value_7 == 'MN'):
            df['addr_state:AR_MI_PA_OH_MN'] = np.where((df['addr_state:AR_MI_PA_OH_MN'] == 0), 1, 0)
        elif (value_7 == 'RI') or (value_7 == 'MA') or (value_7 == 'DE') or (value_7 == 'SD') or (value_7 == 'IN'):
            df['addr_state:RI_MA_DE_SD_IN'] = np.where((df['addr_state:RI_MA_DE_SD_IN'] == 0), 1, 0)
        elif (value_7 == 'GA') or (value_7 == 'WA'):
            df['addr_state:GA_WA'] = np.where((df['addr_state:GA_WA'] == 0), 1, 0)
        elif (value_7 == 'WI') or (value_7 == 'MT'):
            df['addr_state:WI_MT'] = np.where((df['addr_state:WI_MT'] == 0), 1, 0)
        elif (value_7 == 'TX'):
            df['addr_state:TX'] = np.where((df['addr_state:TX'] == 0), 1, 0)
        elif (value_7 == 'IL') or (value_7 == 'CT'):
            df['addr_state:IL_CT'] = np.where((df['addr_state:IL_CT'] == 0), 1, 0)
        elif (value_7 == 'KS') or (value_7 == 'SC') or (value_7 == 'CO') or (value_7 == 'VT') or (value_7 == 'AK') or (value_7 == 'MS'):
            df['addr_state:KS_SC_CO_VT_AK_MS'] = np.where((df['addr_state:KS_SC_CO_VT_AK_MS'] == 0), 1, 0)
        elif (value_7 == 'WV') or (value_7 == 'NH') or (value_7 == 'WY') or (value_7 == 'DC') or (value_7 == 'ME') or (value_7 == 'ID'):
            df['addr_state:WV_NH_WY_DC_ME_ID'] = np.where((df['addr_state:WV_NH_WY_DC_ME_ID'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_1 == 'A':
            df['grade:A'] = np.where((df['grade:A'] == 0), 1, 0)
        elif assumption_1 == 'B':
            df['grade:B'] = np.where((df['grade:B'] == 0), 1, 0)
        elif assumption_1 == 'C':
            df['grade:C'] = np.where((df['grade:C'] == 0), 1, 0)
        elif assumption_1 == 'D':
            df['grade:D'] = np.where((df['grade:D'] == 0), 1, 0)
        elif assumption_1 == 'E':
            df['grade:E'] = np.where((df['grade:E'] == 0), 1, 0)
        elif assumption_1 == 'F':
            df['grade:F'] = np.where((df['grade:F'] == 0), 1, 0)
        elif assumption_1 == 'G':
            df['grade:G'] = np.where((df['grade:G'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_2 == 'Not Verified':
            df['verification_status:Not Verified'] = np.where((df['verification_status:Not Verified'] == 0), 1, 0)
        elif assumption_2 == 'Source Verified':
            df['verification_status:Source Verified'] = np.where((df['verification_status:Source Verified'] == 0), 1, 0)
        elif assumption_2 == 'Verified':
            df['verification_status:Verified'] = np.where((df['verification_status:Verified'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_3 == 'Fractional':
            df['initial_list_status:f'] = np.where((df['initial_list_status:f'] == 0), 1, 0)
        elif assumption_3 == 'Whole':
            df['initial_list_status:w'] = np.where((df['initial_list_status:w'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_4 <= 38:
            df['mths_since_issue_d:<38'] = np.where((df['mths_since_issue_d:<38'] == 0), 1, 0)
        elif assumption_4 > 38 & assumption_4 <= 39:
            df['mths_since_issue_d:38-39'] = np.where((df['mths_since_issue_d:38-39'] == 0), 1, 0)
        elif assumption_4 > 39  & assumption_4 <= 41:
            df['mths_since_issue_d:40-41'] = np.where((df['mths_since_issue_d:40-41'] == 0), 1, 0)
        elif assumption_4 > 41 & assumption_4 <= 48:
            df['mths_since_issue_d:42-48'] = np.where((df['mths_since_issue_d:42-48'] == 0), 1, 0)
        elif assumption_4 > 48 & assumption_4 <= 52:
            df['mths_since_issue_d:49-52'] = np.where((df['mths_since_issue_d:49-52'] == 0), 1, 0)
        elif assumption_4 > 52 & assumption_4 <= 64:
            df['mths_since_issue_d:53-64'] = np.where((df['mths_since_issue_d:53-64'] == 0), 1, 0)
        elif assumption_4 > 64 & assumption_4 <= 84:
            df['mths_since_issue_d:65-84'] = np.where((df['mths_since_issue_d:65-84'] == 0), 1, 0)
        elif assumption_4 > 84:
            df['mths_since_issue_d:>84'] = np.where((df['mths_since_issue_d:>84'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_5 <= 9.548:
            df['int_rate:<9.548'] = np.where((df['int_rate:<9.548'] == 0), 1, 0)
        elif assumption_5 > 9.548 & assumption_5 <= 12.025:
            df['int_rate:9.548-12.025'] = np.where((df['int_rate:9.548-12.025'] == 0), 1, 0)
        elif assumption_5 > 12.025 & assumption_5 <= 15.74:
            df['int_rate:12.025-15.74'] = np.where((df['int_rate:12.025-15.74'] == 0), 1, 0)
        elif assumption_5 > 15.74 & assumption_5 <= 20.281:
            df['int_rate:15.74-20.281'] = np.where((df['int_rate:15.74-20.281'] == 0), 1, 0)
        elif assumption_5 > 20.281:
            df['int_rate:>20.281'] = np.where((df['int_rate:>20.281'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_6 <= 140:
            df['mths_since_earliest_cr_line:<140'] = np.where((df['mths_since_earliest_cr_line:<140'] == 0), 1, 0)
        elif assumption_6 > 140 & assumption_6 <= 164:
            df['mths_since_earliest_cr_line:141-164'] = np.where((df['mths_since_earliest_cr_line:141-164'] == 0), 1, 0)
        elif assumption_6 > 164 & assumption_6 <= 247:
            df['mths_since_earliest_cr_line:165-247'] = np.where((df['mths_since_earliest_cr_line:165-247'] == 0), 1, 0)
        elif assumption_6 > 247 & assumption_6 <= 270:
            df['mths_since_earliest_cr_line:248-270'] = np.where((df['mths_since_earliest_cr_line:248-270'] == 0), 1, 0)
        elif assumption_6 > 270 & assumption_6 <= 352:
            df['mths_since_earliest_cr_line:271-352'] = np.where((df['mths_since_earliest_cr_line:271-352'] == 0), 1, 0)
        elif assumption_6 > 352:
            df['mths_since_earliest_cr_line:>352'] = np.where((df['mths_since_earliest_cr_line:>352'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        if assumption_7 == 0:
            df['inq_last_6mths:0'] = np.where((df['inq_last_6mths:0'] == 0), 1, 0)
        elif assumption_7 > 0 & assumption_7 <= 2:
            df['inq_last_6mths:1-2'] = np.where((df['inq_last_6mths:1-2'] == 0), 1, 0)
        elif assumption_7 > 2 & assumption_7 <= 6:
            df['inq_last_6mths:3-6'] = np.where((df['inq_last_6mths:3-6'] == 0), 1, 0)
        elif assumption_7 > 6:
            df['inq_last_6mths:>6'] = np.where((df['inq_last_6mths:>6'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        df['acc_now_delinq:0'] = np.where((assumption_8 == 0), 1, 0)
        df['acc_now_delinq:>=1'] = np.where((assumption_8 >= 1), 1, 0)
        df['acc_now_delinq'] = np.where((df['acc_now_delinq'] == 0), assumption_8, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        if assumption_9 == 0:
            df['mths_since_last_delinq:Missing'] = np.where((df['mths_since_last_delinq:Missing'] == 0), 1, 0)
        elif assumption_9 > 0 & assumption_9 <= 3:
            df['mths_since_last_delinq:0-3'] = np.where((df['mths_since_last_delinq:0-3'] == 0), 1, 0)
        elif assumption_9 > 3 & assumption_9 <= 30:
            df['mths_since_last_delinq:4-30'] = np.where((df['mths_since_last_delinq:4-30'] == 0), 1, 0)
        elif assumption_9 > 30 & assumption_9 <= 56:
            df['mths_since_last_delinq:31-56'] = np.where((df['mths_since_last_delinq:31-56'] == 0), 1, 0)
        elif assumption_9 > 56:
            df['mths_since_last_delinq:>=57'] = np.where((df['mths_since_last_delinq:>=57'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        df['delinq_2yrs'] = np.where((df['delinq_2yrs'] == 0), assumption_10, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        if assumption_11 == 0:
            df['mths_since_last_record:Missing'] = np.where((df['mths_since_last_record:Missing'] == 0), 1, 0)
        elif assumption_11 > 0 & assumption_11 <= 2:
            df['mths_since_last_record:0-2'] = np.where((df['mths_since_last_record:0-2'] == 0), 1, 0)
        elif assumption_11 > 2 & assumption_11 <= 20:
            df['mths_since_last_record:3-20'] = np.where((df['mths_since_last_record:3-20'] == 0), 1, 0)
        elif assumption_11 > 20 & assumption_11 <= 31:
            df['mths_since_last_record:21-31'] = np.where((df['mths_since_last_record:21-31'] == 0), 1, 0)
        elif assumption_11 > 31 & assumption_11 <= 80:
            df['mths_since_last_record:32-80'] = np.where((df['mths_since_last_record:32-80'] == 0), 1, 0)
        elif assumption_11 > 80 & assumption_11 <= 86:
            df['mths_since_last_record:81-86'] = np.where((df['mths_since_last_record:81-86'] == 0), 1, 0)
        elif assumption_11 > 86:
            df['mths_since_last_record:>86'] = np.where((df['mths_since_last_record:>86'] == 0), 1, 0)
    except TypeError:
        pass

    try:
        df['open_acc'] = np.where((df['open_acc'] == 0), assumption_12, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        df['pub_rec'] = np.where((df['pub_rec'] == 0), assumption_13, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        df['total_acc'] = np.where((df['total_acc'] == 0), assumption_14, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    try:
        df['total_rev_hi_lim'] = np.where((df['total_rev_hi_lim'] == 0), assumption_15, 0) # possibilmente potrebbe dover essre rewrite ??
    except TypeError:
        pass

    loan_data_inputs_pd_temp = df[features_all_pd]
    loan_data_inputs_pd_temp = loan_data_inputs_pd_temp.drop(ref_categories_pd, axis = 1)
    df['PD'] = reg_pd.model.predict_proba(loan_data_inputs_pd_temp)[: ][: , 0]

    loan_data_preprocessed_lgd_ead = df[features_all]
    loan_data_preprocessed_lgd_ead = loan_data_preprocessed_lgd_ead.drop(features_reference_cat, axis = 1)
    df['recovery_rate_st_1'] = reg_lgd_st_1.model.predict(loan_data_preprocessed_lgd_ead)
    df['recovery_rate_st_2'] = reg_lgd_st_2.predict(loan_data_preprocessed_lgd_ead)
    df['recovery_rate'] = df['recovery_rate_st_1'] * df['recovery_rate_st_2']
    df['recovery_rate'] = np.where(df['recovery_rate'] < 0, 0, df['recovery_rate'])
    df['recovery_rate'] = np.where(df['recovery_rate'] > 1, 1, df['recovery_rate'])
    df['LGD'] = 1 - df['recovery_rate']
    df['CCF'] = reg_ead.predict(loan_data_preprocessed_lgd_ead)
    df['CCF'] = np.where(df['CCF'] < 0, 0, df['CCF'])
    df['CCF'] = np.where(df['CCF'] > 1, 1, df['CCF'])
    df['EAD'] = df['funded_amnt'] * df['CCF']

    try:
        df['EL'] = df['PD'] * df['LGD'] * df['EAD']
        EL = round(df['EL'].iloc[-1])
        PD = round(df['PD'].iloc[-1], ndigits=2)
        LGD = round(df['LGD'].iloc[-1], ndigits=2)
        EAD = round(df['EAD'].iloc[-1])
    except IndexError:
        pass

    expected_loss = 'The Expected Loss for {} is USD {}.'.format(value_0, EL)
    PD = 'The Probability of Default is {0:.1%}.'.format(PD)
    lgd = 'The Loss Given Default is {0:.1%}.'.format(LGD)
    ead = 'The Exposure at Default is USD {}.'.format(EAD)


    return expected_loss, PD, lgd, ead




if __name__ == '__main__':
    app.run_server(debug=debug)
