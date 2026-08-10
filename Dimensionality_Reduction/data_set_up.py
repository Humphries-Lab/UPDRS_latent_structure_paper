import os, sys
import numpy as np
import datetime as dt
import data_formatting_functions as data_fun
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer




def find_boundary(new_list):
    change_positions = []
    change_vector = []

    # Iterate through the list and compare each element with the next one
    for i in range(len(new_list) - 1):
        if new_list[i] != new_list[i + 1]:
            change_positions.append(i + 1)
            change_vector.append(1)
        else:
            change_vector.append(0)

    # Add 1 at the beginning if the first element is not '0'
    if new_list and new_list[0] != '0':
        change_positions.insert(0, 0)
        change_vector.insert(0, 1)
    change_positions.append(len(new_list))
    return change_positions, change_vector


def data_pre_process_selected_patients(patient_list):
    if patient_list is None:
        raise ValueError("You must provide a 'patient_list' to use this function.")
    
   
    new_test_score_idx=[]
    new_test_score=0
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    motor_dir = os.path.join(base_dir, 'data', 'Motor___MDS-UPDRS')
    non_motor_dir = os.path.join(base_dir, 'data', 'Non-motor_Assessments')
    file_columns = {
        "MDS_UPDRS_Part_II__Patient_Questionnaire.csv": ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN",
                                                        "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"],
        
        "MDS-UPDRS_Part_I_Patient_Questionnaire.csv": ["NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"],

        "MDS-UPDRS_Part_I.csv": ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"],

        "MDS-UPDRS_Part_III.csv": ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR",\
                                "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3PSTBL",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NHY"],


        "Modified_Schwab___England_Activities_of_Daily_Living.csv": ["MSEADLG"],

    }

    #Choose the PD patients from the "Consensus_Committee_Analytic_Datasets_28OCT21.csv" in the PPMI dataset in folder 'Quick_Start'


    # patient_list = data_fun.select_patient_list('Consensus_Committee_Analytic_Datasets_23Sep2022.xlsx')
    
    # Initialize an empty list and data frame to store the data matrices and dataframes from each file
    data_matrix_list = []
    dataframes = pd.DataFrame()
    # Initialize an empty list to store the file names
    test_names = []
    for filename in os.listdir(motor_dir):
        # Construct the path to the current file
        file_path = os.path.join(motor_dir, filename)

        print(filename)
        df = pd.read_csv(file_path)
        new_df = pd.DataFrame(np.nan, index=np.arange(len(patient_list)), columns=df.columns)
        new_df['PATNO'] = patient_list
        for index, row in new_df.iterrows():
            if row['PATNO'] in patient_list.values:
                mask = (df['PATNO'] == row['PATNO']) & (df['EVENT_ID'] == 'BL')
                if mask.any():
                    new_df.loc[index] = df.loc[mask].iloc[0]

        columns_to_select = file_columns[filename]
        df_select = new_df[columns_to_select]
        num_nans0 = df_select.isna().sum().sum()
        patno_list=new_df['PATNO']
        #print("No. data missing: ", num_nans0)
        dataframes = pd.concat([dataframes, df_select], axis=1)
        # Append the file name to the list 'test_names' for each column in 'df_select'
        test_names.extend([filename] * len(columns_to_select))

    new_list = patient_list.tolist()
    # Remove any columns that have only zeros in them
    dataframes = dataframes.loc[:, (dataframes != 0).any(axis=0)]
    dataframes4 = pd.concat([pd.DataFrame(new_list, columns=['PATNO']), dataframes], axis=1)


    # # #Calculate the percentage of missing values
    # # num_nans = dataframes.isna().sum().sum()
    # # print("Percentage of missing data: ", num_nans/dataframes.size)
    # dataframes_ref = dataframes4
    # Check for non-numeric values and replace with NaN
    dataframes4= dataframes4.apply(pd.to_numeric, errors='coerce')
    dataframes4 = dataframes4.dropna()

    # Extract all columns except 'PATNO'
    columns_to_extract2 = [col for col in dataframes4.columns if col != 'PATNO']
    df_except_patno = dataframes4[columns_to_extract2]
    dataframes=df_except_patno



    # # Check if the scale of the features is similar and standardize if necessary
    # if standardization==1:
    #     scaler = StandardScaler()
    #     scaled_df = scaler.fit_transform(dataframes)
    #     dataframes = pd.DataFrame(scaled_df, columns=dataframes.columns)

    patno_list_new=dataframes4['PATNO']

    return dataframes, test_names, patno_list_new



# def data_pre_process_non_motor (outlier_detection=1,standardization=1):
#     new_test_score_idx=[]
#     new_test_score=0
#     base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
#     motor_dir = os.path.join(base_dir, 'data', 'Motor___MDS-UPDRS')
#     non_motor_dir = os.path.join(base_dir, 'data', 'Non-motor_Assessments')
#     dataframes2, test_names, new_PATNO = data_pre_process(outlier_detection=0, standardization=0)
#     file_columns = {
#         "Benton_Judgement_of_Line_Orientation.csv": ["JLO_TOTRAW"],
        
#         # "Clock_Drawing.csv": ["CLCKTOT"],

#         "Cognitive_Change.csv": ["COGCHG"],
#         "Cognitive_Categorization.csv":["COGDECLN", "FNCDTCOG", "COGSTATE", "COGDXCL", "RVWNPSY", "COGCAT"],

#         "Epworth_Sleepiness_Scale.csv": ["ESS1", "ESS2", "ESS3", "ESS4","ESS5",	"ESS6", "ESS7", "ESS8"],


#         "Geriatric_Depression_Scale__Short_Version_.csv": ["GDSSATIS","GDSDROPD","GDSEMPTY",
#                                                            "GDSBORED","GDSGSPIR","GDSAFRAD","GDSHAPPY"
#                                                            ,"GDSHLPLS","GDSHOME","GDSMEMRY","GDSALIVE",
#                                                            "GDSWRTLS","GDSENRGY","GDSHOPLS","GDSBETER"],
        
#         "Hopkins_Verbal_Learning_Test_-_Revised.csv":["DVT_TOTAL_RECALL","DVT_DELAYED_RECALL",
#                                                       "DVT_RETENTION","DVT_RECOG_DISC_INDEX"],
#         "Letter_-_Number_Sequencing.csv": ["DVS_LNS","LNS_TOTRAW"],
#         "Lexical_Fluency.csv":["LXFLUEF","LXFLUEA","LXFLUES"],
#         "Modified_Boston_Naming_Test.csv":["MBSTNSCR", "MBSTNCRC"],
#         "Modified_Semantic_Fluency.csv":["DVS_SFTANIM", "DVT_SFTANIM"],
#         # "Montreal_Cognitive_Assessment__MoCA_.csv":["MCAALTTM","MCACUBE","MCACLCKC","MCACLCKN","MCACLCKH",
#         #                                             "MCALION","MCARHINO","MCACAMEL","MCAFDS","MCABDS","MCAVIGIL",
#         #                                             "MCASER7","MCASNTNC","MCAVFNUM","MCAVF","MCAABSTR","MCAREC1",
#         #                                             "MCAREC2","MCAREC3","MCAREC4","MCAREC5","MCADATE","MCAMONTH",
#         #                                             "MCAYR","MCADAY","MCAPLACE","MCACITY"],

#         "Neuro_QoL__Cognition_Function_-_Short_Form.csv":["NQCOG22R","NQCOG24R","NQCOG25R","NQCOG40R"],
#         "Neuro_QoL__Communication_-_Short_Form.csv":["NQCOG01","NQCOG04","NQCOG08",
#                                                                            "NQCOG10","NQCOG11"],
#         "QUIP-Current-Short.csv":["TMGAMBLE","CNTRLGMB","TMSEX","CNTRLSEX","TMBUY","CNTRLBUY",
#                                   "TMEAT","CNTRLEAT","TMTORACT","TMTMTACT","TMTRWD","TMDISMED","CNTRLDSM"],
#         "REM_Sleep_Behavior_Disorder_Questionnaire.csv":["DRMVIVID","DRMAGRAC","DRMNOCTB","SLPLMBMV","SLPINJUR",
#                                                          "DRMVERBL","DRMFIGHT","DRMUMV","DRMOBJFL","MVAWAKEN",
#                                                          "DRMREMEM","SLPDSTRB","STROKE","HETRA","PARKISM",
#                                                          "RLS","NARCLPSY","DEPRS","EPILEPSY","BRNINFM"],
#         "SCOPA-AUT.csv":["SCAU1","SCAU2","SCAU3","SCAU4","SCAU5","SCAU6","SCAU7","SCAU8","SCAU9","SCAU10",
#                           "SCAU11","SCAU12","SCAU13","SCAU14","SCAU15","SCAU16","SCAU17","SCAU18","SCAU19",
#                           "SCAU20","SCAU21","SCAU22","SCAU23","SCAU23A","SCAU24","SCAU25","SCAU26A",
#                           "SCAU26B","SCAU26C","SCAU26D"],
#         "State-Trait_Anxiety_Inventory.csv":["STAIAD1","STAIAD2","STAIAD3","STAIAD4","STAIAD5","STAIAD6","STAIAD7",
#                                              "STAIAD8","STAIAD9","STAIAD10","STAIAD11","STAIAD12","STAIAD13",
#                                              "STAIAD14","STAIAD15","STAIAD16","STAIAD17","STAIAD18","STAIAD19",
#                                              "STAIAD20","STAIAD21","STAIAD22","STAIAD23","STAIAD24","STAIAD25",
#                                              "STAIAD26","STAIAD27","STAIAD28","STAIAD29","STAIAD30","STAIAD31",
#                                              "STAIAD32","STAIAD33","STAIAD34","STAIAD35","STAIAD36","STAIAD37",
#                                              "STAIAD38","STAIAD39","STAIAD40"],      
#         "Symbol_Digit_Modalities_Test.csv":["DVSD_SDM","DVT_SDM"],
#         "Trail_Making_A_and_B.csv":["TMTASEC", "TMTBSEC"],          
#         # "University_of_Pennsylvania_Smell_Identification_Test__UPSIT_.csv":["UPSIT_PRCNTGE"]                               


        


#     }

#     #Choose the PD patients from the "Consensus_Committee_Analytic_Datasets_28OCT21.csv" in the PPMI dataset in folder 'Quick_Start'


#     patient_list = new_PATNO#data_fun.select_patient_list('Consensus_Committee_Analytic_Datasets_23Sep2022.xlsx')
    
#     # Initialize an empty list and data frame to store the data matrices and dataframes from each file
#     data_matrix_list = []
#     dataframes = pd.DataFrame()
#     # Initialize an empty list to store the file names
#     test_names = []
#     for filename in os.listdir(non_motor_dir):
#         # Construct the path to the current file
#         file_path = os.path.join(non_motor_dir, filename)

#         print(filename)
#         df = pd.read_csv(file_path)
#         new_df = pd.DataFrame(np.nan, index=np.arange(len(patient_list)), columns=df.columns)
#         new_df['PATNO'] = patient_list
#         for index, row in new_df.iterrows():
#             if row['PATNO'] in patient_list.values:
#                 mask = (df['PATNO'] == row['PATNO']) & (df['EVENT_ID'] == 'BL')
#                 if mask.any():
#                     new_df.loc[index] = df.loc[mask].iloc[0]

#         columns_to_select = file_columns[filename]
#         df_select = new_df[columns_to_select]
#         num_nans0 = df_select.isna().sum()
#         # Calculate the threshold for NaN values
#         threshold = 0.2 * len(df_select)

#         # Filter columns where the number of NaN values exceeds the threshold
#         columns_to_drop = df_select.columns[df_select.isna().sum() > threshold]
#         print("Dropped columns:")
#         for column in columns_to_drop:
#             print(column)
#         # Drop the selected columns
#         if len(df_select.columns) == len(columns_to_drop):
#             print("All columns dropped. Continuing to next iteration.")
#             continue
#         df_select = df_select.drop(columns=columns_to_drop)
#         #
#         dataframes = pd.concat([dataframes, df_select], axis=1)
#         # Append the file name to the list 'test_names' for each column in 'df_select'
#         test_names.extend([filename] * len(columns_to_select))

#     new_list = patient_list.tolist()
#     # Remove any columns that have only zeros in them
#     dataframes = dataframes.loc[:, (dataframes != 0).any(axis=0)]
#     dataframes = pd.concat([pd.DataFrame(new_list, columns=['PATNO']), dataframes], axis=1)


#     #Calculate the percentage of missing values
#     num_nans = dataframes.isna().sum().sum()
#     print("Percentage of missing data: ", num_nans/dataframes.size)
#     missing_values_count = dataframes.isnull().sum()



#     # #Check for non-numeric values and replace with NaN
#     # dataframes = dataframes.apply(pd.to_numeric, errors='coerce')
#     # dataframes = dataframes.dropna()
#     # Create an instance of SimpleImputer with desired strategy (e.g., mean, median, most_frequent)
#     imputer = SimpleImputer(strategy='most_frequent')

#     # Apply imputation to the DataFrame
#     dataframes = pd.DataFrame(imputer.fit_transform(dataframes), columns=dataframes.columns)
    

#     if outlier_detection==1:
#     # Check for outliers using Tukey's fences and replace with NaN
#         Q1 = dataframes.quantile(0.25)
#         Q3 = dataframes.quantile(0.75)
#         IQR = Q3 - Q1
#         dataframes[(dataframes < (Q1 - 1.5 * IQR)) |(dataframes > (Q3 + 1.5 * IQR))] = np.nan

#     # Check for the full nan columns
#     # Count NaN values in each column
#     nan_counts = dataframes.isnull().sum()

#     # dataframes.fillna(dataframes.mean(), inplace=True)

#     # remove the column with all NaN values
#     dataframes = dataframes.dropna(axis=1, how='all')

#     # Check if the scale of the features is similar and standardize if necessary
#     if standardization==1:
#         scaler = StandardScaler()
#         scaled_df = scaler.fit_transform(dataframes)
#         dataframes = pd.DataFrame(scaled_df, columns=dataframes.columns)



#     return dataframes, test_names


def data_pre_process_4_DBS_model(OFF_minus_ON_method=1,patient_class='sporadic'):

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    motor_dir = os.path.join(base_dir, 'data', 'MOTOR_2024')
    # non_motor_dir = os.path.join(base_dir, 'data', 'Non-motor_Assessments')
    file_columns = {
        "MDS_UPDRS_Part_II__Patient_Questionnaire.csv": ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN",
                                                        "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"],
        
        "MDS-UPDRS_Part_I_Patient_Questionnaire.csv": ["NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"],

        "MDS-UPDRS_Part_I.csv": ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"],

        "MDS-UPDRS_Part_III.csv": ['PDSTATE',"DBSYN","NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR",\
                                "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NHY","NP3PSTBL"],


        "Modified_Schwab___England_Activities_of_Daily_Living.csv": ["MSEADLG"],

    }

    updrs_3_full_columns=file_columns["MDS-UPDRS_Part_III.csv"]

    #Choose the PD patients from the "Consensus_Committee_Analytic_Datasets_28OCT21.csv" in the PPMI dataset in folder 'Quick_Start'
    updrs3_columns=["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR", \
                    "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NP3PSTBL"]

 # Convert specified columns to a set (flattened if needed)
    specified_columns_set = set(updrs3_columns)
    
    print('ON-OFF method is',OFF_minus_ON_method)
    if OFF_minus_ON_method==0:
        response = input(f"OFF_minus_ON_method is disabled..Do you want to continue? (yes/no): ")
        if response.lower() == 'yes':
            response2 = input(f"Do you want to enable OFF-ON space? (yes/no): ")
            if response2.lower() == 'yes':
                OFF_minus_ON_method=1
                print('OFF-ON enabled')
            else:
                print("Continuing with the operation without the on off space")
        # Perform more operations if needed
        else:
            print("Operation aborted.")
            sys.exit()
    patient_list = data_fun.select_patient_list('PPMI_Consensus_Committee_Analytic_Datasets_23Oct2023.xlsx',patient_class=patient_class)
   


    dataframes = pd.DataFrame()
    dataframe4DBS = pd.DataFrame()
    dataframeOFF = pd.DataFrame()
    dataframeON = pd.DataFrame()
    # df10=pd.DataFrame()
    # Initialize an empty list to store the file names
    test_names = []
    new_df=[]
    start_button=1
    for filename in os.listdir(motor_dir):
        # Construct the path to the current file
        file_path = os.path.join(motor_dir, filename)

        print(filename)
        df = pd.read_csv(file_path)
        df.reset_index(drop=True, inplace=True)
        df.replace(101, np.nan, inplace=True)
 
        df_columns_set = set(df.columns)
        if df_columns_set.intersection(specified_columns_set):
            # Find which specified columns are in the DataFrame
            matching_columns = [col for col in df.columns if col in updrs3_columns]
            df['NP3TOT'] = df[matching_columns].sum(axis=1)


            
        if 'PDSTATE'in df.columns and OFF_minus_ON_method==1:
            groups = df.groupby(['PATNO', 'EVENT_ID'])

            # Initializing an empty dataframe to store resulting DataFrames
            result_dfs = pd.DataFrame()
            result_dfs_off = pd.DataFrame()
            result_dfs_on = pd.DataFrame()
            # Iterating over each group
            for _, group_df in groups:          
                # Checking if both 'ON' and 'OFF' PDSTATE values exist for the patient
                if 'ON' in group_df['PDSTATE'].values and 'OFF' in group_df['PDSTATE'].values:
                    # Filtering 'OFF' rows minus 'ON' rows
                    off_rows = group_df[group_df['PDSTATE'] == 'OFF']
                    on_rows = group_df[group_df['PDSTATE'] == 'ON']
                    copy=off_rows.copy()
                    copy_on=on_rows.copy()
                    non_numeric_columns2 = off_rows.select_dtypes(exclude='number').columns
                    if 'DBSYN' in copy.columns:
                        non_numeric_columns = non_numeric_columns2.tolist() + ['PATNO','DBSYN']
                    else:
                        non_numeric_columns = non_numeric_columns2.tolist() + ['PATNO']
                    off_rows=off_rows.drop(columns=non_numeric_columns)
                    on_rows=on_rows.drop(columns=non_numeric_columns)
                    ans=(np.asarray(off_rows)-np.asarray(on_rows))
                    off_minus_on=pd.DataFrame(ans,columns=off_rows.columns)
                    # Extracting selected columns from 'copy' DataFrame and converting column names to list of strings
                    # mandatory_columns = list(copy[['INFODT', 'PATNO', 'EVENT_ID']].columns)
                    copy.reset_index(drop=True, inplace=True)
                    if 'DBSYN' in copy.columns:
                        off_minus_on=pd.concat([off_minus_on, copy[['PATNO','EVENT_ID','INFODT','DBSYN','PDSTATE']]], axis=1)
                    else:
                        off_minus_on=pd.concat([off_minus_on, copy[['PATNO','EVENT_ID','INFODT','PDSTATE']]], axis=1)
                    

                    result_dfs_off=pd.concat([result_dfs_off,copy],ignore_index=True)
                    result_dfs_on=pd.concat([result_dfs_on,copy_on],ignore_index=True)
                    result_dfs=pd.concat([result_dfs,off_minus_on],ignore_index=True)

            # Concatenating all resulting DataFrames into a single DataFrame
            df4DBS=df.copy()
            df = result_dfs.copy()
            df_ON=result_dfs_on.copy()
            df_OFF=result_dfs_off.copy()
            # Resetting index
        
            df.reset_index(drop=True, inplace=True)
        else:
            df4DBS=df.copy()
            df_ON=df.copy()
            df_OFF=df.copy()

        mask=df['PATNO'].isin(patient_list)
        DBSmask=df4DBS['PATNO'].isin(patient_list)
        ON_mask=df_ON['PATNO'].isin(patient_list)
        OFF_mask=df_OFF['PATNO'].isin(patient_list)
        new_df=df[mask]
        new_df4DBS=df4DBS[DBSmask]
        new_df_ON=df_ON[ON_mask]
        new_df_OFF=df_OFF[OFF_mask]
        columns_to_select = file_columns[filename]
        columns_to_select2=columns_to_select.copy()
        if 'NP3TOT' in new_df.columns:
            columns_to_select2.extend(['PATNO','EVENT_ID','INFODT','NP3TOT'])
        else:
            columns_to_select2.extend(['PATNO','EVENT_ID','INFODT'])
        df_select = new_df[columns_to_select2].copy()
        df_select4DBS=new_df4DBS[columns_to_select2].copy()
        df_selectON=new_df_ON[columns_to_select2].copy()
        df_selectOFF=new_df_OFF[columns_to_select2].copy()
        df_select.reset_index(drop=True, inplace=True)
        df_select4DBS.reset_index(drop=True, inplace=True)
        df_selectON.reset_index(drop=True, inplace=True)
        df_selectOFF.reset_index(drop=True, inplace=True)
        
        # df_select4DBS3=df_select4DBS[df_select4DBS['EVENT_ID']=='BL']
        # df_select4DBS3=df_select4DBS3.drop(columns=['EVENT_ID'])
        num_nans0 = df_select.isna().sum()
        #Temporarily disabling the threshold check
        # # Calculate the threshold for NaN values
        # threshold = 0.2 * len(df_select)

        # # Filter columns where the number of NaN values exceeds the threshold
        # columns_to_drop = df_select.columns[df_select.isna().sum() > threshold]
        # print("Dropped columns:")
        # for column in columns_to_drop:
        #     print(column)
        # # Drop the selected columns
        # if len(df_select.columns) == len(columns_to_drop):
        #     print("All columns dropped. Continuing to next iteration.")
        #     continue
        # df_select = df_select.drop(columns=columns_to_drop)

        if start_button==1:
            dataframes=df_select.copy()
            dataframe4DBS=df_select4DBS.copy()
            dataframeOFF=df_selectOFF.copy()
            dataframeON=df_selectON.copy()
            # df10=df_select4DBS3.copy()
        else:
            dataframes=pd.merge(dataframes, df_select, on=['PATNO', 'EVENT_ID','INFODT'], how='outer')
            dataframe4DBS=pd.merge(dataframe4DBS, df_select4DBS, on=['PATNO', 'EVENT_ID','INFODT'], how='outer')
            dataframeOFF=pd.merge(dataframeOFF, df_selectOFF, on=['PATNO', 'EVENT_ID','INFODT'], how='outer')
            dataframeON=pd.merge(dataframeON, df_selectON, on=['PATNO', 'EVENT_ID','INFODT'], how='outer')
            # df_select4DBS3=df_select4DBS[df_select4DBS['EVENT_ID']=='BL']
            # df_select4DBS3=df_select4DBS3.drop(columns=['EVENT_ID'])
            # df10=pd.merge(df10, df_select4DBS3, on=['PATNO'], how='outer')
            dataframes.reset_index(drop=True, inplace=True)
            dataframe4DBS.reset_index(drop=True, inplace=True)
            dataframeOFF.reset_index(drop=True, inplace=True)
            dataframeON.reset_index(drop=True, inplace=True)
            # df10.reset_index(drop=True, inplace=True)
        start_button=0

        # Append the file name to the list 'test_names' for each column in 'df_select'
        test_names.extend([filename] * len(columns_to_select))
        
    # dataframes = pd.concat([dataframes, new_df[['INFODT','PATNO','EVENT_ID']]], axis=1)
    
    df_copy=dataframes.copy()
    df_copy_4DBS=dataframe4DBS.copy()
    # dataframe4DBS = dataframe4DBS[(dataframe4DBS['PDSTATE'] == 'OFF')]
    dataframe4DBS=dataframe4DBS.drop(columns='PDSTATE')
    dataframes=dataframes.drop(columns=['PDSTATE'])

    response2 = input(f"Do you need Both targets or just Schwab and England:(both/se) ").lower()
    if response2== 'both':
        dataframeON=dataframeON.drop(columns=['PDSTATE'])
        dataframeOFF=dataframeOFF.drop(columns=['PDSTATE'])
    elif response2=='se':
        dataframeON=dataframeON[['PATNO','EVENT_ID','INFODT','DBSYN','MSEADLG']]
        dataframeOFF=dataframeOFF[['PATNO','EVENT_ID','INFODT','DBSYN','MSEADLG']]

    else:
        print("Only both or se are accepted answers...Code execution aborted.")
        sys.exit()
    dataframes = dataframes.dropna()
    df2=dataframes.copy()
    dataframes=dataframes.drop(columns=['DBSYN','NP3TOT','MSEADLG'])

    # dataframes.replace(101, np.nan, inplace=True)
    # dataframe4DBS.replace(101, np.nan, inplace=True)
    dataframes = dataframes.dropna()
    dataframe4DBS=dataframe4DBS.dropna()
    dataframeON=dataframeON.dropna()
    dataframeOFF=dataframeOFF.dropna()
    

    dataframes.reset_index(drop=True, inplace=True)
    dataframe4DBS.reset_index(drop=True, inplace=True)
    dataframeOFF.reset_index(drop=True, inplace=True)
    dataframeON.reset_index(drop=True, inplace=True)
    # Remove any columns that have only zeros in thems
    dataframes = dataframes.loc[:, (dataframes != 0).any(axis=0)]
    dataframe4DBS = dataframe4DBS.loc[:, (dataframe4DBS != 0).any(axis=0)]
    if response2=='both':
        response = input(f"Do you need the OFF or ON measurements as target(OFF/ON): ").lower()
        if response== 'off':
            df3=dataframeOFF.copy()
        elif response=='on':
            df3=dataframeON.copy()
        else:
            print("Only OFF or ON are accepted answers...Code execution aborted.")
            sys.exit()
    else:
        df3=dataframeON.copy()


    #df_copy
    df2['INFODT'] = pd.to_datetime(df2['INFODT'])
    df3['INFODT'] = pd.to_datetime(df3['INFODT'])

    dataframes['INFODT'] = pd.to_datetime(dataframes['INFODT'])
    dataframe4DBS['INFODT'] = pd.to_datetime(dataframe4DBS['INFODT'])
    df2 = df2.sort_values(by=['PATNO', 'INFODT'])

    # Find the rows where DBSYN is first equal to 1 in each 'PATNO' group
    first_dbsyn_1_rows = dataframe4DBS[dataframe4DBS['DBSYN'] == 1].groupby('PATNO').first()
    df_dbs=df2[df2['DBSYN'] == 1]
    df_dbs_copy=df_dbs.copy()
    df_dbs_full_data=df_copy_4DBS[df_copy_4DBS['DBSYN'] == 1]
    total_dbs_patients=df_dbs_full_data['PATNO'].nunique()
    on_off_dbs=df_dbs['PATNO'].nunique()

    # Initialize an empty list to store the result rows
    result_rows = []
    # Initialize an empty DataFrame to store the result
    result_after1 = []
    # Iterate over rows where DBSYN is first equal to 1

    PD_pre=[]
    PD_post_on_off_diabled=[]
    PD_pre_on_off_disabled=[]
    counter=0
    counter2=0
    counter3=0
    counter4=0
   
    print(f'To begin we have',df_dbs['PATNO'].nunique(),'patients')
    for patno, row_dbsyn_1 in first_dbsyn_1_rows.iterrows():
        if patno in df_dbs_copy['PATNO'].unique():
            # same_patno_rows = dataframes[dataframes['PATNO'] == patno]
            same_patno_rows= df2[df2['PATNO'] == patno]
            same_patno_rows_on_off_disabled=df3[df3['PATNO'] == patno]
            # Find rows with INFODT before the INFODT of the row where DBSYN is first equal to 1
            before_dbsyn_1_rows = same_patno_rows[same_patno_rows['INFODT'] < row_dbsyn_1['INFODT']]
            before_dbsyn_1_rows_on_off_diabled = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] < row_dbsyn_1['INFODT']]
            after_dbsyn_1_rows_on_off_diabled = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] > row_dbsyn_1['INFODT']]

            # before_dbsyn_1_rows_full_data = same_patno_rows_full_data [same_patno_rows_full_data ['INFODT'] < row_dbsyn_1['INFODT']]
            # after_dbsyn_1_rows_full_data = same_patno_rows_full_data [same_patno_rows_full_data ['INFODT'] > row_dbsyn_1['INFODT']]
            # Find rows with INFODT more than one year away from the INFODT of the row where DBSYN is first equal to 1
            one_year_after_dbsyn = row_dbsyn_1['INFODT'] + dt.timedelta(days=365)
            rows_more_than_one_year_apart = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] > one_year_after_dbsyn]
            
            # if (after_dbsyn_1_rows_full_data['DBSYN'] != 1).any() and ~((after_dbsyn_1_rows['DBSYN'] != 1).any()):
            #     print(f'The patient {patno} doesnt have Off-ON, and however, their DBS data is unclear - DBSYN changes from 1 to another value after the first occurrence of DBSYN=1\n')
            #     counter_no_on_off_invalid+=1
            #     df_dbs=df_dbs[df_dbs['PATNO']!=patno]

            if (after_dbsyn_1_rows_on_off_diabled['DBSYN'] != 1).any():
                # Print a message for patients where it is unclear
                print(f'The patient {patno} is unclear - DBSYN changes from 1 to another value after the first occurrence of DBSYN=1\n, Hence dropping the patient')
                df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
                # print(df2[df2['PATNO'] == patno][['PATNO', 'EVENT_ID', 'DBSYN', 'INFODT']])
                counter+=1
            else:
                if not before_dbsyn_1_rows.empty:
                    # Select the row with the closest INFODT before the INFODT of the row where DBSYN is first equal to 1
                    closest_date_row_before = before_dbsyn_1_rows.iloc[-1].copy()  # Make a copy to avoid SettingWithCopyWarning
                    closest_date_row_before_on_off_disabled = before_dbsyn_1_rows_on_off_diabled.iloc[-1].copy()
                    if not rows_more_than_one_year_apart.empty:
                        # Select the row with the closest INFODT more than one year after the INFODT of the row where DBSYN is first equal to 1
                        closest_date_row = rows_more_than_one_year_apart.iloc[0].copy()  # Make a copy to avoid SettingWithCopyWarning
                        post= rows_more_than_one_year_apart.loc[(rows_more_than_one_year_apart['EVENT_ID'] == closest_date_row['EVENT_ID'])].iloc[0]
                        PD_post_on_off_diabled.append(post)
                
                        # Only append the 'pre dbs' is 'post dbs' is available  
                        pre= before_dbsyn_1_rows.loc[(before_dbsyn_1_rows['EVENT_ID'] == closest_date_row_before['EVENT_ID'])].iloc[0]
                        pre_on_off_disabled= before_dbsyn_1_rows_on_off_diabled.loc[(before_dbsyn_1_rows_on_off_diabled['EVENT_ID'] == closest_date_row_before_on_off_disabled['EVENT_ID'])].iloc[0]

                        PD_pre_on_off_disabled.append(pre_on_off_disabled)
                        PD_pre.append(pre)
                        

                        # # Append the post dbs to the result DataFrame
                        # result_after1.append(closest_date_row)
                        # # Only append the 'pre dbs' is 'post dbs' is available                    
                        # result_rows.append(closest_date_row_before)
                    else:
                        print(f'the patient ',patno,' post DBS data is less than 12 months apart, dropping the patient')
                        df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                        counter2+=1
                        print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
                else:
                    print(f'the patient ',patno,' No pre DBS values\n, hence dropping the patient')
                    # print(df2[df2['PATNO'] == patno][['PATNO','EVENT_ID','DBSYN','INFODT']])
                    df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                    print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
        
                    counter3+=1
        else:
            counter4+=1        
    

    selected_dbs_patients=df_dbs['PATNO'].nunique()
    print(f'Out of', total_dbs_patients,'patients, after removing NaNs, there are',on_off_dbs,'DBS patients who has ON and OFF visit info. But out of that only',selected_dbs_patients,'have pre and post values\n. This is because',counter,'patient\'s DBS data is inconsistant with time and\n',
          counter2,'patients\' post DBS results are less than 12 months apart and there are',counter3,'patients with no pre-DBS data')# and',counter_no_on_off_invalid,'who doesnt have proper On OFF data and whose DBS data is invalid')

    patno_list_new=dataframes['PATNO']
    result_after_on_off_disabled = pd.DataFrame(PD_post_on_off_diabled)
    result_before = pd.DataFrame(PD_pre)    
    result_before_on_off_disabled=pd.DataFrame(PD_pre_on_off_disabled)
    result_after_on_off_disabled.reset_index(drop=True, inplace=True)
    result_before=result_before.drop(columns=['DBSYN','NP3TOT','MSEADLG'])
    result_before.reset_index(drop=True, inplace=True)
    result_before_on_off_disabled.reset_index(drop=True, inplace=True)
    dataframes_select=dataframes.copy()
    # dataframes_select=dataframes_select.drop(columns=['NP3TOT','MSEADLG'])

    return dataframes_select, test_names, patno_list_new,result_before_on_off_disabled,result_after_on_off_disabled,matching_columns,result_before,OFF_minus_ON_method,updrs_3_full_columns,response2


def data_pre_process_all_visits(patient_class='sporadic'):
  
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    motor_dir = os.path.join(base_dir, 'data', 'MOTOR_2024')
    non_motor_dir = os.path.join(base_dir, 'data', 'Non-motor_Assessments')
    file_columns = {
        "C_MDS_UPDRS_Part_II__Patient_Questionnaire.csv": ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN",
                                                        "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"],
        
        "B_MDS-UPDRS_Part_I_Patient_Questionnaire.csv": ["NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"],

        "A_MDS-UPDRS_Part_I.csv": ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"],

        "D_MDS-UPDRS_Part_III.csv": ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR",\
                                "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3PSTBL",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NHY","PDTRTMNT","PDSTATE","PDMEDYN","DBSYN"],


        "E_Modified_Schwab___England_Activities_of_Daily_Living.csv": ["MSEADLG"], # this feature is not considered, we drop it later, including it since we need it for pre post dbs analysis and hence need to be in the data folder

    }

    #Choose the PD patients from the "Consensus_Committee_Analytic_Datasets_28OCT21.csv" in the PPMI dataset in folder 'Quick_Start'


    patient_list = data_fun.select_patient_list('PPMI_Consensus_Committee_Analytic_Datasets_23Oct2023.xlsx',patient_class=patient_class)
    
    # Initialize an empty list and data frame to store the data matrices and dataframes from each file
    
    dataframes = pd.DataFrame()
    # Initialize an empty list to store the file names
    test_names = []
    start_button=1


    for filename in sorted(os.listdir(motor_dir)):
        # Construct the path to the current file
        file_path = os.path.join(motor_dir, filename)

        print(filename)
        df = pd.read_csv(file_path)

        df.reset_index(drop=True, inplace=True)

        new_df = pd.DataFrame(columns=df.columns)
 
        columns_to_select = file_columns[filename].copy()
        test_names.extend([filename] * len(columns_to_select))
        columns_to_select.extend(['PATNO','EVENT_ID','INFODT',])
        # Use `isin` to filter rows where PATNO is in patient_list
        mask = df['PATNO'].isin(patient_list)
        filtered_df = df.loc[mask]

        if not filtered_df.empty:
            # Directly assign filtered rows to new_df
            new_df = filtered_df.copy()
        else:
            print('No patient data available for this file.')
        
        df_select = new_df[columns_to_select]


        if start_button==1:
            dataframes=df_select.copy()
            
            start_button=0
        else:
            dataframes=pd.merge(dataframes, df_select, on=['PATNO','EVENT_ID','INFODT'], how='inner')
            dataframes.reset_index(drop=True, inplace=True)
        # Append the file name to the list 'test_names' for each column in 'df_select'
        

    new_list = patient_list.tolist()
    # Remove any columns that have only zeros in them
    dataframes = dataframes.loc[:, (dataframes != 0).any(axis=0)]
    # dataframes = pd.concat([pd.DataFrame(new_list, columns=['PATNO']), dataframes], ignore_index=True)

    # Replace all instances of 101 with NaN

    dataframes.replace(101, np.nan, inplace=True)
#!!! note that we are not removing nan rows since we are grouping all events together and for baseline, the PDSTATE is always nan

    # # # #Calculate the percentage of missing values
    # # # num_nans = dataframes.isna().sum().sum()
    # # # print("Percentage of missing data: ", num_nans/dataframes.size)
    # # dataframes_ref = dataframes4
    # # Check for non-numeric values and replace with NaN
    # # dataframes = dataframes.apply(pd.to_numeric, errors='coerce')
    # dataframes = dataframes.dropna()
    # dataframes.reset_index(drop=True, inplace=True)

    patno_list_new=dataframes['PATNO']


    return dataframes, test_names, patno_list_new



def data_extraction_OFF_ON_model(df):

    updrs3_columns=["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR", \
                    "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NP3PSTBL"]

    
    full_updrs3_columns = updrs3_columns + ["NHY"]
 
    dataframes = pd.DataFrame()
    dataframe4DBS = pd.DataFrame()
    dataframeOFF = pd.DataFrame()
    dataframeON = pd.DataFrame()

    df.replace(101, np.nan, inplace=True)

    # Step 1: Separate columns in full_updrs3_columns from the rest of the df
    updrs3_df = df[full_updrs3_columns + ['PATNO', 'EVENT_ID','DBSYN','PDSTATE','INFODT']] # Subset of df with only UPDRS3 and NHY columns
    other_columns_df = df.drop(columns=full_updrs3_columns)  # Remaining columns in df

    groups = updrs3_df.groupby(['PATNO', 'EVENT_ID'])

    # Initializing an empty dataframe to store resulting DataFrames
    result_dfs = pd.DataFrame()
    result_dfs_off = pd.DataFrame()
    result_dfs_on = pd.DataFrame()
    # Iterating over each group
    for _, group_df in groups:          
        # Checking if both 'ON' and 'OFF' PDSTATE values exist for the patient
        if 'ON' in group_df['PDSTATE'].values and 'OFF' in group_df['PDSTATE'].values:
            # Filtering 'OFF' rows minus 'ON' rows
            off_rows = group_df[group_df['PDSTATE'] == 'OFF']
            on_rows = group_df[group_df['PDSTATE'] == 'ON']
            copy=off_rows.copy()
            copy_on=on_rows.copy()
            non_numeric_columns2 = off_rows.select_dtypes(exclude='number').columns
            if 'DBSYN' in copy.columns:
                non_numeric_columns = non_numeric_columns2.tolist() + ['PATNO','DBSYN']
            else:
                non_numeric_columns = non_numeric_columns2.tolist() + ['PATNO']
            off_rows=off_rows.drop(columns=non_numeric_columns)
            on_rows=on_rows.drop(columns=non_numeric_columns)
            ans=(np.asarray(off_rows)-np.asarray(on_rows))
            off_minus_on=pd.DataFrame(ans,columns=off_rows.columns)

            copy.reset_index(drop=True, inplace=True)
            if 'DBSYN' in copy.columns:
                off_minus_on=pd.concat([off_minus_on, copy[['PATNO','EVENT_ID','INFODT','DBSYN','PDSTATE']]], axis=1)
            else:
                off_minus_on=pd.concat([off_minus_on, copy[['PATNO','EVENT_ID','INFODT','PDSTATE']]], axis=1)
            

            result_dfs_off=pd.concat([result_dfs_off,copy],ignore_index=True)
            result_dfs_on=pd.concat([result_dfs_on,copy_on],ignore_index=True)
            result_dfs=pd.concat([result_dfs,off_minus_on],ignore_index=True)


    dataframes = pd.merge(other_columns_df, result_dfs, on=['PATNO', 'EVENT_ID', 'INFODT'], how='outer', suffixes=('', '_duplicate'))
    dataframeOFF=pd.merge(other_columns_df, result_dfs_off, on=['PATNO', 'EVENT_ID','INFODT'], how='outer', suffixes=('', '_duplicate'))
    dataframeON=pd.merge(other_columns_df, result_dfs_on, on=['PATNO', 'EVENT_ID','INFODT'], how='outer', suffixes=('', '_duplicate'))
    dataframe4DBS=df.copy()
    # Drop columns with '_duplicate' suffix in each dataframe

    # For dataframes
    duplicate_columns_dataframes = dataframes.filter(regex='_duplicate$').columns
    dataframes = dataframes.drop(columns=duplicate_columns_dataframes)

    # For dataframeOFF
    duplicate_columns_dataframeOFF = dataframeOFF.filter(regex='_duplicate$').columns
    dataframeOFF = dataframeOFF.drop(columns=duplicate_columns_dataframeOFF)

    # For dataframeON
    duplicate_columns_dataframeON = dataframeON.filter(regex='_duplicate$').columns
    dataframeON = dataframeON.drop(columns=duplicate_columns_dataframeON)

    dataframes.reset_index(drop=True, inplace=True)
    dataframe4DBS.reset_index(drop=True, inplace=True)
    dataframeOFF.reset_index(drop=True, inplace=True)
    dataframeON.reset_index(drop=True, inplace=True)


    #Temporarily disabling the threshold check
    # # Calculate the threshold for NaN values
    # threshold = 0.2 * len(df_select)

    # # Filter columns where the number of NaN values exceeds the threshold
    # columns_to_drop = df_select.columns[df_select.isna().sum() > threshold]
    # print("Dropped columns:")
    # for column in columns_to_drop:
    #     print(column)
    # # Drop the selected columns
    # if len(df_select.columns) == len(columns_to_drop):
    #     print("All columns dropped. Continuing to next iteration.")
    #     continue
    # df_select = df_select.drop(columns=columns_to_drop)

    dataframes=dataframes.drop(columns=['PDSTATE'])
    dataframes = dataframes.dropna()
    dataframes.reset_index(drop=True, inplace=True)

    # 
    import sys

    # Identify columns with only zeros
    zero_only_columns = dataframes.columns[(dataframes == 0).all(axis=0)]

    # Check if there are columns with only zeros
    if len(zero_only_columns) > 0:
        # Print a warning with the column names
        print(f"Warning: The following columns contain only zeros and will be removed:\n{list(zero_only_columns)}")

        # Pause execution for user acknowledgment
        input("""
        Press Enter to continue after reviewing the above warning.
        You might need to modify how NP3TOT is calculated, the dataframe4DBS and 
        updrs3column info that is given out 
        """)


        # Remove columns that have only zeros
        
    dataframes = dataframes.drop(columns=zero_only_columns)
    patno_list_new=dataframes['PATNO']
    dataframe4DBS['NP3TOT'] = dataframe4DBS[updrs3_columns].sum(axis=1)
    dataframeON['NP3TOT'] = dataframeON[updrs3_columns].sum(axis=1)
    dataframeOFF['NP3TOT'] = dataframeOFF[updrs3_columns].sum(axis=1)
    return patno_list_new,dataframes, dataframe4DBS,dataframeON,dataframeOFF,updrs3_columns,full_updrs3_columns

def get_input_output_DBS_model(dataframes, dataframe4DBS,dataframeON,dataframeOFF):
    df2=dataframes.copy()
    df_copy_4DBS=dataframe4DBS.copy()
    # dataframe4DBS = dataframe4DBS[(dataframe4DBS['PDSTATE'] == 'OFF')]
    dataframe4DBS=dataframe4DBS.drop(columns='PDSTATE')
    

    response2 = input(f"Do you need Both targets or just Schwab and England:(both/se) ").lower()
    if response2== 'both':
        dataframeON=dataframeON.drop(columns=['PDSTATE'])
        dataframeOFF=dataframeOFF.drop(columns=['PDSTATE'])
    elif response2=='se':
        dataframeON=dataframeON[['PATNO','EVENT_ID','INFODT','DBSYN','MSEADLG']]
        dataframeOFF=dataframeOFF[['PATNO','EVENT_ID','INFODT','DBSYN','MSEADLG']]

    else:
        print("Only both or se are accepted answers...Code execution aborted.")
        sys.exit()


    dataframe4DBS=dataframe4DBS.dropna()
    dataframeON=dataframeON.dropna()
    dataframeOFF=dataframeOFF.dropna()
    

  
    dataframe4DBS.reset_index(drop=True, inplace=True)
    dataframeOFF.reset_index(drop=True, inplace=True)
    dataframeON.reset_index(drop=True, inplace=True)
    # Remove any columns that have only zeros in thems

    # dataframe4DBS = dataframe4DBS.loc[:, (dataframe4DBS != 0).any(axis=0)]
    if response2=='both':
        response = input(f"Do you need the OFF or ON measurements as target(OFF/ON): ").lower()
        if response== 'off':
            df3=dataframeOFF.copy()
        elif response=='on':
            df3=dataframeON.copy()
        
        else:
            print("Only OFF or ON are accepted answers...Code execution aborted.")
            sys.exit()
        response2=response
    else:
        df3=dataframeON.copy()
   

    #df_copy
    df2['INFODT'] = pd.to_datetime(df2['INFODT'])
    df3['INFODT'] = pd.to_datetime(df3['INFODT'])

    
    dataframe4DBS['INFODT'] = pd.to_datetime(dataframe4DBS['INFODT'])
    df2 = df2.sort_values(by=['PATNO', 'INFODT'])

    # Find the rows where DBSYN is first equal to 1 in each 'PATNO' group
    first_dbsyn_1_rows = dataframe4DBS[dataframe4DBS['DBSYN'] == 1].groupby('PATNO').first()
    df_dbs=df2[df2['DBSYN'] == 1]
    df_dbs_copy=df_dbs.copy()
    df_dbs_full_data=df_copy_4DBS[df_copy_4DBS['DBSYN'] == 1]
    total_dbs_patients=df_dbs_full_data['PATNO'].nunique()
    on_off_dbs=df_dbs['PATNO'].nunique()




    PD_pre=[]
    PD_post_on_off_diabled=[]
    PD_pre_on_off_disabled=[]
    counter=0
    counter2=0
    counter3=0
    counter4=0
   
    print(f'To begin we have',df_dbs['PATNO'].nunique(),'patients')
    # Iterate over rows where DBSYN is first equal to 1
    for patno, row_dbsyn_1 in first_dbsyn_1_rows.iterrows():
        if patno in df_dbs_copy['PATNO'].unique():
            # same_patno_rows = dataframes[dataframes['PATNO'] == patno]
            same_patno_rows= df2[df2['PATNO'] == patno] # the data with on-off
            same_patno_rows_on_off_disabled=df3[df3['PATNO'] == patno]
            # Find rows with INFODT before the INFODT of the row where DBSYN is first equal to 1
            before_dbsyn_1_rows = same_patno_rows[same_patno_rows['INFODT'] < row_dbsyn_1['INFODT']]
            before_dbsyn_1_rows_on_off_diabled = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] < row_dbsyn_1['INFODT']]
            after_dbsyn_1_rows_on_off_diabled = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] > row_dbsyn_1['INFODT']]

            # before_dbsyn_1_rows_full_data = same_patno_rows_full_data [same_patno_rows_full_data ['INFODT'] < row_dbsyn_1['INFODT']]
            # after_dbsyn_1_rows_full_data = same_patno_rows_full_data [same_patno_rows_full_data ['INFODT'] > row_dbsyn_1['INFODT']]
            # Find rows with INFODT more than one year away from the INFODT of the row where DBSYN is first equal to 1
            one_year_after_dbsyn = row_dbsyn_1['INFODT'] + dt.timedelta(days=365)
            rows_more_than_one_year_apart = same_patno_rows_on_off_disabled[same_patno_rows_on_off_disabled['INFODT'] > one_year_after_dbsyn]
            
            # if (after_dbsyn_1_rows_full_data['DBSYN'] != 1).any() and ~((after_dbsyn_1_rows['DBSYN'] != 1).any()):
            #     print(f'The patient {patno} doesnt have Off-ON, and however, their DBS data is unclear - DBSYN changes from 1 to another value after the first occurrence of DBSYN=1\n')
            #     counter_no_on_off_invalid+=1
            #     df_dbs=df_dbs[df_dbs['PATNO']!=patno]

            if (after_dbsyn_1_rows_on_off_diabled['DBSYN'] != 1).any():
                # Print a message for patients where it is unclear
                print(f'The patient {patno} is unclear - DBSYN changes from 1 to another value after the first occurrence of DBSYN=1\n, Hence dropping the patient')
                df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
                # print(df2[df2['PATNO'] == patno][['PATNO', 'EVENT_ID', 'DBSYN', 'INFODT']])
                counter+=1
            else:
                if not before_dbsyn_1_rows.empty:
                    # Select the row with the closest INFODT before the INFODT of the row where DBSYN is first equal to 1
                    closest_date_row_before = before_dbsyn_1_rows.iloc[-1].copy()  # Make a copy to avoid SettingWithCopyWarning
                    closest_date_row_before_on_off_disabled = before_dbsyn_1_rows_on_off_diabled.iloc[-1].copy()
                    if not rows_more_than_one_year_apart.empty:
                        # Select the row with the closest INFODT more than one year after the INFODT of the row where DBSYN is first equal to 1
                        closest_date_row = rows_more_than_one_year_apart.iloc[0].copy()  # Make a copy to avoid SettingWithCopyWarning
                        post= rows_more_than_one_year_apart.loc[(rows_more_than_one_year_apart['EVENT_ID'] == closest_date_row['EVENT_ID'])].iloc[0]
                        PD_post_on_off_diabled.append(post)
                
                        # Only append the 'pre dbs' is 'post dbs' is available  
                        pre= before_dbsyn_1_rows.loc[(before_dbsyn_1_rows['EVENT_ID'] == closest_date_row_before['EVENT_ID'])].iloc[0]
                        pre_on_off_disabled= before_dbsyn_1_rows_on_off_diabled.loc[(before_dbsyn_1_rows_on_off_diabled['EVENT_ID'] == closest_date_row_before_on_off_disabled['EVENT_ID'])].iloc[0]

                        PD_pre_on_off_disabled.append(pre_on_off_disabled)
                        PD_pre.append(pre)
                        

                        # # Append the post dbs to the result DataFrame
                        # result_after1.append(closest_date_row)
                        # # Only append the 'pre dbs' is 'post dbs' is available                    
                        # result_rows.append(closest_date_row_before)
                    else:
                        print(f'the patient ',patno,' post DBS data is less than 12 months apart, dropping the patient')
                        df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                        counter2+=1
                        print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
                else:
                    print(f'the patient ',patno,' No pre DBS values\n, hence dropping the patient')
                    # print(df2[df2['PATNO'] == patno][['PATNO','EVENT_ID','DBSYN','INFODT']])
                    df_dbs=df_dbs[df_dbs['PATNO']!=patno]
                    print(f'now there are',df_dbs['PATNO'].nunique(),'patients')
        
                    counter3+=1
        else:
            counter4+=1        
    

    selected_dbs_patients=df_dbs['PATNO'].nunique()
    print(f'Out of', total_dbs_patients,'patients, after removing NaNs, there are',on_off_dbs,'DBS patients who has ON and OFF visit info. But out of that only',selected_dbs_patients,'have pre and post values\n. This is because',counter,'patient\'s DBS data is inconsistant with time and\n',
          counter2,'patients\' post DBS results are less than 12 months apart and there are',counter3,'patients with no pre-DBS data')# and',counter_no_on_off_invalid,'who doesnt have proper On OFF data and whose DBS data is invalid')

    
    result_after_on_off_disabled = pd.DataFrame(PD_post_on_off_diabled)
    result_before = pd.DataFrame(PD_pre)    
    result_before_on_off_disabled=pd.DataFrame(PD_pre_on_off_disabled)
    result_after_on_off_disabled.reset_index(drop=True, inplace=True)
    result_before=result_before.drop(columns=['DBSYN','MSEADLG'])
    result_before.reset_index(drop=True, inplace=True)
    result_before_on_off_disabled.reset_index(drop=True, inplace=True)
 

    return result_before_on_off_disabled,result_after_on_off_disabled,result_before,response2


def data_pre_process_all_visits_version_2025(patient_class='sporadic'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
   

    motor_dir = os.path.join(base_dir, 'data', 'MOTOR_2024')
    non_motor_dir = os.path.join(base_dir, 'data', 'Non-motor_Assessments')
    file_columns = {
        "C_MDS_UPDRS_Part_II__Patient_Questionnaire.csv": ["NP2SPCH", "NP2SALV", "NP2SWAL", "NP2EAT", "NP2DRES", "NP2HYGN",
                                                        "NP2HWRT", "NP2HOBB", "NP2TURN", "NP2TRMR", "NP2RISE", "NP2WALK", "NP2FREZ"],
        
        "B_MDS-UPDRS_Part_I_Patient_Questionnaire.csv": ["NP1SLPN", "NP1SLPD", "NP1PAIN", "NP1URIN", "NP1CNST", "NP1LTHD", "NP1FATG"],

        "A_MDS-UPDRS_Part_I.csv": ["NP1COG", "NP1HALL", "NP1DPRS", "NP1ANXS", "NP1APAT", "NP1DDS"],

        "D_MDS-UPDRS_Part_III.csv": ["NP3SPCH", "NP3FACXP", "NP3RIGN", "NP3RIGRU",	"NP3RIGLU",	"NP3RIGRL", "NP3RIGLL", "NP3FTAPR", "NP3FTAPL",	"NP3HMOVR",\
                                "NP3HMOVL",	"NP3PRSPR",	"NP3PRSPL", "NP3TTAPR",	"NP3TTAPL",	"NP3LGAGR",	"NP3LGAGL",	"NP3RISNG",	"NP3GAIT",	"NP3FRZGT",	"NP3PSTBL",	"NP3POSTR",	\
                                "NP3BRADY",	"NP3PTRMR", "NP3PTRML",	"NP3KTRMR",	"NP3KTRML", "NP3RTARU", "NP3RTALU", "NP3RTARL", "NP3RTALL", "NP3RTALJ", "NP3RTCON","NHY","PDTRTMNT","PDSTATE","PDMEDYN","DBSYN"],


        "E_Modified_Schwab___England_Activities_of_Daily_Living.csv": ["MSEADLG"], # this feature is not considered, we drop it later, including it since we need it for pre post dbs analysis and hence need to be in the data folder

    }

    #Choose the PD patients from the "Consensus_Committee_Analytic_Datasets_28OCT21.csv" in the PPMI dataset in folder 'Quick_Start'


    patient_list = data_fun.select_patient_list('PPMI_Consensus_Committee_Analytic_Datasets_23Oct2023.xlsx',patient_class=patient_class)
    
    # Initialize an empty list and data frame to store the data matrices and dataframes from each file
    
    dataframes = pd.DataFrame()
    # Initialize an empty list to store the file names
    test_names = []
    start_button=1


    for filename in sorted(os.listdir(motor_dir)):
        # Construct the path to the current file
        file_path = os.path.join(motor_dir, filename)

        print(filename)
        df = pd.read_csv(file_path)

        df.reset_index(drop=True, inplace=True)

        new_df = pd.DataFrame(columns=df.columns)
 
        columns_to_select = file_columns[filename].copy()
        test_names.extend([filename] * len(columns_to_select))
        columns_to_select.extend(['PATNO','EVENT_ID','INFODT',])
        # Use `isin` to filter rows where PATNO is in patient_list
        mask = df['PATNO'].isin(patient_list)
        filtered_df = df.loc[mask]

        if not filtered_df.empty:
            # Directly assign filtered rows to new_df
            new_df = filtered_df.copy()
        else:
            print('No patient data available for this file.')
        
        df_select = new_df[columns_to_select]


        if start_button==1:
            dataframes=df_select.copy()
            
            start_button=0
        else:
            dataframes=pd.merge(dataframes, df_select, on=['PATNO','EVENT_ID','INFODT'], how='inner')
            dataframes.reset_index(drop=True, inplace=True)
        # Append the file name to the list 'test_names' for each column in 'df_select'
        

    new_list = patient_list.tolist()
    # Remove any columns that have only zeros in them
    dataframes = dataframes.loc[:, (dataframes != 0).any(axis=0)]
    # dataframes = pd.concat([pd.DataFrame(new_list, columns=['PATNO']), dataframes], ignore_index=True)

    # Replace all instances of 101 with NaN

    dataframes.replace(101, np.nan, inplace=True)
#!!! note that we are not removing nan rows since we are grouping all events together and for baseline, the PDSTATE is always nan

    # # # #Calculate the percentage of missing values
    # # # num_nans = dataframes.isna().sum().sum()
    # # # print("Percentage of missing data: ", num_nans/dataframes.size)
    # # dataframes_ref = dataframes4
    # # Check for non-numeric values and replace with NaN
    # # dataframes = dataframes.apply(pd.to_numeric, errors='coerce')
    # dataframes = dataframes.dropna()
    # dataframes.reset_index(drop=True, inplace=True)

    patno_list_new=dataframes['PATNO']


    return dataframes, test_names, patno_list_new
