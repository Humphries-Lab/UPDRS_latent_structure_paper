from scipy.stats import rankdata
import os, sys
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min
from kneed import KneeLocator
import os

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

data_dir = os.path.join(base_dir, 'data')
saved_data_dir = os.path.join(base_dir, 'saved_data')
results_dir = os.path.join(base_dir, 'results')
figures_dir = os.path.join(base_dir, 'figures')
def select_patient_list(xlsx_file,patient_class='sporadic'):
    if patient_class!='sporadic':
        print(f'The patient class is',patient_class)
        response= input(f"The class of patients is not sporadic. Do you wish to continue? (yes/no): ")
        if response.lower()=='no':
            sys.exit()
    
    # The missing values in thid particular case is indicated by '.', and after studying the document, it was evident that the classification in sheet 'Summary Analytic' was made
    # by considering the missing values as zeros
    df = pd.read_excel(os.path.join(data_dir, xlsx_file), sheet_name='PD', na_values='.')
    df = df.fillna(0)
    # Define the conditions for choosing the patients: Alter the following conditions to choose specific subgroups in the PD cohort
    if patient_class=='sporadic': 
        conditions = ((df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0))#|
        print('considering just sporadic PD group')
    elif patient_class=='both':
        conditions = ((df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 1) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0) |
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 1) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 1) & (df['CONGBA'] == 1) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 1) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==1) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==1))
        print('considering sporadic and genetic PD')
    elif patient_class=='genetic':
        conditions = ((df['CONPD'] == 1) & (df['CONLRRK2'] == 1) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0) |
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 1) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 1) & (df['CONGBA'] == 1) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 1) & (df['CONPRKN']==0) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==1) & (df['CONPINK1']==0)|
                      (df['CONPD'] == 1) & (df['CONLRRK2'] == 0) & (df['CONGBA'] == 0) & (df['CONSNCA'] == 0) & (df['CONPRKN']==0) & (df['CONPINK1']==1))
        print('considering genetic PD')


    # Filter the data based on the conditions and select the PATNO column
    selected_patients = df.loc[conditions, 'PATNO']

    # Return the filtered PATNO values
    return selected_patients


def pca_fun(dataframes, n_components, pick_components):

    # Create a PCA object with the number of components you want to keep
    
    pca = PCA(n_components=n_components)

    # Fit the PCA object to the dataframes dataframe
    dataframes_pca = pca.fit_transform(dataframes)

    # Convert the PCA results to a new pandas DataFrame
    pca_columns = ['PCA{}'.format(i) for i in range(1, n_components+1)]
    dataframes_pca = pd.DataFrame(data=dataframes_pca, columns=pca_columns)


    # Get the explained variance ratio of each component
    explained_variance_ratio = pca.explained_variance_ratio_
    if pick_components==1:
    # Create a bar plot of the explained variance ratios
        # plt.bar(range(1, n_components+1), explained_variance_ratio)

        # # Add labels and title to the plot
        # plt.xlabel('Principal Components')
        # plt.ylabel('Explained Variance Ratio')
        # plt.title('Scree Plot')

        # Show the plot
        # plt.show(block=False)
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)


        # Find the index where the cumulative variance ratio exceeds 80%
        optimal_index = np.argmax(cumulative_variance_ratio >= 0.8)
        if optimal_index==0 and cumulative_variance_ratio[0]<0.8:
            print('Increase the no. of components. The maximum cumulative variance is ',cumulative_variance_ratio[-1])
        # The optimal number of components is the index + 1
        else:
            optimal_components = optimal_index + 1

            print("The optimal number of components is:", optimal_components)
    return dataframes_pca, pca

def cluster(X,k):
    

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

   

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)

    # Get cluster labels
    labels = kmeans.labels_

    return X_scaled,labels


def pca_cumu_var(input_vector):


    # Compute the cumulative sum
    cumulative_sum = np.cumsum(input_vector)

    # Print the new vector
    print(cumulative_sum)
    # Number of dimensions
    num_dimensions = len(cumulative_sum)

    # Calculate the slope for each dimension
    total_variance = cumulative_sum[-1]
    slope = total_variance / num_dimensions

    # Create x-axis values (number of dimensions)
    x = np.arange(1, num_dimensions + 1)

    # Plot cumulative variance
    plt.plot(x, cumulative_sum, marker='o', linestyle='-', color='b')
    
    # Plot the straight line y=x
    plt.plot(x, slope * x, linestyle='--', color='r')

    # Set labels and title
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Cumulative Variance')
    plt.title('Cumulative Variance Explained by Principal Components')

    # Display the plot
    plt.show(block=False)


    # Compute the derivative
    derivative = np.diff(cumulative_sum) / np.diff(x)

    # Plot the derivative
    plt.plot(x[:-1], derivative, marker='o', linestyle='-', color='b')

    # Set labels and title
    plt.xlabel('Number of Dimensions')
    plt.ylabel('Derivative of Cumulative Variance')
    plt.title('Derivative of Cumulative Variance Explained by Principal Components')

    # Display the plot
    plt.show(block=False)
    return cumulative_sum


def create_null_model(corr_matrix):
    # Step 1: Calculate node strengths
    node_strengths = np.sum(corr_matrix, axis=1)

    # Step 2: Calculate sum total of unique weights in the network
    unique_weights = np.unique(corr_matrix)
    sum_total_weights = np.sum(unique_weights)

    # Step 3: Create null model adjacency matrix
    num_nodes = corr_matrix.shape[0]
    null_model_adj_matrix = np.zeros_like(corr_matrix)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            # Check for self-links and negative values in the correlation matrix
            if i != j and corr_matrix[i, j] >= 0:
                # Calculate edge weight based on node strengths and sum total of weights
                edge_weight = (
                    node_strengths[i] * node_strengths[j]) / sum_total_weights

                # Assign edge weight to null model adjacency matrix
                null_model_adj_matrix[i, j] = edge_weight
                # Undirected graph, so set symmetric values
                null_model_adj_matrix[j, i] = edge_weight

    return null_model_adj_matrix


def elbow_method(min_clusters, max_clusters,data):

    cluster_range = range(min_clusters, max_clusters + 1)

    # Store the sum of squared distances for each number of clusters
    sse = []

    # Perform clustering for different numbers of clusters
    for num_clusters in cluster_range:
        X_scaled, labels = cluster(data, num_clusters)
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(X_scaled)
        sse.append(kmeans.inertia_)
    # Convert the SSE list to a numpy array for easier manipulation
    sse = np.array(sse)
    cluster_range= np.array(list(cluster_range))
    # Find the elbow point using the KneeLocator
    knee = KneeLocator(cluster_range, sse, curve='convex', direction='decreasing')

    # Plot the SSE values and mark the elbow point
    plt.plot(cluster_range, sse)
    plt.xlabel('Number of Clusters')
    plt.ylabel('SSE')
    plt.title('Elbow Method')
    plt.axvline(x=knee.knee, color='r', linestyle='--', label='Elbow Point')
    plt.legend()
    plt.show(block=False)

    # The elbow point can be accessed using the 'knee.knee' attribute
    elbow_point = knee.knee

    # Plot the sum of squared distances for different numbers of clusters
    plt.plot(cluster_range, sse, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distances')
    plt.title('Elbow Method')
    plt.show(block=False)
    return elbow_point


def convert_to_numpy(matrix):
    if isinstance(matrix, np.ndarray):
        return matrix
    elif isinstance(matrix, pd.DataFrame):
        return matrix.values
    else:
        raise ValueError(
            "Unsupported data type. Expected pandas DataFrame or NumPy array.")


def find_rmse(data1, data2):
    if isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray) and data1.dtype == data2.dtype:
        mse = np.mean((data1-data2)**2)
        rmse = np.sqrt(mse)
        return rmse

    else:
        raise ValueError("data1 and data2 must be NumPy arrays of the same type")


def extract_column_values_by_feature_names(file_data, feature_names, column_name):
    """
    Extracts specified column values from a CSV file for specified feature names in "ITM_NAME" column.

  
        
    Args:
        file_data: CSV file 
        feature_names (set): Set of feature names to filter by.
        column_name (str): Name of the column to extract values from.

    Returns:
        dict: A dictionary where keys are feature names and values are lists of values from the specified column.
    """
    # Initialize an empty dictionary to store results
    result_dict = {feature: [] for feature in feature_names}
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_data)
    # Iterate through feature names
    for feature in feature_names:
        # Filter the DataFrame to include only rows where "ITM_NAME" matches the feature name
        filtered_df = df[df['ITM_NAME'] == feature]

        # Extract values from the specified column and store in the result_dict
        result_dict[feature] = filtered_df[column_name].tolist()

    return result_dict


def get_value_by_feature_name(result_dict, feature_name):
    """
    Get the value associated with a specific feature name in a result dictionary.

    Args:
        result_dict (dict): The result dictionary where keys are feature names.
        feature_name (str): The feature name for which you want to retrieve the value.

    Returns:
        Any: The value associated with the specified feature name, or None if not found.
    """
    if feature_name in result_dict:
        return result_dict[feature_name]
    else:
        return None
    

def convert_to_ranks_and_find_range(code_feat):
   #make sure that the code list doesnt include the "unable to measure"=101 code
    code_feat = list(filter(lambda x: x != '101', code_feat))
     # Convert ordinal values to ranks
    ranks = rankdata(code_feat, method='ordinal')

    # Calculate the range
    # Adding 1 to account for the inclusive range
    data_range = max(ranks) - min(ranks) + 1

    return ranks, data_range,code_feat


def project_and_recover(data,space):
    projected_data = np.dot(data, space)
    # Reconstruct the data from the projected data
    data_reconstructed = np.dot(projected_data, space.T)
    rmse= find_rmse(data_reconstructed, data)
    return rmse,data_reconstructed,projected_data

def rank_the_data(ordinal_values,ranks,code_feat,feat_name):
 
    if isinstance(ordinal_values, pd.DataFrame):
        ordinal_values = ordinal_values.to_numpy().flatten()
    
    # Ensure B and C are numpy arrays
    if not isinstance(ranks, np.ndarray):
        ranks = np.array(ranks)
    if not isinstance(code_feat, np.ndarray):
        code_feat = np.array(code_feat, dtype=float)  # Convert C to floats for comparison
    if len(ranks)!=len(code_feat):
        raise ValueError("rank_array and code_dictionary_value array length mismatch ")
    
        # Ensure all values in A are present in C
    missing_values = [a_val for a_val in ordinal_values if a_val not in code_feat]

    if not all(a_val in code_feat for a_val in ordinal_values):
        raise ValueError(f"The data contains values not identified by code dictionary, {missing_values}")
    
     # Create a mapping from the values in code_feat to the corresponding values in ranks
    value_map = {c_val: b_val for c_val, b_val in zip(code_feat, ranks)}
    
    # Replace entries in ordinal values based on the mapping
    ranked_list = [value_map.get(a_val, a_val) for a_val in ordinal_values]
    ranked=np.array(ranked_list)
    return ranked

def NoMedPatnos(df):
    # # Condition for PDTRTMNT being 1 and PDSTATE being 'off'
    # condition_pdtrtmnt_1_pdstate_off = (df['PDTRTMNT'] == 1) & (df['PDSTATE'] == 'off')
    
    # Condition for PDTRTMNT being 0
    condition_pdtrtmnt_0 = df['PDTRTMNT'] == 0
    
    # Combine the conditions using the | operator
    
    
    # Filter the DataFrame using the combined condition
    filtered_df = df[condition_pdtrtmnt_0]
    return filtered_df
def rank_normalization_bsk(dataframes):
    

    # Extract feature names
    feature_names = dataframes.columns

    # Extract 'CODE' from the feature names
    column_to_extract = 'CODE'
    code_list_file = os.path.join(data_dir,'Code_List_-__Annotated_.csv')

    code_dict = extract_column_values_by_feature_names(
        code_list_file,
        feature_names,
        'CODE'
    )
    # Rank normalization for each feature
    dataframes_rank_norm = dataframes.copy()
    for feat_name in feature_names:
        #print('Rank normalizing feature:', feat_name)
        code_feat_101 = get_value_by_feature_name(code_dict, feat_name)
        ranks, data_range, code_feat =convert_to_ranks_and_find_range(code_feat_101)
        ordinal_values = dataframes[feat_name]
        column_ranks = rank_the_data(ordinal_values, ranks, code_feat, feat_name)
        transformed_values = (column_ranks - 1) / (data_range - 1)
        dataframes_rank_norm[feat_name] = transformed_values
    return dataframes_rank_norm

def rank_normalization_bsk_for_ON_plus_OFF(dataframes,columns_to_check,copy_earliest_OFF):
    

    # Extract feature names
    feature_names = dataframes.columns
    original_columns = copy_earliest_OFF.columns

    # Extract 'CODE' from the feature names
    column_to_extract = 'CODE'
    code_list_file = os.path.join(data_dir, 'Code_List_-__Annotated_.csv')
    code_dict = extract_column_values_by_feature_names(code_list_file, original_columns, column_to_extract)

    # Rank normalization for each feature
    dataframes_rank_norm = dataframes.copy()
    for column in feature_names:
        feat_name=column
        ordinal_values = dataframes[feat_name]
        if column in columns_to_check:
            feat_name = feat_name.replace('_ON', '').replace('_OFF', '')

        code_feat_101 = get_value_by_feature_name(code_dict, feat_name)
        ranks, data_range, code_feat =convert_to_ranks_and_find_range(code_feat_101)
        
        column_ranks = rank_the_data(ordinal_values, ranks, code_feat, feat_name)
        transformed_values = (column_ranks - 1) / (data_range - 1)
        if column in columns_to_check:
            feat_name = feat_name + ("_ON" if "_ON" in column else "_OFF")
        dataframes_rank_norm[feat_name] = transformed_values
    return dataframes_rank_norm



def rank_normalize_ON_OFF_space(feature_names,code_dict,updrs3columns,df,matching_columns,np3tot_find=0):
    max_rank_np3tot = 0
    min_rank_np3tot = 0
    for feat_name in feature_names:
        
        code_feat_101 = get_value_by_feature_name(code_dict, feat_name)

        if feat_name in updrs3columns:
            # Initialize an empty set to store the unique results
            code_feat_101 = list(filter(lambda x: x != '101', code_feat_101))
            unique_results = set()

            if not isinstance(code_feat_101, np.ndarray):
                code_feat_101 = np.array(code_feat_101, dtype=float)

            # Compute all possible values of a - b
            for a in code_feat_101:
                for b in code_feat_101:
                    unique_results.add(a - b)

            # Convert the set to a list and sort it
            unique_result_list = sorted(list(unique_results))
            code_feat_101 = unique_result_list

        ranks, data_range, code_feat = convert_to_ranks_and_find_range(
            code_feat_101)

        ordinal_values = df[feat_name]
        

        
        column_ranks = rank_the_data(
            ordinal_values, ranks, code_feat, feat_name)


        transformed_values = (column_ranks - 1) / (data_range - 1)


        df[feat_name] = transformed_values
        

        if feat_name in matching_columns and np3tot_find==1:
            max_rank_np3tot += max(ranks)
            min_rank_np3tot += min(ranks)

    return df,max_rank_np3tot,min_rank_np3tot


def drop_specific_columns(df,specified_columns):
    for column in specified_columns:
        if column in df.columns:
            # Remove the column from the DataFrame
            df.drop(columns=[column], inplace=True)
    return df