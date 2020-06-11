# all frameworks for cpu utilisation forecasting
# all the frameworks make forecasts for start_time to end_time

################################################### correlation based predictions ###################################################

def cluster_map(correlation_matrix, number_of_cluster):
    # takes in correlation matrix and creates a map of cluster index to list of machine index in the cluster
    
    machine_index = correlation_matrix.columns
    number_of_machine = len(machine_index)
    
    # computing the clusters
    cluster = agc(n_clusters=number_of_cluster, affinity='euclidean', linkage='ward')
    clustering_output = cluster.fit_predict(correlation_matrix)
    
    # initialising a map for clusters
    cluster_map = {}
    for x in range(0,number_of_cluster):
        cluster_map[x] = []

    # get a map for each cluster
    # key is cluster index; values is list of machines in that cluster
    for x in range(0,number_of_machine):
        cluster_map[clustering_output[x]].append(machine_index[x])
        
    return cluster_map

def get_best_machine(correlation_matrix):
    # takes in the correlation_matrix
    # returns the index of the machine which minimises the correlation between all machines
    
    # Take the sum of the squared correlation of all machines
    # Note that taking the sum penalizes small values and rewards large values
    squared_matrix = (correlation_matrix**2).sum()
    
    return squared_matrix.idxmax()

def get_regression_parameters(cpu_data, best_machine):
    # get the linear regression parameters for each machine wrt the best machine
    # returns a sklearn.linear_model._base.LinearRegression (model)
    
    model_list = []
    for cpu_index in cpu_data.columns:
        X = np.array(cpu_data[best_machine]).reshape(-1,1)
        model = LinearRegression().fit(X, cpu_data[cpu_index])
        model_list.append(model)
        
    return model_list   

def arima_predictions(timeseries, input_arima_order = (3,0,0)):
    # takes in a timeseries
    # outputs a single prediction in the next timestep
    
    timeseries = timeseries.values
    model = ARIMA(timeseries, order = input_arima_order)
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0][0]
    
    return prediction

def get_cluster_predictions(cpu_data, curr_time_segment, cluster_corr_matrix, input_arima_order):
    # takes in information about a cluster 
    # cpu_data is the full length of values of all the timeseries IN THE CLUSTER
    # curr_time_segment is the windowed version of cpu_data
    
    # makes predictions for all machines in cluster at next single timestep
    # returns a map of machine to predictions
    
    # get best machine
    best_machine = get_best_machine(cluster_corr_matrix)
    
    # get_regression_parameters wrt best machine for every machine
    linreg_model_list = get_regression_parameters(curr_time_segment, best_machine)
    
    # make predictions on best machine
    best_machine_prediction = arima_predictions(cpu_data[best_machine], input_arima_order = input_arima_order)
    
    # scale all the other machines to fit the best machine
    cluster_prediction = dict()
    for index, machine_index in enumerate(cpu_data.columns):
        current_prediction = linreg_model_list[index].predict(np.array(best_machine_prediction).reshape(-1,1))[0]
        
        # if the prediction is out of bounds, use arima to predict instead
        if(current_prediction > 1 or current_prediction < 0):
            current_prediction = arima_predictions(cpu_data[machine_index], input_arima_order = input_arima_order)
            
        # print("prediction: ", linreg_model_list[index].predict(np.array(best_machine_prediction).reshape(-1,1))[0])
        cluster_prediction[machine_index] = current_prediction
        
    return cluster_prediction    

def corr_predictions(cpu_data, window = 6, number_of_cluster = 50, start_time = 288, end_time = None, rolling_error_window = 0):

    ''' high level function for making correlation based single-step predictions

        param:  cpu_data:   df with CPU index as columns and time-step as rows, must contain time-step for at least [start_time-3:end_time-2]
                            ARIMA models are trained with all time-steps before the "prediction time-step" i.e. prediction x[n] is made using data from x[:n]
                            Predictions are made for all machines index included in cpu_data, to exclude predicting certain machine index, drop from cpu_data
        param:  window:     window size of rolling correlation function
        param:  number_of_cluster : number of clusters used for clustering per time-step
        param:  start_time, end_time :  prediction is made for from start_time to (end_time - 1) inclusive
                                     :  keep end_time = None to make predictions for the entire length of cpu_data i.e. predictions made from x[start_time:]
        param: rolling_error_window  :  predictions are offset using a rolling error window of size (defined here) using past errors   

    returns : df matrix of prediction with columns as machine and rows as time index
    '''

    rolling_error_multiplier = 0.25
    arima_order = (3,0,0)
    machine_index = cpu_data.columns
    map_min_corr = {}
    number_of_machine = len(cpu_data.columns)
    all_predictions = pd.DataFrame(columns = cpu_data.columns)
    
    # df_rolling_error has the prediction error for the last 5 time-step for each machine
    df_rolling_error = pd.DataFrame(columns = cpu_data.columns)
    
    if end_time == None:
        end_time = len(cpu_data.index) - 1
    
    with progressbar.ProgressBar(max_value = end_time - start_time) as bar:
        for current_time in range(start_time, end_time):
    
            bar.update(current_time-start_time)

            # computing the correlation matrix at current time index
            curr_time_segment = cpu_data.iloc[current_time-window:current_time]
            curr_all_time = cpu_data.iloc[:current_time]
            curr_corr_matrix = curr_time_segment.corr().abs()
            

            # perform clustering at current time index
            cluster = cluster_map(curr_corr_matrix, number_of_cluster = number_of_cluster) 

            # initialise dict for machine to prediction in current timestep
            curr_all_machine_pred = dict()

            for ls_machine_in_cluster in cluster.values():
                # make predictions for all machine in each cluster
                
                
                cluster_predictions = get_cluster_predictions(cpu_data = curr_all_time[ls_machine_in_cluster],
                                                              curr_time_segment = curr_time_segment[ls_machine_in_cluster],
                                                              cluster_corr_matrix = curr_corr_matrix[ls_machine_in_cluster].loc[ls_machine_in_cluster],
                                                              input_arima_order = arima_order)
                
                # curr_all_machine_pred is a dict with key = machine, value = current timestep prediction
                curr_all_machine_pred = {**curr_all_machine_pred, **cluster_predictions}

            if rolling_error_window > 0:
        
                if len(df_rolling_error) == 0:
                    # 1st prediction does not have a rolling error
                    df_current_predictions = pd.DataFrame(curr_all_machine_pred, index = [current_time])
                else:
                    
                    # get final predictions by adding in the rolling error
                    df_current_predictions = pd.DataFrame(curr_all_machine_pred, index = [current_time]) + rolling_error_multiplier * df_rolling_error.mean()

                # update the rolling error df for next time-step
                df_rolling_error = df_rolling_error.append(cpu_data.iloc[current_time] - df_current_predictions)

                # remove nunwanted history from the rolling error df
                if len(df_rolling_error) > rolling_error_window:
                    df_rolling_error = df_rolling_error.drop(df_rolling_error.index[0])
            
            else:
                df_current_predictions = pd.DataFrame(curr_all_machine_pred, index = [current_time])
                
            all_predictions = all_predictions.append(df_current_predictions, sort = True)
        
    return all_predictions

################################################### mse based predictions ###################################################

def kmeans_cluster_map(cpu_t, number_of_cluster = 50):
    # takes in cpu at a single time index and creates a map of cluster index to list of machine index in the cluster
    # cpu_t is a pd.series with values of cpu at time t 
    
    # get machine index
    machine_index = cpu_t.columns
    number_of_machine = len(machine_index)
    
    # converting cpu_t to appropriate data structure
    cpu_t = cpu_t.mean().values.reshape(-1,1)
    
    # computing the clusters
    kmeans = KMeans(n_clusters=number_of_cluster, random_state = 0)
    clustering_output = kmeans.fit(cpu_t).labels_
    
    # initialising a map for clusters
    cluster_map = {}
    for x in range(0,number_of_cluster):
        cluster_map[x] = []

    # get a map for each cluster
    # key is cluster index; values is list of machines in that cluster
    for x in range(0,number_of_machine):
        cluster_map[clustering_output[x]].append(machine_index[x])
        
    return cluster_map

def arima_predictions(timeseries, input_arima_order = (3,0,0)):
    # takes in a timeseries
    # outputs a single prediction in the next timestep
    
    timeseries = timeseries.values
    model = ARIMA(timeseries, order = input_arima_order)
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0][0]
    
    return prediction

def mse_cluster_prediction(cpu_data, input_arima_order, past_error):
    # generate predictions for all machines in cluster at next single timestep
    # returns a map of machine to predictions
    
    # get generalisation model (analogous to best_machine in correlation_prediction)
    general_model = cpu_data.mean(axis = 1)
    
    # make predictions on best machine
    general_model_prediction = arima_predictions(general_model, input_arima_order = input_arima_order)
    
    # get the rolling_error
    rolling_error = past_error.mean()
    
    # scale general_model to fit all the other machines
    cluster_prediction = dict()
    for index, machine_index in enumerate(cpu_data.columns):
        cluster_prediction[machine_index] = general_model_prediction + rolling_error[machine_index]
        
    return cluster_prediction    

def mse_predictions(cpu_data, number_of_cluster = 50, start_time = 288, end_time = None, past_error_range = 2, rolling_cluster_window = 3):

    ''' high level function for making predictons using the tuor framework

        param:  cpu_data:   df with CPU index as columns and time-step as rows, must contain time-step for at least [start_time-3:end_time-2]
                            ARIMA models are trained with all time-steps before the "prediction time-step" i.e. prediction x[n] is made using data from x[:n]
                            Predictions are made for all machines index included in cpu_data, to exclude predicting certain machine index, drop from cpu_data
        param:  rolling_cluster_window:     number of past values to be considered for clustering at each time-step
        param:  number_of_cluster : number of clusters used for clustering per time-step
        param:  start_time, end_time :  prediction is made for from start_time to (end_time - 1) inclusive
                                     :  keep end_time = None to make predictions for the entire length of cpu_data i.e. predictions made from x[start_time:]
        param: past_error_range  :  predictions are offset using a rolling error window of size (defined here) using past errors   

    returns : df matrix of prediction with columns as machine and rows as time index
    '''

    arima_order = (3,0,0)
    machine_index = cpu_data.columns
    number_of_machine = len(cpu_data.columns)
    all_predictions = pd.DataFrame(columns = cpu_data.columns)
    df_past_error = pd.DataFrame(0, columns = cpu_data.columns, index = np.arange(0,past_error_range))
    
    if end_time == None:
        end_time = len(cpu_data.index)
    
    with progressbar.ProgressBar(max_value = end_time-start_time) as bar:
        for current_time in range(start_time, end_time):
    
            bar.update(current_time-start_time)

            # perform clustering at (current time - 1) index
            # note clustering is only done using last "rolling_cluster_window" points
            cluster = kmeans_cluster_map(cpu_data.iloc[current_time - rolling_cluster_window: current_time], 
                                         number_of_cluster = number_of_cluster)

            # initialise dict for machine to prediction in current timestep
            curr_all_machine_pred = {}

            for ls_machine_in_cluster in cluster.values():
                
                # make predictions for all machine in each cluster
                cluster_predictions = mse_cluster_prediction(cpu_data = cpu_data[ls_machine_in_cluster], 
                                                             input_arima_order = (3,0,0), 
                                                             past_error = df_past_error[ls_machine_in_cluster])
                
                # curr_all_machine_pred is a dict with key = machine, value = current timestep prediction
                curr_all_machine_pred = {**curr_all_machine_pred, **cluster_predictions}

            # update df_past_error with most updated time index
            # start by getting the current error
            curr_all_machine_error = cpu_data.iloc[current_time] - pd.DataFrame(curr_all_machine_pred, index = [current_time])
            
            df_past_error = df_past_error.append(curr_all_machine_error, 
                                                 ignore_index = True)
            df_past_error = df_past_error.drop(0).reset_index().drop("index", axis = 1)
            
            # append the current predicition to all the predictions
            current_df = pd.DataFrame(curr_all_machine_pred, index = [current_time])
            
            all_predictions = all_predictions.append(current_df, sort = True)
        
    return all_predictions

################################################### brut4e force method ###################################################

def one_timeseries_one_model(cpu_data, start_time, end_time):

    ''' using one arima model to make predictions for each time-series

        param:  cpu_data:   df with CPU index as columns and time-step as rows, must contain time-step for at least [start_time-3:end_time-2]
                            ARIMA models are trained with all time-steps before the "prediction time-step" i.e. prediction x[n] is made using data from x[:n]
                            Predictions are made for all machines index included in cpu_data, to exclude predicting certain machine index, drop from cpu_data
        param:  start_time, end_time :  prediction is made for from start_time to (end_time - 1) inclusive
                                     :  keep end_time = None to make predictions for the entire length of cpu_data i.e. predictions made from x[start_time:]

    returns : df matrix of prediction with columns as machine and rows as time index
    '''

    input_order = (3,0,0)
    
    # prediction map is the dict for all predictions where the key is the machine index and the value is a list of predictions
    predictions_map = dict()
    with progressbar.ProgressBar(max_value = len(cpu_data.columns)) as bar:     
        
        for number_of_machines, cpu_index in enumerate(cpu_data):
            bar.update(number_of_machines)
            
            temp_predictions = list()
            for current_time in range (start_time, end_time):
                
                model = ARIMA(cpu_data[cpu_index].iloc[:current_time], order = input_order)
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                temp_predictions.append(yhat[0])
                
            predictions_map[cpu_index] = temp_predictions
            
    return pd.DataFrame(predictions_map, columns = cpu_data.columns, index = np.arange(start_time, end_time))  

################################################### naive method ###################################################

def lagged_timeseries(cpu_data, start_time, end_time):
    ''' 
    Naive method: Taking the observation from the last time-step as the prediction for the next time-step
    Implementation is just reindexing of the required segment of time-series
    '''
    
    rename_mapping = dict()
    for x in range(start_time, end_time):
        rename_mapping[x-1] = x
    
    return cpu_data.iloc[start_time-1:end_time-1].rename(index = rename_mapping)