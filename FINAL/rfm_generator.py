from datetime import datetime, timedelta  # Provides classes for manipulating dates and times
from polars import Float32, Int64, Float64, Datetime  # Data types from Polars for efficient data handling
from polars import DataType as DataTypes  # Alias for data type access in Polars
from concurrent.futures import ThreadPoolExecutor, as_completed  # Tools for asynchronous execution of callable objects
from collections import OrderedDict  # Dict subclass that remembers the order entries were added
from polars import col, lit  # Functions for referencing dataframe columns and creating literal expressions
import pandas as pd  # Primary data manipulation library in Python
from time import time  # Import time for tracking execution times
import PV as pv  # Custom module, assumed to contain paths or project variables
import numpy as np  # Fundamental package for scientific computing with Python
import polars as pl  # Importing Polars library for dataframe operations
from timeit import default_timer as timer  # Timer for benchmarking purposes
import xlwings as xw  # Library for manipulating Excel files with Python
import Miscl_Func as miscl_func  # Module for miscellaneous functions, details needed
import RFM_Classes as rfm_cls  # Module containing classes for Reduced Form Models
import pyproj  # Library for cartographic projections and geographic transformations
from pyproj import Proj


# pyproj setups for geographic transformations
onv_latlong_UTM = pyproj.Proj(proj='utm', zone=12, ellps='WGS84', preserve_units=True)
myProj = Proj(proj='lcc', lat_1=51.0, lon_0=-115.0, lat_0=53.0, lat_2=57.0, R=6370997.0, x_0=0.0, y_0=0.0, units='m')

# Define transformations and coordinate systems for spatial data operations, essential for accurate location mapping and projections.




def get_grid_name(r, c):
    """
    Retrieve the grid name based on row and column indices.
    
    Parameters:
    r (int): Row index
    c (int): Column index
    
    Returns:
    str: The name of the grid or "GRID" if no specific name is found.
    """
    grid_names = {
        (34, 43): "Edmonton Central",
        (33, 43): "Edmonton South",
        (36, 41): "St. Albert",
        (34, 49): "Androssan",
        (38, 48): "Fort Saskatchewan",
        (39, 48): "Ross Creek",
        (25, 19): "Drayton Valley",
        (50, 8): "Whitecourt",
        (50, 28): "Barrhead",
        (20, 55): "Camrose"
    }
    
    return grid_names.get((r, c), "GRID")


def excel_conc_maps(df_output,top_left,top_right,bottom_left,bottom_right,work_sheet,run_name):
    """
    Populate an Excel worksheet with concentration map data from a dataframe.

    Parameters:
    df_output (DataFrame): The dataframe containing output data.
    top_left, top_right, bottom_left, bottom_right (str): Column names for data positioning.
    work_sheet (object): An xlwings sheet object where data will be written.
    run_name (str): Name of the run for labeling purposes.
    """

    #turn off left as there will be no change
    dfx=miscl_func.convert_to_i_j_format(df_output,top_left)
    work_sheet["A2"].options(pd.DataFrame, header=0, index=False, expand='table').value =dfx
    work_sheet["A1"].value=top_left

    dfx=miscl_func.convert_to_i_j_format(df_output,top_right)
    work_sheet["CJ2"].options(pd.DataFrame, header=0, index=False, expand='table').value =dfx
    work_sheet["CJ1"].value=top_right

    aqms=miscl_func.get_all_stations()
    i=0
    work_sheet["FX5"].value = run_name
    work_sheet["FX6"].value = top_left
    work_sheet["FY6"].value = top_right
    for st in aqms:
        work_sheet["FW" + str(7+i)].value = st[2]
        a=df_output[(df_output['ROW']==st[0]) & (df_output['COL']==st[1])][[top_left,top_right]].values[0]
        work_sheet["FX" + str(7+i)].value = a[0]
        work_sheet["FY" + str(7+i)].value = a[1]
        if a[0]<=0:
            work_sheet["FX" + str(7+i)].value = 0
        if a[1]<=0:
             work_sheet["FY" + str(7+i)].value = 0

        i+=1

    dfx=miscl_func.convert_to_i_j_format(df_output,bottom_left)
    work_sheet["A74"].options(pd.DataFrame, header=0, index=False, expand='table').value =dfx
    work_sheet["A73"].value=bottom_left

    dfx=miscl_func.convert_to_i_j_format(df_output,bottom_right)
    work_sheet["CJ74"].options(pd.DataFrame, header=0, index=False, expand='table').value =dfx
    work_sheet["CJ73"].value=bottom_right

    i=0
    work_sheet["FX82"].value =run_name
    work_sheet["FX84"].value = bottom_left
    work_sheet["FY84"].value = bottom_right
    for st in aqms:
        work_sheet["FW" + str(85+i)].value = st[2]
        a=df_output[(df_output['ROW']==st[0]) & (df_output['COL']==st[1])][[bottom_left,bottom_right]].values[0]
        work_sheet["FX" + str(85+i)].value = a[0]
        work_sheet["FY" + str(85+i)].value = a[1]
        
        if a[0]<=0:
            work_sheet["FX" + str(85+i)].value = 0
        if a[1]<=0:
            work_sheet["FY" + str(85+i)].value = 0
        i+=1


def calc_sens_coeff(rfm_inp:rfm_cls.RFM_Inp_2):
    """
    Calculate and update sensitivity coefficients for a given RFM input object.
    
    Parameters:
    rfm_inp (RFM_Inp_2 object): An object containing input data for the RFM calculation.
    
    Returns:
    RFM_Inp_2: The input object with updated sensitivity coefficients.
    """

    ##PT_IND
    phi_j =float(rfm_inp.phi_j_ind)
    phi_k =float(rfm_inp.phi_k_ind)
    eps_j =abs(float(rfm_inp.pert_fact_pt_ind)) # The absolute value here indiacates that the magnitude of the change plays no role in the sensitivity
    f1_pt_ind= (1+phi_j)*eps_j
    f2_pt_ind= 0.5*(1+phi_j)*(1+phi_j)*eps_j*eps_j

    ##PT_NONIND
    phi_j =float(rfm_inp.phi_j_nonind)
    phi_k =float(rfm_inp.phi_k_nonind)
    eps_j =abs(float(rfm_inp.pert_fact_pt_nonind))
    f1_pt_nonind= (1+phi_j)*eps_j
    f2_pt_nonind= 0.5*(1+phi_j)*(1+phi_j)*eps_j*eps_j

    ##PT_SUOG
    phi_j =float(rfm_inp.phi_j_suog)
    phi_k =float(rfm_inp.phi_k_suog)
    eps_j =abs(float(rfm_inp.pert_fact_pt_suog))
    f1_pt_suog= (1+phi_j)*eps_j
    f2_pt_suog= 0.5*(1+phi_j)*(1+phi_j)*eps_j*eps_j

    ##GR_ONROAD
    phi_j =float(rfm_inp.phi_j_onroad)
    phi_k =float(rfm_inp.phi_k_onroad)
    eps_j =abs(float(rfm_inp.pert_fact_gr_onroad))
    f1_gr_onroad= (1+phi_j)*eps_j
    f2_gr_onroad= 0.5*(1+phi_j)*(1+phi_j)*eps_j*eps_j
    
    rfm_inp.f1_pt_ind=f1_pt_ind
    rfm_inp.f2_pt_ind=f2_pt_ind

    rfm_inp.f1_pt_nonind=f1_pt_nonind
    rfm_inp.f2_pt_nonind=f2_pt_nonind

    rfm_inp.f1_pt_suog=f1_pt_suog
    rfm_inp.f2_pt_suog=f2_pt_suog

    rfm_inp.f1_gr_onroad=f1_gr_onroad
    rfm_inp.f2_gr_onroad=f2_gr_onroad

    return rfm_inp


def process_group(df_group, rfm_inp_updated):
    """
    Process a group of data rows to compute new pollutant concentrations based on RFM parameters.
    
    Parameters:
    df_group (DataFrame): Data for a particular group, typically sliced by columns.
    rfm_inp_updated (RFM_Inp_2 object): RFM input data with updated coefficients.
    
    Returns:
    DataFrame: The modified group DataFrame with new pollutant concentration columns.
    """
    # Iterate over the pollutants
    for poll in rfm_inp_updated.polls:
        # Calculate the delta expressions based on the updated RFM input values
        delta_a_expr = rfm_inp_updated.f1_pt_ind * pl.col(f'{poll}_A1N') + rfm_inp_updated.f2_pt_ind * pl.col(f'{poll}_A2N')
        delta_b_expr = rfm_inp_updated.f1_pt_nonind * pl.col(f'{poll}_B1N') + rfm_inp_updated.f2_pt_nonind * pl.col(f'{poll}_B2N')
        delta_c_expr = rfm_inp_updated.f1_pt_suog * pl.col(f'{poll}_C1N') + rfm_inp_updated.f2_pt_suog * pl.col(f'{poll}_C2N')
        delta_d_expr = rfm_inp_updated.f1_gr_onroad * pl.col(f'{poll}_D1N') + rfm_inp_updated.f2_gr_onroad * pl.col(f'{poll}_D2N')

        # Adjust deltas based on perturbation factors
        delta_a_expr = pl.when(pl.lit(rfm_inp_updated.pert_fact_pt_ind) < 0).then(-delta_a_expr).otherwise(delta_a_expr)
        delta_b_expr = pl.when(pl.lit(rfm_inp_updated.pert_fact_pt_nonind) < 0).then(-delta_b_expr).otherwise(delta_b_expr)
        delta_c_expr = pl.when(pl.lit(rfm_inp_updated.pert_fact_pt_suog) < 0).then(-delta_c_expr).otherwise(delta_c_expr)
        delta_d_expr = pl.when(pl.lit(rfm_inp_updated.pert_fact_gr_onroad) < 0).then(-delta_d_expr).otherwise(delta_d_expr)

        # Create a new column expression by adding the original value and the deltas
        new_col_expr = (pl.col(poll) + delta_a_expr + delta_b_expr + delta_c_expr + delta_d_expr).alias(f'{poll}_NEW')

        # Update the DataFrame for this group with new columns
        df_group = df_group.with_columns([
            delta_a_expr.alias(f'{poll}_DELTA_A'),
            delta_b_expr.alias(f'{poll}_DELTA_B'),
            delta_c_expr.alias(f'{poll}_DELTA_C'),
            delta_d_expr.alias(f'{poll}_DELTA_D'),
            new_col_expr
        ])

    return df_group

output_schema = [
    ("NO", Float32),
    ("NO2", Float32),
    ("O3", Float32),
    ("NO_A1N", Float32),
    ("NO_A2N", Float32),
    ("NO_B1N", Float32),
    ("NO_B2N", Float32),
    ("NO_C1N", Float32),
    ("NO_C2N", Float32),
    ("NO_D1N", Float32),
    ("NO_D2N", Float32),
    ("NO2_A1N", Float32),
    ("NO2_A2N", Float32),
    ("NO2_B1N", Float32),
    ("NO2_B2N", Float32),
    ("NO2_C1N", Float32),
    ("NO2_C2N", Float32),
    ("NO2_D1N", Float32),
    ("NO2_D2N", Float32),
    ("O3_A1N", Float32),
    ("O3_A2N", Float32),
    ("O3_B1N", Float32),
    ("O3_B2N", Float32),
    ("O3_C1N", Float32),
    ("O3_C2N", Float32),
    ("O3_D1N", Float32),
    ("O3_D2N", Float32),
    ("ROW", Int64),
    ("COL", Int64),
    ("DATE", Datetime(time_unit='ns', time_zone=None)),
    ("X_M", Int64),
    ("Y_M", Int64),
    ("LON", Float64),
    ("LAT", Float64),
    # New columns for NO2
    ("NO2_DELTA_A", Float32),
    ("NO2_DELTA_B", Float32),
    ("NO2_DELTA_C", Float32),
    ("NO2_DELTA_D", Float32),
    ("NO2_NEW", Float32),
    # New columns for O3
    ("O3_DELTA_A", Float32),
    ("O3_DELTA_B", Float32),
    ("O3_DELTA_C", Float32),
    ("O3_DELTA_D", Float32),
    ("O3_NEW", Float32),
]


output_schema = OrderedDict(output_schema)

def run_rfm_for_row(df, rfm_inp_updated):
    """
    Run RFM calculations for each row within a DataFrame and apply sensitivity adjustments.
    
    Parameters:
    df (DataFrame): The DataFrame containing rows of air quality data.
    rfm_inp_updated (RFM_Inp_2 object): RFM input data with updated sensitivity coefficients.
    
    Returns:
    DataFrame: The DataFrame with updated pollutant values for each row.
    """
    
    # Use lazy evaluation with groupby and map_groups
    final_df = (
        df.lazy()
        .group_by('COL')
        .map_groups(lambda group: process_group(group, rfm_inp_updated), output_schema)
        .collect()
    )

    return final_df


def summarize_output(r,df_x,polls):
    """
    Summarize output data by calculating geographic and pollution statistics.
    
    Parameters:
    r (int): Row index to process.
    df_x (DataFrame): DataFrame containing pollutant data.
    polls (list): List of pollutants to be summarized.
    
    Returns:
    DataFrame: A DataFrame summarizing max, mean, and other statistics for pollutants.
    """
    lst_x=list()
    x_org, y_org = -74500, -74000
    for c in range(84):
        df = df_x.filter((pl.col('COL') == c))
        x = x_org + c * 4000
        y = y_org + r * 4000
        ln, lt = myProj(x, y, inverse=True)
        
        grid_name = get_grid_name(r, c)
        
        lst = [x, y, ln, lt, r, c, grid_name]
       
        for poll in polls:
            #max hourly
            a1 = 1000 * df.select(pl.col(poll)).quantile(0.99, interpolation= 'linear').to_numpy()[0,0]
            column_name_new = poll + '_NEW'  # Constructs the new column name
            a2 = 1000 * df.select(pl.col(column_name_new).quantile(0.99, interpolation = 'linear')).to_numpy()[0,0]
            lst.append(a1)
            lst.append(a2)


            a1 = df.select(pl.col(poll)).mean().to_numpy()[0,0] * 1000
            # For the second operation, assuming 'poll' needs to be concatenated with '_NEW' to form the column name
            a2 = df.select(pl.col(poll + "_NEW")).mean().to_numpy()[0,0] * 1000
            lst.append(a1)
            lst.append(a2)
            for sect in ['A', 'B', 'C', 'D']:
                sect1N_avg =  df.select(pl.col(poll + "_" + sect + "1N")).mean().to_numpy()[0,0] * 1000
                sect2N_avg =  df.select(pl.col(poll + "_" + sect + "2N")).mean().to_numpy()[0,0] * 1000
                
                lst += [sect1N_avg, sect2N_avg]
                
        lst_x.append(lst)
    cols=list()
    cols.append('X_M')
    cols.append('Y_M')
    cols.append('LON')
    cols.append('LAT')
    cols.append('ROW')
    cols.append('COL')
    cols.append('GRID')
    for poll in polls:
        cols.append(poll + '_MAXHOURLY_PPB')
        cols.append(poll +  '_MAXHOURLY_NEW_PPB')
        cols.append(poll + '_AVG_PPB')
        cols.append(poll +  '_AVG_NEW_PPB')
        for sect in ['A','B','C','D']:
            cols.append(poll +'_' + sect + '1N' )
            cols.append(poll + '_' + sect + '2N' )
            
        
    df=pl.DataFrame(lst_x,schema=cols)
    return df




def run_sequential(excel_path, obj_path):
    """
    Orchestrates the entire process from data input to output, managing RFM calculations and Excel interactions.
    
    Detailed Steps:
    - Initializes environment and reads input parameters from an Excel file.
    - Sets up data structures and performs data preprocessing.
    - Processes data through the RFM model using parallel processing to optimize performance.
    - Outputs results to Excel, including detailed maps of concentration levels and summaries.
    - Handles exceptions and logs errors appropriately to ensure robustness.
    """
    start = timer()
    print ('run single')
    path = excel_path
    wb = xw.Book(path)
    ws_inp = wb.sheets['Input']

    rfm_inp = rfm_cls.RFM_Inp_2()
    rfm_inp.pert_fact_gr_onroad =float(ws_inp.range('B2').value)/100
    rfm_inp.pert_fact_pt_suog =float(ws_inp.range('B3').value)/100
    rfm_inp.pert_fact_pt_ind =float(ws_inp.range('B4').value)/100
    rfm_inp.pert_fact_pt_nonind =float(ws_inp.range('B5').value)/100

    rfm_inp.phi_j_onroad=ws_inp.range('F2').value
    rfm_inp.phi_j_suog=ws_inp.range('F3').value
    rfm_inp.phi_j_ind=ws_inp.range('F4').value
    rfm_inp.phi_j_nonind=ws_inp.range('F5').value

    rfm_inp.phi_k_onroad=ws_inp.range('G2').value
    rfm_inp.phi_k_suog=ws_inp.range('G3').value
    rfm_inp.phi_k_ind=ws_inp.range('G4').value
    rfm_inp.phi_k_nonind=ws_inp.range('G5').value




    polls=list()
    #polls.append('NO')
    polls.append('NO2')
    polls.append('O3')
    rfm_inp.polls=polls

    df_full =pd.read_pickle(obj_path)

    df_full = pl.from_pandas(df_full)

    # Update the rfm_inp with sensitivity coefficients
    rfm_inp_updated = calc_sens_coeff(rfm_inp)

    # Function to apply operation for each value of r
    def process_row(r):
        print(r)
        return r, run_rfm_for_row(df_full.filter(pl.col('ROW') == r), rfm_inp_updated)

    ptime = time()
    # '''
    # Use ThreadPoolExecutor to process rows in parallel
    with ThreadPoolExecutor() as executor:
        # Map each future to its row number for ordered retrieval
        future_to_row = {executor.submit(process_row, r): r for r in range(69)}
        
        # Initialize an ordered dictionary to store the result DataFrames
        lst_1 = OrderedDict()
        
        # As each future completes, store its result in lst_df using the row number as key
        for future in as_completed(future_to_row):
            r = future_to_row[future]
            try:
                r, result_df = future.result()
                lst_1[r] = result_df
            except Exception as exc:
                print(f'Row {r} generated an exception: {exc}')

    # Collect all DataFrames from the ordered dictionary to maintain the processing order
    data_frames = [df for df in lst_1.values()]
    executor.shutdown()
    # Concatenate all resulting DataFrames into a single DataFrame
    df_x = pl.concat(data_frames)

    for col in df_x.columns:
        if pd.api.types.is_numeric_dtype(df_x[col]):
            df_x[col] = df_x[col].astype('float64')

    utime = time()

    print(f"{utime - ptime}")

    # Function to be executed in parallel for each value of r
    def process_and_summarize(r):
        print(r)
        # Assuming summarize_output, df_x, and polls are defined elsewhere
        return r, summarize_output(r, df_x.filter(pl.col('ROW') == r), polls)

    utime = time()

    # Use ThreadPoolExecutor to execute the function in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor and map future to row index
        future_to_row = {executor.submit(process_and_summarize, r): r for r in range(69)}
        
        # Initialize an ordered dictionary to store the results
        lst_2 = OrderedDict()
        
        # Process results as they complete
        for future in as_completed(future_to_row):
            r = future_to_row[future]
            try:
                r, result_data = future.result()
                # Store the result in the ordered dictionary with r as the key
                lst_2[r] = result_data
            except Exception as exc:
                print(f'Row {r} generated an exception: {exc}')
    executor.shutdown()
    # Collect all DataFrames in a list maintaining the order
    data_frames = [df for df in lst_2.values()]

    # Concatenate all resulting DataFrames into a single DataFrame
    # Assuming you're using polars, replace pd.concat with pl.concat if needed
    final_result = pl.concat(data_frames)

    ptime = time()


    df_x = final_result.to_pandas()

    print(f"{ptime - utime}")



    # Push data to excel 
    ws_data = wb.sheets['Data']
    try:
        ws_data["A1"].options(pd.DataFrame, header=1, index=False, expand='table').value = df_x
    except Exception as e:
        print(f"Error writing DataFrame to Excel: {e}")

    ws_inp.range('D2').value=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    ws_conc_no2 = wb.sheets['CONC_NO2']
    ws_conc_o3 = wb.sheets['CONC_O3']
    excel_conc_maps(df_x,'NO2_MAXHOURLY_PPB','NO2_MAXHOURLY_NEW_PPB','NO2_AVG_PPB','NO2_AVG_NEW_PPB',ws_conc_no2,'Run Name')
    excel_conc_maps(df_x,'O3_MAXHOURLY_PPB','O3_MAXHOURLY_NEW_PPB','O3_AVG_PPB','O3_AVG_NEW_PPB',ws_conc_o3,'Run Name')
    no2=np.sum(df_x['NO2_AVG_PPB'])
    no2_new=np.sum(df_x['NO2_AVG_NEW_PPB'])
    o3=np.sum(df_x['O3_AVG_PPB'])
    o3_new=np.sum(df_x['O3_AVG_NEW_PPB'])
    ws_inp["B13"].value = no2
    ws_inp["C13"].value = no2_new
    ws_inp["B14"].value = o3
    ws_inp["C14"].value = o3_new
               
    end = timer()
    print('Time:',timedelta(seconds=end-start))
    



    
