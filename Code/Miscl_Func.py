import os
import glob
import pandas as pd
import netCDF4
from netCDF4 import Dataset
import numpy as np
import pickle
from joblib import Parallel, delayed
from dateutil.rrule import rrule, DAILY
from datetime import datetime,date,timedelta
import PV as pv


def get_all_stations():
    lst=list()
    lst.append([34,43,"Edmonton Central"])
    lst.append([33, 43, "Edmonton South"])
    lst.append([36, 41, "St. Albert"])
    lst.append([34, 49, "Androssan"])
    lst.append([38, 48, "Fort Saskatchewan"])
    lst.append([39, 48, "Ross Creek"])
    lst.append([25, 19, "Drayton Valley"])
    lst.append([50, 8, "Whitecourt"])
    lst.append([50, 28, "Barrhead"])
    lst.append([20, 55, "Camrose"])
    return lst

def get_station(r,c):
    lst=list()
    lst.append([34,43,"Edmonton Central"])
    lst.append([33, 43, "Edmonton South"])
    lst.append([36, 41, "St. Albert"])
    lst.append([34, 49, "Androssan"])
    lst.append([38, 48, "Fort Saskatchewan"])
    lst.append([39, 48, "Ross Creek"])
    lst.append([25, 19, "Drayton Valley"])
    lst.append([50, 8, "Whitecourt"])
    lst.append([50, 28, "Barrhead"])
    lst.append([20, 55, "Camrose"])
    for i in lst:
        if i[0]==r and i[1]==c:
            return i[2]
    return 'NONE'

def get_cell_color_no2_hourly(cell_value):
    if cell_value<=20:
        return '#008000'
    elif cell_value<=30:
        return '#FFFF00'
    elif cell_value<=60:
        return '#FFA500'
    else:
        return '#FF0000'
    

def get_cell_color(poll_avg_time,conc):
    if poll_avg_time in ['NO2_MAX_HOURLY_PPB','NO2_NEW_MAX_HOURLY_PPB']:
        if conc<=20:
            return '#008000'
        elif conc<=30:
            return '#FFFF00'
        elif conc<=60:
            return '#FFA500'
        else:
            return '#FF0000'
    elif poll_avg_time in ['NO2_ANN_AVG_PPB','NO2_NEW_ANN_AVG_PPB']:
        if conc<=2:
            return '#008000'
        elif conc<=7:
            return '#FFFF00'
        elif conc<=17:
            return '#FFA500'
        else:
            return '#FF0000'
    elif poll_avg_time=='o3_8hr':
        if conc<=50:
            return '#008000'
        elif conc<=57:
            return '#FFFF00'
        elif conc<=62:
            return '#FFA500'
        else:
            return '#FF0000'




def get_cell_color_given_breaks(breaks,colors,cell_value):
    for i in range(len(breaks)):
        if cell_value<=breaks[i]:
            return colors[i]
        
def get_cell_ref(r,c,delta_row,delta_col):
    col_refs = get_excel_col_refs()
    for row in range(69):
        for col in range(84):
            if row==r and col==c:
                row_x=69-row+1
                return (col_refs[col+delta_col] + str(row_x+delta_row))
                
    print ('none')

def get_cell_ref_2(r,c):
    values=get_alpha()
    
    for row in range(69):
        for col in range(84):
            if row==r and col==c:
                row_x=69-row
                return (values[col] + str(row_x))
                
    print ('none')
        
def get_cell_ref_2(r,c):
    values=get_alpha()
    for row in range(68,-1,-1):
        for col in range(84):
            if row==r and col==c:
                return (values[col] + str(row+2))
    print ('none')
        
                    

def get_excel_col_refs():
    col_refs =list()
    col_refs.append('A')
    col_refs.append('B')
    col_refs.append('C')
    col_refs.append('D')
    col_refs.append('E')
    col_refs.append('F')
    col_refs.append('G')
    col_refs.append('H')
    col_refs.append('I')
    col_refs.append('J')
    col_refs.append('K')
    col_refs.append('L')
    col_refs.append('M')
    col_refs.append('N')
    col_refs.append('O')
    col_refs.append('P')
    col_refs.append('Q')
    col_refs.append('R')
    col_refs.append('S')
    col_refs.append('T')
    col_refs.append('U')
    col_refs.append('V')
    col_refs.append('W')
    col_refs.append('X')
    col_refs.append('Y')
    col_refs.append('Z')
    col_refs.append('AA')
    col_refs.append('AB')
    col_refs.append('AC')
    col_refs.append('AD')
    col_refs.append('AE')
    col_refs.append('AF')
    col_refs.append('AG')
    col_refs.append('AH')
    col_refs.append('AI')
    col_refs.append('AJ')
    col_refs.append('AK')
    col_refs.append('AL')
    col_refs.append('AM')
    col_refs.append('AN')
    col_refs.append('AO')
    col_refs.append('AP')
    col_refs.append('AQ')
    col_refs.append('AR')
    col_refs.append('AS')
    col_refs.append('AT')
    col_refs.append('AU')
    col_refs.append('AV')
    col_refs.append('AW')
    col_refs.append('AX')
    col_refs.append('AY')
    col_refs.append('AZ')
    col_refs.append('BA')
    col_refs.append('BB')
    col_refs.append('BC')
    col_refs.append('BD')
    col_refs.append('BE')
    col_refs.append('BF')
    col_refs.append('BG')
    col_refs.append('BH')
    col_refs.append('BI')
    col_refs.append('BJ')
    col_refs.append('BK')
    col_refs.append('BL')
    col_refs.append('BM')
    col_refs.append('BN')
    col_refs.append('BO')
    col_refs.append('BP')
    col_refs.append('BQ')
    col_refs.append('BR')
    col_refs.append('BS')
    col_refs.append('BT')
    col_refs.append('BU')
    col_refs.append('BV')
    col_refs.append('BW')
    col_refs.append('BX')
    col_refs.append('BY')
    col_refs.append('BZ')
    col_refs.append('CA')
    col_refs.append('CB')
    col_refs.append('CC')
    col_refs.append('CD')
    col_refs.append('CE')
    col_refs.append('CF')
    col_refs.append('CG')
    col_refs.append('CH')
    col_refs.append('CI')
    col_refs.append('CJ')
    col_refs.append('CK')
    col_refs.append('CL')
    col_refs.append('CM')
    col_refs.append('CN')
    col_refs.append('CO')
    col_refs.append('CP')
    col_refs.append('CQ')
    col_refs.append('CR')
    col_refs.append('CS')
    col_refs.append('CT')
    col_refs.append('CU')
    col_refs.append('CV')
    col_refs.append('CW')
    col_refs.append('CX')
    col_refs.append('CY')
    col_refs.append('CZ')
    col_refs.append('DA')
    col_refs.append('DB')
    col_refs.append('DC')
    col_refs.append('DD')
    col_refs.append('DE')
    col_refs.append('DF')
    col_refs.append('DG')
    col_refs.append('DH')
    col_refs.append('DI')
    col_refs.append('DJ')
    col_refs.append('DK')
    col_refs.append('DL')
    col_refs.append('DM')
    col_refs.append('DN')
    col_refs.append('DO')
    col_refs.append('DP')
    col_refs.append('DQ')
    col_refs.append('DR')
    col_refs.append('DS')
    col_refs.append('DT')
    col_refs.append('DU')
    col_refs.append('DV')
    col_refs.append('DW')
    col_refs.append('DX')
    col_refs.append('DY')
    col_refs.append('DZ')
    col_refs.append('EA')
    col_refs.append('EB')
    col_refs.append('EC')
    col_refs.append('ED')
    col_refs.append('EE')
    col_refs.append('EF')
    col_refs.append('EG')
    col_refs.append('EH')
    col_refs.append('EI')
    col_refs.append('EJ')
    col_refs.append('EK')
    col_refs.append('EL')
    col_refs.append('EM')
    col_refs.append('EN')
    col_refs.append('EO')
    col_refs.append('EP')
    col_refs.append('EQ')
    col_refs.append('ER')
    col_refs.append('ES')
    col_refs.append('ET')
    col_refs.append('EU')
    col_refs.append('EV')
    col_refs.append('EW')
    col_refs.append('EX')
    col_refs.append('EY')
    col_refs.append('EZ')
    col_refs.append('FA')
    col_refs.append('FB')
    col_refs.append('FC')
    col_refs.append('FD')
    col_refs.append('FE')
    col_refs.append('FF')
    col_refs.append('FG')
    col_refs.append('FH')
    col_refs.append('FI')
    col_refs.append('FJ')
    col_refs.append('FK')
    col_refs.append('FL')
    col_refs.append('FM')
    col_refs.append('FN')
    col_refs.append('FO')
    col_refs.append('FP')
    col_refs.append('FQ')
    return col_refs



def get_alpha():
    values =list()
    values.append('A')
    values.append('B')
    values.append('C')
    values.append('D')
    values.append('E')
    values.append('F')
    values.append('G')
    values.append('H')
    values.append('I')
    values.append('J')
    values.append('K')
    values.append('L')
    values.append('M')
    values.append('N')
    values.append('O')
    values.append('P')
    values.append('Q')
    values.append('R')
    values.append('S')
    values.append('T')
    values.append('U')
    values.append('V')
    values.append('W')
    values.append('X')
    values.append('Y')
    values.append('Z')
    values.append('AA')
    values.append('AB')
    values.append('AC')
    values.append('AD')
    values.append('AE')
    values.append('AF')
    values.append('AG')
    values.append('AH')
    values.append('AI')
    values.append('AJ')
    values.append('AK')
    values.append('AL')
    values.append('AM')
    values.append('AN')
    values.append('AO')
    values.append('AP')
    values.append('AQ')
    values.append('AR')
    values.append('AS')
    values.append('AT')
    values.append('AU')
    values.append('AV')
    values.append('AW')
    values.append('AX')
    values.append('AY')
    values.append('AZ')
    values.append('BA')
    values.append('BB')
    values.append('BC')
    values.append('BD')
    values.append('BE')
    values.append('BF')
    values.append('BG')
    values.append('BH')
    values.append('BI')
    values.append('BJ')
    values.append('BK')
    values.append('BL')
    values.append('BM')
    values.append('BN')
    values.append('BO')
    values.append('BP')
    values.append('BQ')
    values.append('BR')
    values.append('BS')
    values.append('BT')
    values.append('BU')
    values.append('BV')
    values.append('BW')
    values.append('BX')
    values.append('BY')
    values.append('BZ')
    values.append('CA')
    values.append('CB')
    values.append('CC')
    values.append('CD')
    values.append('CE')
    values.append('CF')
    return values





def convert_to_i_j_format(df,poll):
    df=df[['ROW','COL',poll]]
    lst=list()
    for row in range(68,-1,-1):
        df_1=df[df['ROW']==row][poll].values
        lst.append(np.round(df_1,2))
    
    cols=list()
    for i in range(84):
        cols.append(i)

    df_o = pd.DataFrame(lst,columns=cols)
    return df_o

def convert_to_i_j_format_2(df,poll):
    df=df[['ROW','COL',poll]]
    lst=list()
    for row in range(68,-1,-1):
        df_1=df[df['ROW']==row][poll].values
        lst.append(np.round(df_1,6))
    
    df_o = pd.DataFrame(lst)
    return df_o

def convert_to_i_j_format_3(df,poll):
    df=df[['ROW','COL',poll]]
    lst_1=list()
    lst_2=list()
    lst_3=list()
    for row in range(68,-1,-1):
        lst_vals=list()
        lst_clrs=list()
        lst_cellrefs=list()
        for col in range(83,-1,-1):
            a=df[(df['ROW']==row) & (df['COL']==col)][poll].values[0]
            clr= "'" + get_cell_color(poll,a) + "'"
            lst_vals.append(a)
            lst_clrs.append(clr)
            lst_cellrefs.append(get_cell_ref(row,col,0,0))
        lst_1.append(lst_vals)
        lst_2.append(lst_clrs)
        lst_3.append(lst_cellrefs)
    df_1=pd.DataFrame(lst_1)
    df_2=pd.DataFrame(lst_2)
    df_3=pd.DataFrame(lst_3)
    return df_1,df_2,df_3
        
           

            
    
    df_o = pd.DataFrame(lst)
    return df_o


def convert_netcdf_df_1(yyyy_mm_dd):
    polls=['NO','NO2','O3']
    sens=['A1N','A2N','B1N','B2N','C1N','C2N','D1N','D2N']
    inp_date=yyyy_mm_dd.replace('-','')
    f1= dir + '/CCTM_ACONC_v54_DDM3D_gcc_aep_diz_4km_' + inp_date + '.nc'
    f2= dir + '/CCTM_ASENS_v54_DDM3D_gcc_aep_diz_4km_' + inp_date + '.nc'
    dataset1 = Dataset(f1)
    dataset2 = Dataset(f2)
    
    for h in range(24):
        lst_df1=list()
        print (h)
        for i in range(69):
            df1 = pd.DataFrame()
            df1['DATE']= [yyyy_mm_dd]*84
            df1['HOUR']= [h]*84
            df1['ROW']= [i]*84
            df1['COL']= range(84)
            for p in polls:
                df1[p]=dataset1.variables[p][h,0,i,:]
                for sen in sens:
                    x= p + '_' + sen
                    df1[x]=dataset2.variables[x][h,0,i,:]
                lst_df1.append(df1)
        df2=pd.concat(lst_df1)
    return df2


def convert_netcdf_df():
 
    d1 = date(2018, 1, 1)
    d2 = date(2018, 12, 31)
    lst1=list()
    for dt in rrule(DAILY, dtstart=d1, until=d2):
        yyyy_mm_dd=str(dt.year) + '-' + str(dt.month).zfill(2) + '-' + str(dt.day).zfill(2)
        lst1.append(yyyy_mm_dd)
    
    
    n=10
    chunks = [lst1[i:i+n] for i in range(0,len(lst1),n)]
    for lst_chunk in chunks:
        for chunk in lst_chunk:
            df=convert_netcdf_df_1(chunk)
            df.to_excel(dir + '/test.xlsx')
            exit()

def test1():
    f=r"D:\ddm_2018_q1_sens_nox\ACONC\merged"
    dataset1 = Dataset(f)
    a=dataset1.variables['O3'][:,0,0,0]
    print (len(a))


def test2():
    directory_path ="D:/ddm_2018_q1_sens_nox/ASENS"
    file_extension="CCTM*"
    search_pattern = os.path.join(directory_path, f'*{file_extension}')
    matching_files = glob.glob(search_pattern)
    dataset1 = netCDF4.MFDataset(matching_files)
    key = list(dataset1.variables.keys())[1]
    rows=len(dataset1.variables[key][0,0,:,0])
    cols=len(dataset1.variables[key][0,0,0,:])
    
    polls=['NO','NO2','O3']
    sens=['A1N','A2N','B1N','B2N','C1N','C2N','D1N','D2N']
    df=pd.DataFrame()
    for poll in polls:
        for sen in sens:
            lst=list()
            for r in range(rows):
                for c in range(cols):
                    a=dataset1.variables[poll + '_' + sen][:,0,r,0]
                    lst.extend(a)
            print (lst)
            exit()
    print (df.columns)
    exit()
            
    
    key = list(dataset1.variables.keys())[1]
    a=dataset1.variables[key][:,0,0,0]
    print (len(a))
    
def convert_aconc_asens_to_df_2():
    # Use merged files (created by CDO mergetime) for concentrations and sensitivities

    # 1. Open the netCDF file for concentrations
    f = r"D:\ddm_2018_q1_sens_nox_2\ACONC_Jan_Feb_Mar_2018_2"
    ds_aconc = Dataset(f)
    key = list(ds_aconc.variables.keys())[2]
    rows = len(ds_aconc.variables[key][0, 0, :, 0])
    cols = len(ds_aconc.variables[key][0, 0, 0, :])

    # 2. Open the netCDF file for sensitivity data
    f = r"D:\ddm_2018_q1_sens_nox_2\ASENS_Jan_Feb_Mar_2018_2"
    ds_asens = Dataset(f)

    # 3. Create an empty list to hold DataFrames for each grid column
    ret_lst = list()

    # 4. Loop over each column in the grid
    for col in range(cols):
        print(col)
        # For each column, convert the data from both datasets into a DataFrame.
        # The helper function convert_aconc_asens_to_df_x handles the conversion for one column.
        df = convert_aconc_asens_to_df_x(col, ds_aconc, ds_asens)
        ret_lst.append(df)

    # 5. Concatenate all DataFrames from all columns into one large DataFrame
    df_x = pd.concat(ret_lst)

    # 6. Save the final DataFrame to a pickle file (.obj file)
    pd.to_pickle(df_x, pv.rfm_dir + '/input/df_jan_feb_mar_2018_from_merged_new_2.obj')
