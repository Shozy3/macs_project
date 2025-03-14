from datetime import datetime,date,timedelta


rfm_dir=r"C:\Users\Shahzaib.Ahmed\OneDrive - Government of Alberta\Documents\MACS Project\RFM"
polls=['NO_A1N','NO2_A1N','O3_A1N']
sens_coeff=['A1N','A2N','B1N','B2N','C1N','C2N','D1N','D2N']

st_day = datetime(2018,1,1,0)
en_day = datetime(2018,3,31,23)

#colors_seven=['#fbe1df','#f3a59e','#eb685d','#e32c1c','#a22014','#61130c','#200604']
colors_seven=['#fcecea','#f7c5c1','#f29f98','#ed796f','#e85345','#e32c1c','#ba2417']

colors_seven_o3=['#f5eafc','#e2c1f7','#ce98f2','#ba6fed','#a745e8','#931ce3','#7817ba']

breaks_no2_hourly=[20,30,60,500]
colors_no2_hourly=['#008000','#FFFF00','#FFA500','#FF0000']

breaks_o3_hourly=[50,56,62,500]
colors_o3_hourly=['#008000','#FFFF00','#FFA500','#FF0000']


breaks_no2_ann_avg=[2.0,7.0,17.0,500]
colors_no2_ann_avg=['#008000','#FFFF00','#FFA500','#FF0000']

