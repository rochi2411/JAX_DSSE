import pandas as pd
import matplotlib.pyplot as plt
result=pd.read_excel("Results_DSSE_13NodeIEEE.xlsx")

#print(result)

v1_dss=result['V1(pu)_DSS']
v1_est=result['V1(pu)_EST']
ang_dss=result['Ang1(deg)_DSS']
ang_est=result['Ang1(deg)_EST']
#print(v1_dss.shape)
x=result['Bus Nro.']

plt.plot(x, v1_dss, color = 'g', linestyle ='solid', 
		label = "OpenDSS")
plt.plot(x, v1_est, color = 'b', linestyle ='solid', 
		label = "Estimated")

plt.xticks(rotation = 25) 
plt.xlabel('Bus no.') 
plt.ylabel('Voltage (p.u)') 
plt.title('Bus Voltage', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 

plt.plot(x, ang_dss, color = 'g', linestyle ='solid', 
		label = "OpenDSS")
plt.plot(x, ang_est, color = 'b', linestyle ='solid', 
		label = "Estimated")

plt.xticks(rotation = 25) 
plt.xlabel('Bus no.') 
plt.ylabel('Angle (deg.)') 
plt.title('Bus Angle', fontsize = 20) 
plt.grid() 
plt.legend() 
plt.show() 