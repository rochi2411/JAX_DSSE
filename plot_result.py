import pandas as pd
import matplotlib.pyplot as plt

# Read the DataFrame from the Excel file
volt_ang_frame = pd.read_excel('volt_ang_frame.xlsx')
volt_ang_frame['Nodes']=volt_ang_frame['Nodes'].astype(str)
# Plotting "Nodes vs V_Actual" and "Nodes vs V_Estimated"
plt.figure(figsize=(10, 6))
plt.plot(volt_ang_frame['Nodes'], volt_ang_frame['V_Actual'], label='Actual', linestyle='solid', color='blue')
plt.plot(volt_ang_frame['Nodes'], volt_ang_frame['V_Estimated'], label='Estimated', linestyle='dashed', color='red')

# Add labels, title, and legend for Voltage
plt.xlabel('Nodes',fontsize=14)
plt.ylabel('Voltage (kV)',fontsize=14)
plt.title('Nodes vs Voltage',fontsize=16)
plt.legend(fontsize=14)
plt.xticks(ticks=range(0, len(volt_ang_frame['Nodes']), 20),fontsize=13)
plt.yticks(fontsize=13)

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(10, 18))

# Phase A: Nodes with ".1"
phase_a = volt_ang_frame[volt_ang_frame['Nodes'].str.endswith('.1')]
axs[0].plot(phase_a['Nodes'], phase_a['Ang_Actual'], label='Actual', linestyle='solid', color='blue')
axs[0].plot(phase_a['Nodes'], phase_a['Ang_Estimated'], label='Estimated', linestyle='dashed', color='red')
axs[0].set_title('Phase A', fontsize=25)
#axs[0].set_xlabel('Nodes', fontsize=14)
axs[0].set_ylabel('Angle (rad.)', fontsize=20)
axs[0].legend(fontsize=17)
axs[0].tick_params(axis='both', labelsize=20)
axs[0].set_xticklabels([])

# Phase B: Nodes with ".2"
phase_b = volt_ang_frame[volt_ang_frame['Nodes'].str.endswith('.2')]
axs[1].plot(phase_b['Nodes'], phase_b['Ang_Actual'], label='Actual', linestyle='solid', color='blue')
axs[1].plot(phase_b['Nodes'], phase_b['Ang_Estimated'], label='Estimated', linestyle='dashed', color='red')
axs[1].set_title('Phase B', fontsize=25)
#axs[1].set_xlabel('Nodes', fontsize=14)
axs[1].set_ylabel('Angle (rad.)', fontsize=20)
axs[1].legend(fontsize=17)
axs[1].tick_params(axis='both', labelsize=20)
axs[1].set_xticklabels([])

# Phase C: Nodes with ".3"
phase_c = volt_ang_frame[volt_ang_frame['Nodes'].str.endswith('.3')]
axs[2].plot(phase_c['Nodes'], phase_c['Ang_Actual'], label='Actual', linestyle='solid', color='blue')
axs[2].plot(phase_c['Nodes'], phase_c['Ang_Estimated'], label='Estimated', linestyle='dashed', color='red')
axs[2].set_title('Phase C', fontsize=25)
axs[2].set_xlabel('Nodes', fontsize=25)
axs[2].set_ylabel('Angle (rad.)', fontsize=20)
axs[2].legend(fontsize=17)
axs[2].tick_params(axis='both', labelsize=20)
axs[2].set_xticklabels([])

# Adjust layout
plt.tight_layout(pad=5.0)

# Show the plot
plt.show()

