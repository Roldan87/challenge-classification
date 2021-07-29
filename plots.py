#DF split per experiments:

control_exp = bear_sign[bear_sign['experiment_id'] == 1]

for i in range(2,113):
    locals()['exp_'+str(i)] = bear_sign[bear_sign['experiment_id'] == i]

#Plots:

plt.plot(control_exp.timestamp.iloc[:62000], control_exp.hz.iloc[:62000])
plt.title('Control Exp / Speed (Hz)')

plt.plot(control_exp.timestamp.iloc[:62000], control_exp.w.iloc[:62000])
plt.title('Control Exp / Power (Watts)')

# To visualize other Axis just replace exp_105.a2_X/Y and exp_26.a2_X/Y:

plt.plot(control_exp.timestamp.iloc[:62000], control_exp.a1_z.iloc[:62000], color='r', alpha=0.5)
plt.plot(exp_105.timestamp.iloc[:62000], exp_105.a2_z.iloc[:62000], color='g', alpha=0.4)
plt.plot(exp_26.timestamp.iloc[:62000], exp_26.a2_z.iloc[:62000], color='b', alpha=0.3)
plt.plot(control_exp.timestamp.iloc[:62000], control_exp.w.iloc[:62000], color='y')
plt.title('Power (W) / Z axis: Control (red) vs Good 105 (green) vs Bad 26 (blue')