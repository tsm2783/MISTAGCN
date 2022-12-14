import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

from share import epochs

scenarios = ['Haikou', 'YellowTrip']

compare_metrics = []

plt.figure()
plt.rcParams.update({'font.size': 20, 'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18})

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 14))

for i, (scenario) in enumerate(scenarios):
    scenario_dir = 'results-in-model/effect-graphs/' + scenario + '/'
    csv_files = glob.glob(os.path.join(scenario_dir, "*.xlsx"))
    csv_files.sort()

    # ---------load results data-------------

    for j, (f) in enumerate(csv_files):
        df_train = pd.read_excel(f, sheet_name='train_records')
        df_validate = pd.read_excel(f, sheet_name='validate_records')
        df_test = pd.read_excel(f, sheet_name='test_records')

        column_start = len(scenario_dir)
        column_end = - len('-results.xlsx')
        column = f[column_start:column_end]
        column = column.upper()
        column = column.replace('-','(',1)
        column = column.replace('-',',')
        column = column + ')'

        if j == 0:
            df_train_rmse = pd.DataFrame({column: df_train['RMSE']})
            df_validate_mae = pd.DataFrame({column: df_validate['MAE']})
            df_validate_rmse = pd.DataFrame({column: df_validate['RMSE']})
            df_test_rmse = pd.DataFrame({column: df_test['RMSE']})
        else:
            df_train_rmse[column] = df_train['RMSE']
            df_validate_mae[column] = df_validate['MAE']
            df_validate_rmse[column] = df_validate['RMSE']
            df_test_rmse[column] = df_test['RMSE']

    # ---------extract comparison data-------------

    s_mae = df_validate_mae.iloc[-1, :]
    s_mae.name = 'MAE'
    s_rmse = df_validate_rmse.iloc[-1, :]
    s_rmse.name = 'RMSE'
    compare_metrics.append(s_mae)
    compare_metrics.append(s_rmse)

    # ---------plot results-------------

    # fig.tight_layout()

    if scenario == 'Haikou':
        # df_validate_mae.plot(ax=axes[0], kind='line', xlabel='Epoch', ylabel='MAE', sharex=True).legend(loc='upper right', bbox_to_anchor=(0.96, 0.92), ncol=2)
        # df_validate_rmse.plot(ax=axes[1], kind='line', xlabel='Epoch', ylabel='RMSE', sharex=True).legend(
        #     loc='upper right', bbox_to_anchor=(0.96, 0.92), ncol=2)
        df_validate_mae.plot(ax=axes[0,0], kind='line', xlabel='Epoch', xlim=[0,epochs], ylabel='MAE', sharex=True, legend=False)
        df_validate_rmse.plot(ax=axes[1,0], kind='line', xlabel='Epoch', xlim=[0,epochs], ylabel='RMSE', sharex=True, legend=False)
    else:
        df_train_rmse = df_train_rmse.loc[10:, :]  # cut off the first couple of epochs to avoid singular data
        df_validate_mae = df_validate_mae.loc[10:, :]
        df_validate_rmse = df_validate_rmse.loc[10:, :]

        # df_validate_mae.plot(ax=axes[0], kind='line', xlabel='Epoch', ylabel='MAE', sharex=True).legend(loc='upper right', bbox_to_anchor=(0.96, 0.92), ncol=2)
        # df_validate_rmse.plot(ax=axes[1], kind='line', xlabel='Epoch', ylabel='RMSE', sharex=True).legend(loc='upper right', bbox_to_anchor=(0.96, 0.92), ncol=2)
        df_validate_mae.plot(ax=axes[0,1], kind='line', xlabel='Epoch', xlim=[0,epochs], sharex=True, legend=False)
        df_validate_rmse.plot(ax=axes[1,1], kind='line', xlabel='Epoch', xlim=[0,epochs], sharex=True, legend=False)

    plt.subplots_adjust(left=0.04,  # plot start at 0.1 from left
                        bottom=0.11,
                        right=0.98,  # plot end at 0.9 from left
                        top=0.90,
                        wspace=0.08, #width space
                        hspace=0.05) #height space

axes[1][0].set_title(F'(a) {scenarios[0]}', y=-0.24)
axes[1][1].set_title(F'(b) {scenarios[1]}', y=-0.24)

#????????????
#?????????labels???????????????
lines=[]
labels=[]
#???????????????????????????labels
for ax in fig.axes:
    axLine, axLabel = ax.get_legend_handles_labels()
    lines.extend(axLine)
    labels.extend(axLabel)
    break
#?????????????????????????????????
fig.legend(lines, labels,loc='upper center',ncol=4, framealpha=False, fontsize=18)

fig_file = 'results/performance-compare-in-model-graphs.pdf'
fig.savefig(fig_file)
cmd = 'code ' + fig_file
os.system(cmd)

# plt.show()

df_compare_metrics = pd.concat(compare_metrics, axis=1)
print(df_compare_metrics)
print(df_compare_metrics.round(2).to_latex())
# print(df_compare_metrics.round(2).style.to_latex())
