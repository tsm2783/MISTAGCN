import glob
import pandas as pd
import os
import matplotlib.pyplot as plt

from share import epochs


def merge(list_df, shared_column_name):
    '''merge a list of dataframe

    Args:
        list_df (list[dataframe]): a list of dataframe
    '''
    assert len(list_df) > 0, 'the list should not be empty'
    df = list_df[0]
    for i in range(1, len(list_df)):
        df = pd.merge(df, list_df[i], left_on=shared_column_name, right_on=shared_column_name)

    return df


def main():
    scenarios = ['Haikou', 'YellowTrip']

    plt.figure()
    plt.rcParams.update({'font.size': 20, 'axes.labelsize': 20, 'xtick.labelsize': 18, 'ytick.labelsize': 18})

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(22, 14))

    list_df_recent = []
    list_df_daily = []
    list_df_weekly = []
    list_df_predict_len = []

    for i, (scenario) in enumerate(scenarios):
        scenario_dir = './results-in-model/effect-parameters/' + scenario + '/'

        file_name = scenario_dir + 'recent-input-change-compare-results.xlsx'
        df = pd.read_excel(file_name, sheet_name='test_records', engine='openpyxl')
        list_df_recent.append(df)

        file_name = scenario_dir + 'daily-input-change-compare-results.xlsx'
        df = pd.read_excel(file_name, sheet_name='test_records', engine='openpyxl')
        list_df_daily.append(df)

        file_name = scenario_dir + 'weekly-input-change-compare-results.xlsx'
        df = pd.read_excel(file_name, sheet_name='test_records', engine='openpyxl')
        list_df_weekly.append(df)

        file_name = scenario_dir + 'prediction-length-change-compare-results.xlsx'
        df = pd.read_excel(file_name, sheet_name='test_records', engine='openpyxl')
        list_df_predict_len.append(df)

    df_recent = merge(list_df_recent, shared_column_name='Tr')
    columns = scenarios.copy()
    columns.append('Tr')
    df_recent = df_recent[columns]

    df_daily = merge(list_df_daily, shared_column_name='Td')
    columns = scenarios.copy()
    columns.append('Td')
    df_daily = df_daily[columns]

    df_weekly = merge(list_df_weekly, shared_column_name='Tw')
    columns = scenarios.copy()
    columns.append('Tw')
    df_weekly = df_weekly[columns]

    df_predict_len = merge(list_df_predict_len, shared_column_name='Tp')
    columns = scenarios.copy()
    columns.append('Tp')
    df_predict_len = df_predict_len[columns]

    # df_recent.plot(ax=axes[0, 0], kind='line', xlabel='Tr', ylabel='RMSE', sharex=False, sharey=False, legend=False)
    x_ticks = list(range(df_recent['Tr'].min(), df_recent['Tr'].max()+1))
    df_recent.plot(ax=axes[0, 0], kind='line', x='Tr', y=scenarios, ylabel='RMSE', style='o-',
                   sharex=False, sharey=False, legend=False, xticks=x_ticks)

    x_ticks = list(range(df_daily['Td'].min(), df_daily['Td'].max()+1))
    df_daily.plot(ax=axes[0, 1], kind='line', x='Td', y=scenarios, 
                  style='o-', sharex=False, sharey=False, legend=False, xticks=x_ticks)

    x_ticks = list(range(df_weekly['Tw'].min(), df_weekly['Tw'].max()+1))
    df_weekly.plot(ax=axes[1, 0], kind='line', x='Tw', y=scenarios, ylabel='RMSE',
                   style='o-', sharex=False, sharey=False, legend=False, xticks=x_ticks)

    x_ticks = list(range(df_predict_len['Tp'].min(), df_predict_len['Tp'].max()+1))
    df_predict_len.plot(ax=axes[1, 1], kind='line', x='Tp', y=scenarios, 
                        style='o-', sharex=False, sharey=False, legend=False, xticks=x_ticks)

    plt.subplots_adjust(left=0.04,  # plot start at 0.1 from left
                        bottom=0.11,
                        right=0.98,  # plot end at 0.9 from left
                        top=0.93,
                        wspace=0.08,  # width space
                        hspace=0.3)  # height space

    axes[0][0].set_title('(a)', y=-0.24)
    axes[0][1].set_title('(b)', y=-0.24)
    axes[1][0].set_title('(c)', y=-0.24)
    axes[1][1].set_title('(d)', y=-0.24)

    # 合并图例
    # 初始化labels和线型数组
    lines = []
    labels = []
    # 通过循环获取线型和labels
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
        break
    # 设置图例和调整图例位置
    fig.legend(lines, labels, loc='upper center', ncol=4, framealpha=False, fontsize=18)

    fig_file = 'results/performance-compare-in-model-parameters.pdf'
    fig.savefig(fig_file)
    cmd = 'code ' + fig_file
    os.system(cmd)


main()
