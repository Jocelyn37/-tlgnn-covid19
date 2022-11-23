
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

print("begin")
os.chdir("./output")

country_dict = {'EN': 'England', 'FR': "France", 'IT': 'Italy', "ES": "Spain"}

def plt_predictions_one_day(country, day, shifts):
    out = pd.read_csv("out_" + country + "_" + str(day) + "_" + str(shifts) + ".csv")
    plt.figure(figsize=(10, 5))
    out['l'].plot(linestyle='-', label="confirmed cases")
    out['o'].plot(linestyle=':', label="predictions")
    # plt.title("labels and predictions of cases for " + country + " on day " + str(day + shifts))
    plt.xlabel(country_dict[country] + " NUTS3 regions")
    plt.ylabel("Number of cases")
    ax = plt.gca()
    # ax.axes.xaxis.set_visible(False)
    # ax.axes.yaxis.set_visible(False)
    # ax.axes.xaxis.set_ticks([])
    ax.axes.xaxis.set_ticklabels([''])
    # plt.minorticks_on()
    plt.legend()
    plt.show()
    return


def plt_predictions_on_shift_and_region(country, start, end, shifts, region):
    outputs = list()
    labels = list()
    for day in range(start, end):
        out = pd.read_csv("out_" + country + "_" + str(day) + "_" + str(shifts) + ".csv")
        outputs.append(out['o'][region])
        labels.append(out['l'][region])

    plt.figure(figsize=(10, 5))
    plt.plot(labels, linestyle='-', label="labels")
    plt.plot(outputs, linestyle=':', label="predictions")
    plt.title("labels and predictions of cases for " + country + " on region " + outs['n'][region]
              + "\n from day " + str(start) + " to " + str(end) + " with shift = " + str(shifts))
    plt.xlabel(country + " from day " + str(start) + " to " + str(end))
    plt.ylabel("Number of cases")
    plt.legend()
    plt.show()
    return


def plt_predictions_on_shift(country, start, end, shift):
    outputs = list()
    labels = list()
    for day in range(start, end):
        out = pd.read_csv("out_" + country + "_" + str(day) + "_" + str(shift) + ".csv")
        predict_sum = 0
        label_sum = 0
        for predict in out['o']:
            predict_sum += predict
        for label in out['l']:
            label_sum += label
        outputs.append(predict_sum)
        labels.append(label_sum)

    plt.figure(figsize=(10, 5))
    plt.plot(labels, linestyle='-', label="labels")
    plt.plot(outputs, linestyle=':', label="predictions")
    plt.title("labels and predictions of cases for " + country
              + "\n from day " + str(start) + " to " + str(end) + " with shift = " + str(shift))
    plt.xlabel(country + " from day " + str(start) + " to " + str(end))
    plt.ylabel("Number of cases")
    plt.legend()
    plt.show()

    return

def plt_predictions_on_shifts(country, start, end, shifts):
    plt.figure(figsize=(12, 4.5))
    plt_labels = []
    linestyles = ['-', ':', '--', '-.', 'None']
    # linestyles = ['-', '-', '-', '-', 'None']
    linestyles_idx = 1
    period = range(start, end)
    for ahead in shifts:
        outputs = list()
        labels = list()
        for day in range(start, end):
            out = pd.read_csv("out_" + country + "_" + str(day-ahead) + "_" + str(ahead) + ".csv")
            predict_sum = 0
            label_sum = 0
            for predict in out['o']:
                predict_sum += predict
            for label in out['l']:
                label_sum += label
            outputs.append(predict_sum)
            labels.append(label_sum)
        # plt.plot(labels, linestyle='-', label="labels")
        plt_labels = labels
        plt.plot(period, outputs, linestyle=linestyles[linestyles_idx], label="predictions " + str(ahead) + " days ahead")
        linestyles_idx += 1
    plt.plot(period, plt_labels, linestyle='-', label="ground truth cases")
    plt.title("Predictions and ground truth of total confirmed cases for " + country_dict[country]
              + "\n from day " + str(start) + " to " + str(end))
    plt.xlabel("From day " + str(start) + " to day " + str(end) + " in " + country_dict[country])
    plt.ylabel("Number of cases")
    # plt.xlim(start, end)
    plt.legend()
    plt.show()

    return


def plt_statistic_cases(country):
    plt.figure(figsize=(10, 4))
    if country == "EN":
        cases = pd.read_csv("../data/England/england_labels.csv")
        col_start = 1
    elif country == "ES":
        cases = pd.read_csv("../data/Spain/spain_labels.csv")
        col_start = 1
    elif country == "IT":
        cases = pd.read_csv("../data/Italy/italy_labels.csv")
        col_start = 2
    elif country == "FR":
        cases = pd.read_csv("../data/France/france_labels.csv")
        col_start = 1
    else:
        return

    regions = list()
    regions_total_cases = list()
    regions_mean_cases = list()
    regions_std_cases = list()
    regions_median_cases = list()
    regions_max_diff_cases = list()
    # print(cases.shape[1])
    days_num = cases.shape[1] - col_start
    for index, row in cases.iterrows():
        regions.append(row.values[col_start-1])
        cases_values = row.values[col_start:-1]
        regions_total_cases.append(np.sum(cases_values))
        regions_mean_cases.append(np.mean(cases_values))
        regions_std_cases.append(np.std(cases_values))
        regions_median_cases.append(np.median(cases_values))
        regions_max_diff_cases.append(np.amax(cases_values) - np.amin(cases_values))
        # print(cases_values)
        # print(row.values)
    regions_num = len(regions)
    print(country_dict[country], "total cases:", np.sum(regions_total_cases))
    print(country_dict[country], "avg cases per day per region:", np.sum(regions_total_cases)/(regions_num*days_num))
    plt.plot(regions_mean_cases, marker='.', color='magenta', linestyle='-', linewidth=1, label="mean")
    plt.plot(regions_std_cases, marker='.', linestyle='-', linewidth=1, label="std")
    #plt.plot(regions_median_cases, marker='.', color='green', linestyle='-', linewidth=1, label="median")
    plt.plot(regions_max_diff_cases, 'c.-', linewidth=1, label="max diff")
    # 绘制点线图，指定线形，点形，颜色
    # plt.plot(regions_median_cases, marker='.', color='green', linestyle='-', linewidth=1, label="median")
    # plt.plot(regions_max_diff_cases, 'c.-', linewidth=1, label="max diff")
    plt.title("Case statistics for " + country_dict[country])
    plt.xlabel(country_dict[country] + " NUTS3 regions")
    plt.ylabel("Number of cases")
    plt.legend()
    plt.show()

    return


def plt_statistic_mobility(country):
    internal_mobilities = list()
    external_mobilities = list()
    for month in range(2, 7):
        for day in range(1, 32):
            if day < 10:
                file_name = country+"_2020-0"+str(month)+"-0"+str(day)+".csv"
            else:
                file_name = country + "_2020-0" + str(month) + "-" + str(day) + ".csv"
            file_path = "../data/"+country_dict[country]+"/graphs/"+file_name
            if not os.path.exists(file_path):
                continue
            mobility = pd.read_csv(file_path)
            for row in mobility.iterrows():
                # print(row[1][0], row[1][1], row[1][2])
                if row[1][0] == row[1][1]:
                    internal_mobilities.append(row[1][2])
                else:
                    external_mobilities.append(row[1][2])
    print(country_dict[country]+" avg internal_mobilities ", np.mean(internal_mobilities))
    print(country_dict[country]+" avg external_mobilities ", np.mean(external_mobilities))
    return

outs = pd.read_csv("out_EN_15_0.csv")
shift = 0
outputs = list()
labels = list()

# plt_predictions_one_day("EN", 16, 2)
# plt_predictions_on_shift_and_region("EN", 16, 58, 0, 1)
# plt_predictions_on_shift("EN", 16, 58, 0)
# plt_predictions_on_shift("EN", 16, 45, 13)
# plt_predictions_on_shifts("EN", 25, 45, [3, 7, 10])


# plt_predictions_one_day("EN", 16, 12)
# plt_predictions_one_day("ES", 16, 12)
# plt_predictions_one_day("IT", 16, 12)
# plt_predictions_one_day("FR", 16, 12)

# predict 08-05-2020 10 days ahead.
# plt_predictions_one_day("EN", 43, 13)
# plt_predictions_one_day("ES", 44, 13)
# plt_predictions_one_day("FR", 46, 13)
# plt_predictions_one_day("IT", 61, 13)


# plt_predictions_on_shifts("IT", 28, 61, [3, 12, 13])
# plt_predictions_on_shifts("FR", 28, 61, [1, 7, 13])
# plt_predictions_on_shifts("ES", 28, 61, [1, 7, 13])
# plt_predictions_on_shifts("EN", 28, 61, [1, 7, 13])

# plt_statistic_cases("EN")
# plt_statistic_cases("ES")
# plt_statistic_cases("IT")
# plt_statistic_cases("FR")


plt_statistic_mobility("EN")
plt_statistic_mobility("ES")
plt_statistic_mobility("IT")
plt_statistic_mobility("FR")

print("end")



"""
for key in TRAIN_LOSS.keys():
    plt.figure(figsize=(18, 8))
    plt.plot(TRAIN_LOSS[key], linestyle='--', label=key + "_train")
    plt.plot(TEST_LOSS[key], linestyle=':', label=key + "_test")
    plt.legend()
    plt.title("Train and Test Loss for " + key)

    plt.xlabel("T")
    plt.ylabel("Error")
    plt.ylim([-.01, 1])

plt.show()
"""



