import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
# import tensorflow as tf
# from sklearn.metrics import confusion_matrix
# from tensorflow.keras.callbacks import TensorBoard
# from sklearn.metrics import roc_curve, auc


metrics = ['Accuracy', 'Recall', 'Precision', 'F1 Score']
# Bar width
bar_width = 0.11
# Set up positions for bars
index = np.arange(len(metrics))
# Plotting
# plt.figure(figsize=(12, 6))
plt.figure(figsize=(8, 4))
# plt.grid = True
plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

# WDNN = [84.81, 85.74, 81.40, 85.37]
AWDNN = [89.87, 89.74, 91.67, 89.74]


def plot_bar_chat_static():
    # Labels

    # Experiment labels
    # experiments = ['WIDENNET', 'DR-GCN', 'GCN', 'Mythril']

    Smartcheck = [54.65, 16.34, 45.71, 24.07]
    Oyente = [65.07, 63.02, 46.56, 53.55]
    Mythril = [64.27, 75.51, 42.86, 54.68]
    Securify = [72.89, 73.06, 68.40, 70.41]
    Slither = [74.02, 73.50, 74.44, 73.97]


    # my_values = [self.accuracy * 100, self.recall * 100, self.precision * 100, self.f1 * 100]

    # bar1 = plt.bar(index, WDNNA, bar_width, label='AWDNNA')
    bar1 = plt.bar(index, AWDNN, bar_width, label='AWDNN')
    bar2 = plt.bar(index + bar_width, Slither, bar_width, label='Slither')
    bar3 = plt.bar(index + 2 * bar_width, Securify, bar_width, label='Securify')
    bar4 = plt.bar(index + 3 * bar_width, Oyente, bar_width, label='Oyente')
    bar5 = plt.bar(index + 4 * bar_width, Mythril, bar_width, label='Mythril')
    bar6 = plt.bar(index + 5 * bar_width, Smartcheck, bar_width, label='Smartcheck')

    # # bar1 = plt.bar(index, WDNNA, bar_width, label='AWDNNA')
    # bar1 = plt.bar(index + bar_width, AWDNN, bar_width, label='AWDNN')
    # bar2 = plt.bar(index + 2 * bar_width, Slither, bar_width, label='Slither')
    # bar3 = plt.bar(index + 3 * bar_width, Securify, bar_width, label='Securify')
    # bar4 = plt.bar(index + 4 * bar_width, Oyente, bar_width, label='Oyente')
    # bar5 = plt.bar(index + 5 * bar_width, Mythril, bar_width, label='Mythril')
    # bar6 = plt.bar(index + 6 * bar_width, Smartcheck, bar_width, label='Smartcheck')

    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # plt.title('Comparison of Metrics between Experiments')
    plt.xticks(index + bar_width / 6, metrics)
    plt.legend(loc='upper right', prop={'size': 5})
    # plt.legend()

    # Show the plot
    plt.show()


def plot_bar_chat_deep():
    # Labels

    # Experiment labels
    # experiments = ['WIDENNET', 'DR-GCN', 'GCN', 'Mythril']

    Vanilla_RNN = [65.90, 72.89, 67.39, 70.03]
    ReChecker = [70.95, 72.92, 70.15, 71.51]
    GCN = [73.21, 73.18, 74.47, 73.82]
    TMP = [76.45, 75.30, 76.04, 75.67]
    AME = [81.06, 78.45, 79.62, 79.03]
    SMS = [83.85, 77.48, 79.46, 78.46]
    DMT = [89.42, 81.06, 83.62, 82.32]

    plt.figure(figsize=(8, 4))
    # plt.grid = True
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='gray')

    # bar1 = plt.bar(index, WDNNA, bar_width, label='AWDNNA')
    bar1 = plt.bar(index, AWDNN, bar_width, label='AWDNN')
    bar2 = plt.bar(index + bar_width, DMT, bar_width, label='DMT')
    bar3 = plt.bar(index + 2 * bar_width, SMS, bar_width, label='SMS')
    bar4 = plt.bar(index + 3 * bar_width, GCN, bar_width, label='GCN')
    bar5 = plt.bar(index + 4 * bar_width, Vanilla_RNN, bar_width, label='Vanilla_RNN')
    bar6 = plt.bar(index + 5 * bar_width, ReChecker, bar_width, label='Rechecker')

    # # bar1 = plt.bar(index, WDNNA, bar_width, label='AWDNNA')
    # bar1 = plt.bar(index + bar_width, AWDNN, bar_width, label='AWDNN')
    # bar2 = plt.bar(index + 2 * bar_width, DMT, bar_width, label='DMT')
    # bar3 = plt.bar(index + 3 * bar_width, SMS, bar_width, label='SMS')
    # bar4 = plt.bar(index + 4 * bar_width, GCN, bar_width, label='GCN')
    # bar5 = plt.bar(index + 5 * bar_width, Vanilla_RNN, bar_width, label='Vanilla_RNN')
    # bar6 = plt.bar(index + 6 * bar_width, ReChecker, bar_width, label='Rechecker')




    # Add labels and title
    plt.xlabel('Metrics')
    plt.ylabel('Values')
    # plt.title('Comparison of Metrics between Experiments')
    plt.xticks(index + bar_width / 6, metrics)
    plt.legend(loc='upper right', prop={'size': 5})
    # plt.legend()

    # Show the plot
    plt.show()


plot_bar_chat_static()
plot_bar_chat_deep()
