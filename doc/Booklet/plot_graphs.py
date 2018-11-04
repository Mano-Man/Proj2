import numpy as np
import matplotlib.pyplot as plt

metrics = ['Direct', '0-1 Loss', 'Neighbor']
x_axis = range(3)
genetic_results = {'Direct': [52.3, 52.64, 51.6], '0-1 Loss': [52.3, 52.3, 50.56], 'Neighbor': [66.49, 55.01, 52.53]}
gradient_results = {'Direct': [54.68, 23, 11.99], '0-1 Loss': [51.72, 14.53, 6.4], 'Neighbor': [57.69, 21.88, 10.23]}
rnn_results = {'Direct': [93.35, 47.1, 29.27], '0-1 Loss': [88.4, 2.9, 0], 'Neighbor': [90.35, 29.59, 14.24]}
rnn_simple_results = {'Direct': [92.52, 47.87, 30.96], '0-1 Loss': [79.01, 0.67, 0], 'Neighbor': [0, 0, 0]}
deep_similarity = {'Direct': [99.98,28.3,13.2 ], '0-1 Loss': [99.98,24.12,12.9], 'Neighbor': [100,23.8,14.5]}

results = [genetic_results, gradient_results, rnn_simple_results, rnn_results,deep_similarity]
colors = ['C0', 'C1', 'C2', 'C3','C4']
labels = ['Genetic Algorithm', 'Gradient Algorithm', 'Simple Softmax RNN', 'RNN No Repeats','Deep Similarity']

for metric in metrics:
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for i, result in enumerate(results):
        if metric == 'Neighbor' and result == rnn_simple_results:
            continue

        ax.scatter(x_axis, result[metric], s=30, c=colors[i], marker="o", label=labels[i])
        plt.plot(x_axis, result[metric], colors[i])
        # for i, j in zip(x_axis, result[metric]):
        #     ax.annotate(str(j), xy=(i, j))

    plt.xlabel('Tiles Per Dimension')
    #plt.xlim([0,2])
    plt.ylabel('Accuracy[%]')
    plt.xticks(x_axis,('T=2','T=4','T=5'))
    plt.title(f'Metric: {metric}')
    plt.grid()
    plt.legend()
    plt.show()



