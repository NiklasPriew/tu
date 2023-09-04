import matplotlib.pyplot as plt

# Data
data = {
    'CustomMLP': {
        'mice': {'f1_score': 0.6148148147009602, 'runtime': 19.875291109085083},
        'iris': {'f1_score': 0.9533333320622223, 'runtime': 5.183925151824951 },
        'breast': {'f1_score': 0.9753521119891887, 'runtime': 2.7196919918060303 }
    },
    'SCIKITMLP': {
        'mice': {'f1_score': 0.8435185185185186, 'runtime': 2.7221641540527344 },
        'iris': {'f1_score': 0.9133333333333333, 'runtime': 0.10587406158447266 },
        'breast': {'f1_score': 0.9647887323943662, 'runtime': 1.2038381099700928 }
    },
    'DecisionTree': {
        'mice': {'f1_score': 0.8027777777777778, 'runtime': 0.07672905921936035 },
        'iris': {'f1_score': 0.94, 'runtime': 0.007366180419921875 },
        'breast': {'f1_score': 0.8908450704225352, 'runtime': 0.016273021697998047 }
    }
}

# Colors and labels
colors = ['red', 'blue', 'green']
labels = ['CustomMLP', 'SCIKIT-MLP', 'DecisionTree']

# Plot
fig, axes = plt.subplots(nrows=len(data['CustomMLP']), ncols=2, figsize=(10, 10))

for i, dataset in enumerate(data['CustomMLP']):
    # F1-Score plot
    ax1 = axes[i, 0]
    for j, algorithm in enumerate(data):
        f1_score = data[algorithm][dataset]['f1_score']
        ax1.bar(labels[j], f1_score, color=colors[j], label=algorithm)
        ax1.set_title(f'Accuracy Comparison - {dataset}')
        ax1.set_ylim(0.55,1)
    ax1.legend()

    # Runtime plot
    ax2 = axes[i, 1]
    for j, algorithm in enumerate(data):
        runtime = data[algorithm][dataset]['runtime']
        ax2.bar(labels[j], runtime, color=colors[j], label=algorithm)
        ax2.set_title(f'Runtime Comparison - {dataset}')
        ax2.set_yscale("log")
        ax2.set_ylim([0.001, 25])
    ax2.legend()

plt.tight_layout()
plt.show()
