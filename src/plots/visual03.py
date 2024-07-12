import matplotlib.pyplot as plt
import numpy as np

# Data

# window_functions = ['None', 'Kaiser (beta=5)', 'Rectangular', 'Hamming', 'Hanning', 'Blackman']
# top5_accuracies = [
#     [
#         [0.04923414, 0.09846827, 0.14770241, 0.18161926, 0.2297593],
#         [0.05251641, 0.10284464, 0.15645514, 0.19474836, 0.23741794],
#         [0.04923414, 0.09846827, 0.14770241, 0.18161926, 0.2297593],
#         [0.05251641, 0.10284464, 0.15645514, 0.19256018, 0.23522976],
#         [0.04814005, 0.09956236, 0.15207878, 0.19474836, 0.22757111],
#         [0.0404814, 0.095186, 0.14989059, 0.19146608, 0.23522976]
#     ]
# ]


# # Data
# window_functions = ['None', 'Kaiser (beta=5)', 'Rectangular', 'Hamming', 'Hanning', 'Blackman']
# top5_accuracies = [
#     [
#         [0.3536313, 0.5094972, 0.575419, 0.61787707, 0.6452514],
#         [0.35075885, 0.5143339, 0.58853287, 0.62057334, 0.6458685],
#         [0.3322091, 0.48735243, 0.56155145, 0.59865093, 0.6256324],
#         [0.3524452, 0.52107924, 0.5851602, 0.6172007, 0.6509275],
#         [0.35581788, 0.5193929, 0.59527826, 0.63237774, 0.6593592],
#         [0.35750422, 0.5143339, 0.57841486, 0.61551434, 0.6509275]
#      ]
# ]

window_functions = ['None', 'Kaiser (beta=5)', 'Rectangular', 'Hamming', 'Hanning', 'Blackman']
# top5_accuracies = [
#     [
#         [0.0731205, 0.14418125, 0.20494336, 0.26261586, 0.31204945],
#         [0.0607621, 0.12873328, 0.18331617, 0.22657056, 0.27909374],
#         [0.0731205, 0.14418125, 0.20494336, 0.26261586, 0.31204945],
#         [0.05973224, 0.12564367, 0.18125644, 0.22657055, 0.27497426],
#         [0.06179197, 0.13285273, 0.19361484, 0.23892894, 0.2801236],
#         [0.06282184, 0.12461381, 0.17919672, 0.22142123, 0.27703398]
#      ]
# ]

top5_accuracies = [
    [
        [0.06945607, 0.12635984, 0.19330545, 0.23933056, 0.29288703],
        [0.08200837, 0.15146443, 0.19330545, 0.22426778, 0.2694561],
        [0.06945607, 0.12635984, 0.19330545, 0.23933056, 0.29288703],
        [0.08033473, 0.1497908, 0.18995817, 0.2284519, 0.26359832],
        [0.07866109, 0.15146443, 0.19497909, 0.22677825, 0.27196655],
        [0.07615063, 0.14058578, 0.17824268, 0.21841004, 0.26192468]
    ]
]

# Convert to percentage strings
def convert_to_percentage(accuracies):
    return [[f"{acc * 100:.2f}%" for acc in top] for top in accuracies[0]]

top5_accuracies_percent = convert_to_percentage(top5_accuracies)

# Plot the top-k accuracy for each window function
fig, ax = plt.subplots(figsize=(12, 8))

index = np.arange(5)
bar_width = 0.12

for i, wf in enumerate(window_functions):
    ax.bar(index + i * bar_width, [float(acc[:-1]) for acc in top5_accuracies_percent[i]], bar_width, label=wf)

ax.set_xlabel('Scenario33 Top-k Accuracy of Baseline ')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Top-5 Accuracy for Different Window Functions')
ax.set_xticks(index + (len(window_functions) - 1) * bar_width / 2)
ax.set_xticklabels(['Top-1', 'Top-2', 'Top-3', 'Top-4', 'Top-5'])
ax.legend()

plt.tight_layout()
plt.show()