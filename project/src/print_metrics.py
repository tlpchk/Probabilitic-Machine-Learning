all_results = {'Naive Bayes': {'avg_fscore': [0.8654, 0.8943], 'avg_precision': [0.9564, 0.8322], 'avg_recall': [0.7903, 0.9665], 'avg_accuracy': 0.8816, 'std_fscore': [0.0094, 0.0063], 'std_precision': [0.0046, 0.0092], 'std_recall': [0.0133, 0.0033], 'std_accuracy': 0.0076}, 'Bayesian Net #1': {'avg_fscore': [0.9294, 0.939], 'avg_precision': [0.9545, 0.9208], 'avg_recall': [0.9078, 0.9596], 'avg_accuracy': 0.9346, 'std_fscore': [0.0318, 0.0236], 'std_precision': [0.0029, 0.0489], 'std_recall': [0.0614, 0.005], 'std_accuracy': 0.0272}, 'Bayesian Net #2': {'avg_fscore': [0.7648, 0.7442], 'avg_precision': [0.8198, 0.8171], 'avg_recall': [0.7741, 0.7744], 'avg_accuracy': 0.7742, 'std_fscore': [0.1253, 0.2212], 'std_precision': [0.1483, 0.1103], 'std_recall': [0.2022, 0.2913], 'std_accuracy': 0.1303}, 'Bayesian Net #3': {'avg_fscore': [0.8886, 0.8727], 'avg_precision': [0.8806, 0.9136], 'avg_recall': [0.9112, 0.8578], 'avg_accuracy': 0.8835, 'std_fscore': [0.0807, 0.1251], 'std_precision': [0.1261, 0.055], 'std_recall': [0.0707, 0.1855], 'std_accuracy': 0.097}}

for _, (model_name, v) in enumerate(all_results.items()):
    print(model_name + "\n")
    print("\\begin{center}\n")
    print("\\begin{tabular}{|c|c|c|}\n")
    print("\hline\n")
    print("Metryka & Wartość średnia & Odchylenie standardowe \\\\ \n")
    print("\hline\n")
    metric = 'accuracy'
    print(f"{metric} & {v['avg_' + metric]} & {v['std_' + metric]} \\\\ \n")
    for metric in ['precision', 'recall', 'fscore']:
        print(f"{metric} & {v['avg_' + metric][0]} & {v['std_' + metric][0]} \\\\ \n")
    print("\hline\n")
    print("\end{tabular}\n")
    print("\end{center}\n")
    print("\n")
    # fscore.append(v['std_fscore'][0])
    # precision.append(v['std_precision'][0])
    # recall.append(v['std_recall'][0])
    # accuracy.append(v['std_accuracy'])