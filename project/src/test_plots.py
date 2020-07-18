from src.research_utils import generate_plots

all_results = {'foo' : {'avg_fscore': [0.8779, 0.9027], 'avg_precision': [0.9605, 0.8447], 'avg_recall': [0.8084, 0.9691], 'avg_accuracy': 0.8917}}

generate_plots(all_results)

print("Done.")