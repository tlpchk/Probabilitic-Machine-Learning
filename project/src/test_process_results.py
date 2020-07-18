from src.research_utils import process_results

results = {'0': {'precision': 0.9547511312217195, 'recall': 0.8084291187739464, 'f1-score': 0.8755186721991701, 'support': 783}, '1': {'precision': 0.8440748440748441, 'recall': 0.9643705463182898, 'f1-score': 0.9002217294900222, 'support': 842}, 'accuracy': 0.8892307692307693, 'macro avg': {'precision': 0.8994129876482818, 'recall': 0.8863998325461181, 'f1-score': 0.8878702008445962, 'support': 1625}, 'weighted avg': {'precision': 0.8974037873585387, 'recall': 0.8892307692307693, 'f1-score': 0.8883186563461839, 'support': 1625}}

out = process_results(results)

print(out + "\n")
print("Done.")