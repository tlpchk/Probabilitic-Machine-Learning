from src.research_utils import test_cross, test_split, generate_plots, generate_plots_std
from src.data import get_factorized_dataset

from src.CategoricalNB import CategoricalNB
from src.pgm import PGM
import networkx as nx
from pomegranate import BayesianNetwork

def run_research(df):
    # Prepare data
    #df = get_factorized_dataset(path="../data").drop(['veil-type', 'stalk-root'],axis=1)
    target_column = "class"

    #Prepare models
    nb = CategoricalNB(num_epochs=20)

    df1 = df
    pgm1 = PGM(df1, num_epochs=20)
    m1 = BayesianNetwork.from_samples(df1, algorithm='chow-liu')
    pgm1.import_pomegranate_model(m1, df1.columns)

    df2 = df.drop(['odor'],axis=1)
    pgm2 = PGM(df2, num_epochs=20)
    m2 = BayesianNetwork.from_samples(df2, algorithm='chow-liu')
    pgm2.import_pomegranate_model(m2, df2.columns)

    df3 = df.get(
        ['odor',
         'class',
         'spore-print-color',
         'gill-color',
         'cap-color',
         'cap-shape',
         'cap-surface',
         'gill-size',
         'gill-spacing',
         'gill-attachment',
         'stalk-color-above-ring',
         'stalk-surface-above-ring',
         'stalk-surface-below-ring',
         'stalk-shape'])

    DAG=nx.DiGraph()

    edges = [
        ('odor','class'),
        ('spore-print-color','class'),
        ('gill-color','class'),
        ('cap-color','class'),
        ('cap-shape','cap-color'),
        ('cap-surface','cap-color'),
        ('gill-size','gill-color'),
        ('gill-spacing','gill-color'),
        ('gill-attachment','gill-color'),
        ('stalk-color-above-ring','class'),
        ('stalk-surface-below-ring','stalk-color-above-ring'),
        ('stalk-surface-above-ring','stalk-color-above-ring'),
        ('stalk-shape','stalk-color-above-ring')
    ]
    DAG.add_edges_from(edges)

    pgm3 = PGM(df3, graph=DAG, num_epochs=20)


    models = [
         (df,    nb, "Naive Bayes"),
         (df1, pgm1, "Bayesian Net #1"),
         (df2, pgm2, "Bayesian Net #2"),
         (df3, pgm3, "Bayesian Net #3"),
    ]

    #Results structure
    all_results = {}

    #Prepare file
    file = open("research_results.txt", "a")

    # Run experiments
    for (data,model, model_name) in models[0:1]:

        print(f"##### {model_name} #####\n")
        file.write(f"##### {model_name} #####\n")

        result = test_split(model=model, n_splits=5, df=data, class_column='class')
        print(result)

        file.write(str(result) + "\n")

        all_results[model_name] = result

    file.write("##### Partial results after test_split #####\n")
    file.write(str(all_results) + "\n")
    print("##### Partial results after test_split #####\n")
    print(str(all_results) + "\n")

    #Run experiments
    for (data,model, model_name) in models[1:]:

        print(f"##### {model_name} #####\n")
        file.write(f"##### {model_name} #####\n")

        result = test_cross(model=model, n_splits=5, df=data, class_column='class')
        print(result)

        file.write(str(result) + "\n")

        all_results[model_name] = result

    file.write("##### Combined results #####\n")
    file.write(str(all_results) + "\n")
    print("##### Combined results #####\n")
    print(str(all_results) + "\n")

    generate_plots(all_results)
    generate_plots_std(all_results)

    #Close file
    file.close()

    print("Done.")

if __name__ == "__main__":
    dataset = get_factorized_dataset(path="../data")
    run_research(dataset)