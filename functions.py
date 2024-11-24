def find_categorical_variables(data, threshold):
    """
    Identifie les variables avec un nombre de valeurs uniques inférieur à un seuil.
    Retourne un DataFrame avec les variables et leurs comptes de valeurs uniques.
    """
    categorical_result = []
    for col in data.columns:
        unique_count = data[col].nunique()
        if unique_count < threshold:
            categorical_result.append([col, unique_count])

    result_df = pd.DataFrame(categorical_result, columns=["Variable", "Unique_Count"])

    print("-"*40)
    print(f"Nombre total de variables ayant moins de {threshold} modalités : {len(result_df)} \n")
    print("-"*40)
    print(result_df.sort_values(by="Unique_Count") if not result_df.empty
           else "Aucune variable trouvée avec un nombre de modalités unique inférieur au seuil donné.")
    print("-"*40)

    return result_df


def find_uniq_mod_variables(data, qualitative_vars, threshold=0.9):
    """
    Identifie les variables qualitatives ayant une modalité dominante dépassant un certain seuil.
    Retourne un DataFrame avec les variables, la modalité dominante, son pourcentage, et le nombre total de modalités.
    """
    unique_mod_result = []

    for var in qualitative_vars:
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences
        if value_counts.iloc[0] >= threshold:
            unique_mod_result.append({
                "Variable": var,
                "Nb_mod": data[var].nunique(),  # Nombre total de modalités
                "Mod_dominante": value_counts.index[0],
                "Frequence": value_counts.iloc[0]
            })

    result_df = pd.DataFrame(unique_mod_result)

    print("-"*40)
    print(f"Nombre de variables ayant une modalité dominante à plus de {threshold * 100}% : {len(result_df)}")
    print("-"*40)
    print(result_df if not result_df.empty else "Aucune variable trouvée avec une modalité dominante.")
    print("-"*40)

    return result_df


import matplotlib.pyplot as plt
import math

def plot_cat_vars_distributions(data, vars_list, cols=2):
    """
    Génère des graphiques montrant la distribution des modalités pour chaque variable dans une grille.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols)  # Calcul du nombre de lignes nécessaires
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile
    
    for i, var in enumerate(vars_list):
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences
        
        # Gestion de l'index pour éviter les erreurs MultiIndex
        index_values = value_counts.index.to_flat_index()  # Aplatir si MultiIndex
        index_values = [str(x) for x in index_values]      # Convertir en chaîne chaque valeur
        
        # Création du graphique
        axes[i].bar(index_values, value_counts.values, color='skyblue')
        axes[i].set_ylabel('Proportion')
        axes[i].set_title(f'{var}')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Supprimer les sous-graphiques inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Distribution des modalités", fontsize=16, y=1.02) 
    
    plt.tight_layout()
    plt.show()



import matplotlib.pyplot as plt
import pandas as pd
import math

import matplotlib.pyplot as plt
import pandas as pd
import math

def tx_rsq_par_var(df, categ_vars, date, target, cols=2):
    """
    Génère une grille de graphiques montrant les taux d'événement moyens par modalité au fil du temps,
    pour une liste de variables catégorielles, en fonction des valeurs cibles fournies.
    """
    df = df.copy()

    # Vérification des colonnes
    missing_cols = [col for col in [date] + categ_vars if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Les colonnes suivantes sont manquantes dans le DataFrame : {missing_cols}")

    # Nettoyer les valeurs manquantes dans les colonnes nécessaires
    df = df.dropna(subset=[date] + categ_vars)

    num_vars = len(categ_vars)
    rows = math.ceil(num_vars / cols)  # Calcul du nombre de lignes nécessaires

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=True, sharey=True)
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, categ_var in enumerate(categ_vars):
        # Calcul des moyennes par date et catégorie
        df_times_series = (df.groupby([date, categ_var])[target].mean() * 100).reset_index()
        df_pivot = df_times_series.pivot(index=date, columns=categ_var, values=target)

        # Création du graphique
        ax = axes[i]
        for category in df_pivot.columns:
            ax.plot(df_pivot.index, df_pivot[category], label=category)
        ax.set_title(f"{categ_var}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Tx de défaut (%)")
        ax.legend(title="Modalités", fontsize='small', loc='upper left')

    # Supprimer les axes inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Taux de défaut par variable catégorielle", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()




import matplotlib.pyplot as plt
import seaborn as sns
import math

def compare_distributions_grid(X_train, X_test, var_list, cols=2):
    """
    Compare les distributions des variables continues dans Train et Test et les affiche sous forme de grille.
    """
    num_vars = len(var_list)
    rows = math.ceil(num_vars / cols)  # Calcul du nombre de lignes nécessaires
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(var_list):
        # Graphique pour chaque variable
        sns.kdeplot(X_train[var], label='Train', shade=True, ax=axes[i])
        sns.kdeplot(X_test[var], label='Test', shade=True, ax=axes[i])
        axes[i].set_title(f"{var}")
        axes[i].legend()

    # Supprimer les sous-graphiques inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Comparaison des distributions dans l'echantillon Train et Test", fontsize=16, y=1.02) 

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

def compare_distributions_summary(X_train, X_test, var_list):
    """
    Compare les distributions des variables continues dans Train et Test et retourne un tableau récapitulatif.
    Affiche les statistiques descriptives et les p-values des tests de Kolmogorov-Smirnov.
    """
    results = []

    for var in var_list:
        # Statistiques descriptives
        train_stats = X_train[var].describe()
        test_stats = X_test[var].describe()

        # Test de Kolmogorov-Smirnov
        ks_stat, ks_p_value = stats.ks_2samp(X_train[var], X_test[var])
        
        # Ajout des résultats dans la liste
        results.append({
            "Variable": var,
            "Train_Mean": train_stats["mean"],
            "Test_Mean": test_stats["mean"],
            "Train_Std": train_stats["std"],
            "Test_Std": test_stats["std"],
            "KS_Statistic": ks_stat,
            "KS_p_value": ks_p_value,
            "Similar_Distribution": "Yes" if ks_p_value > 0.05 else "No"
        })

    # Conversion des résultats en DataFrame
    result_df = pd.DataFrame(results)
    return result_df



import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def plot_modalities_over_time(X_train, date_col, categorical_vars, exclude_vars=None, cols=2):
    """
    Affiche l'évolution du nombre de modalités uniques par variable catégorielle au fil du temps.
    """
    if exclude_vars is None:
        exclude_vars = []

    # Filtrer les variables catégorielles
    cat_vars = [col for col in categorical_vars if col not in exclude_vars]

    # Créer un DataFrame pour stocker les informations pour la visualisation
    modalities_over_time = []

    # Itérer sur les dates d'observation
    for date in X_train[date_col].unique():
        filtered_data = X_train[X_train[date_col] == date]
        for col in cat_vars:
            modalities = filtered_data[col].unique()
            modalities_count = len(modalities)
            modalities_over_time.append({
                'date': date,
                'variable': col,
                'modalities_count': modalities_count
            })

    # Convertir la liste en DataFrame
    modalities_df = pd.DataFrame(modalities_over_time)

    # Déterminer le nombre de graphiques
    num_vars = len(cat_vars)
    rows = math.ceil(num_vars / cols)

    # Créer une grille de graphiques
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, var in enumerate(cat_vars):
        ax = axes[i]
        sns.lineplot(data=modalities_df[modalities_df['variable'] == var],
                     x='date', y='modalities_count', marker='o', ax=ax)
        ax.set_title(f"Évolution de {var}")
        ax.set_xlabel("Date d'observation")
        ax.set_ylabel("Nombre de modalités uniques")
        ax.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Evolution des modalités dans le temps", fontsize=16, y=1.02) 

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_boxplots(data, vars_list, cols=2):
    """
    Affiche des boxplots pour chaque variable continue dans une grille.

    Args:
    - data : DataFrame contenant les données.
    - vars_list : Liste des variables continues à visualiser.
    - cols : Nombre de colonnes dans la grille (par défaut : 2).
    - general_title : Titre général pour la figure (par défaut : None).
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols)  # Calcul du nombre de lignes nécessaires
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(vars_list):
        sns.boxplot(data=data, y=var, ax=axes[i])  # Création du boxplot
        axes[i].set_title(f"Boxplot de {var}")
        axes[i].set_xlabel("")  # Pas besoin d'étiquette pour x
        axes[i].set_ylabel("Valeurs")

    # Supprimer les sous-graphiques inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajouter un titre général
    fig.suptitle("Distribution des variables continues", fontsize=16, y=1.02)  # Positionner le titre légèrement au-dessus

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math

def plot_boxplots_by_target(data, vars_list, target, cols=2):
    """
    Génère des boxplots pour chaque variable continue en fonction des valeurs cibles fournies.
    """

    # Ajouter les valeurs cibles comme colonne temporaire au DataFrame
    data = data.copy()
    # Début de la génération des graphiques
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols)  # Calcul du nombre de lignes nécessaires

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(vars_list):
        # Création du boxplot
        sns.boxplot(data=data, x=target, y=var, ax=axes[i], palette='Set3')
        axes[i].set_title(f"{var}")
        axes[i].set_xlabel("Valeur cible")
        axes[i].set_ylabel(var)
        

    # Supprimer les sous-graphiques inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Ajouter un titre général
    fig.suptitle("Boxplots des variables continues par valeur cible", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

from scipy.stats import chi2_contingency

def cramer_V(cat_var1,cat_var2):
    """
    Calcule le coefficient de Cramer's V pour une paire de variables catégorielles.
    """
    crosstab = np.array(pd.crosstab(cat_var1,cat_var2,rownames=None,colnames=None)) #tableau de contingence
    stat = chi2_contingency(crosstab)[0] #stat de test de khi-2
    obs=np.sum(crosstab) 
    mini = min(crosstab.shape)-1 #min entre les colonnes et ligne du tableau croisé ==> ddl
    return (np.sqrt(stat/(obs*mini)))

def table_cramerV(df):
    """
    Calcule le coefficient de Cramer's V pour chaque paire de variables catégorielles.
    """
    rows=[]
    for var1 in df :
        col=[]
        for var2 in df :
            cramers = cramer_V(df[var1],df[var2])
            col.append(round(cramers,2))
        rows.append(col)
    cramers_results = np.array(rows)
    result=pd.DataFrame(cramers_results,columns=df.columns,index=df.columns)

def compute_cramers_v(df, categorical_vars, target):
    """
    Calcule le coefficient de Cramer's V pour chaque combinaison paire de variables catégorielles dans la liste fournie.
    """
    results = []
    for var1 in categorical_vars :  # Unpack index and column name
        if var1 == target:
            continue  # Skip the calculation if the variable is the target itself
        cv = cramer_V(df[var1], df[target])  # Correctly pass the variable names
        results.append([var1, cv])  # Append the variable name, not the tuple

    # Create a DataFrame to hold the results
    result_df = pd.DataFrame(results, columns=['Columns', "Cramer_V"])
    return result_df

def stats_liaisons_var_quali(df,categorical_columns):
    """
    Calcule le test du chi-deux et le coefficient de cramer_v pour chaque paire de variables qualitatives
    """
    cramer_v_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    p_value_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)
    #tschuprow_t_df = pd.DataFrame(index=categorical_columns,columns=categorical_columns)

    #test de chi-deux pour chaque paire de variables quali
    for i, column1 in enumerate(categorical_columns):
        for j, column2 in enumerate(categorical_columns):
            if column1 != column2:
                contingency_table = pd.crosstab(df[column1], df[column2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                cramer_v = np.sqrt(chi2 / (df.shape[0] * (min(contingency_table.shape)-1) ))
                #tschuprow_t = np.sqrt(chi2 / (df.shape[0] * np.sqrt((contingency_table.shape[0] - 1) * (contingency_table.shape[1] - 1))))
                cramer_v_df.loc[column1,column2] =cramer_v
                #tschuprow_t_df.loc[column1,column2] =tschuprow_t
                p_value_df.loc[column1,column2] = p
                
    return (p_value_df, cramer_v_df)






#### CHECK IF USED
import pandas as pd

def filter_by_cv(df, variables, threshold=0.1):
    """
    Filtre les variables qui ont un Coefficient de Variation (CV) supérieur à un seuil donné.
    """
    cv_values = {}
    high_cv_vars = []
    
    for var in variables:
        if var in df.columns:
            mean = df[var].mean()
            std = df[var].std()
            
            # Calcul du Coefficient de Variation (CV)
            if mean != 0:  # Éviter les divisions par zéro
                cv = std / mean
                cv_values[var] = cv
                # Vérification si le CV est supérieur au seuil
                if cv > threshold:
                    high_cv_vars.append(var)
            else:
                cv_values[var] = float('inf')  # CV infini si la moyenne est 0
    print("----- Resultats -----")
    print(f"Nombre de variables avec un CV élevé: {len(high_cv_vars)}")
    print("\n Variables :")
    if high_cv_vars:
        for var in high_cv_vars:
            print(f"  - {var} (CV: {cv_values[var]:.4f})")
    else:
        print("  None found.")
    
    return high_cv_vars, cv_values,


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def gini_index(data):
    """
    Calcule l'indice de Gini pour une série de données.
    """
    sorted_data = np.sort(data)
    n = len(data)
    cumulative = np.cumsum(sorted_data) / np.sum(sorted_data)
    gini = 1 - (2 / n) * np.sum(cumulative) + 1 / n
    return gini

def summarize_continuous_vars(df, variables):
    """
    Résume les informations de distribution pour un ensemble de variables continues.

    """
    summary = []

    for var in variables:
        if var in df.columns:
            data = df[var].dropna()  # Exclure les valeurs manquantes
            if len(data) > 1:  # Vérifier qu'il y a assez de données pour calculer les stats
                mean = data.mean()
                std = data.std()
                cv = std / mean if mean != 0 else np.nan
                iqr = data.quantile(0.75) - data.quantile(0.25)
                skewness = skew(data)
                kurt = kurtosis(data, fisher=True)
                gini = gini_index(data)

                summary.append({
                    "Variable": var,
                    "Mean": mean,
                    "StdDev": std,
                    "CV": cv,
                    "IQR": iqr,
                    "Skewness": skewness,
                    "Kurtosis": kurt,
                    "Gini": gini
                })

    return pd.DataFrame(summary)

def sum_iqr(df, variables):
    """
    Résume les informations de distribution pour un ensemble de variables continues.

    """
    summary = []

    for var in variables:
        if var in df.columns:
            data = df[var].dropna()  # Exclure les valeurs manquantes
            if len(data) > 1:  # Vérifier qu'il y a assez de données pour calculer les stats
                iqr = data.quantile(0.75) - data.quantile(0.25)
                if iqr == 0 :
                    summary.append({
                        "Variable": var,
                        "IQR": iqr
                    })

    return pd.DataFrame(summary)


def summarize_distribution_table(df):
    """
    Résume les informations de distribution d'un tableau basé sur des seuils pour CV, skewness, kurtosis et Gini.
    """
    summary = {
        #"High Dispersion (CV > 1)": df[df['CV'].abs() > 1]['Variable'].tolist(),
        "Low Dispersion (CV < 0.1)": df[df['CV'].abs() < 0.1]['Variable'].tolist(),
        "Highly Skewed (Skewness > 2 or < -2)": df[(df['Skewness'] > 2) | (df['Skewness'] > -2)]['Variable'].tolist(),
        "Highly Kurtotic (Kurtosis > 3)": df[df['Kurtosis'] > 3]['Variable'].tolist(),
        "Highly Concentrated (Gini > 0.9)": df[df['Gini'] > 0.9]['Variable'].tolist(),
    }
    
    return summary
