
def find_uniq_mod_variables(data, qualitative_vars,threshold=0.9):
    """
    Parcourt toutes les variables qualitatives et identifie celles qui prennent principalement une seule modalité.
    """
    unique_mod_vars = []  # Liste des variables
    details = {}  # Détails des répartitions des modalités pour les variables exclues

    # Filtrer les variables qualitatives
    # qualitative_vars = data.select_dtypes(include=['object', 'category']).columns

    # Parcourir les variables qualitatives
    for var in qualitative_vars:
        value_counts = data[var].value_counts(normalize=True)
        if value_counts.iloc[0] >= threshold:
            unique_mod_vars.append(var)
            details[var] = value_counts.to_dict()

    return unique_mod_vars, details


def find_categorical_variables(data, threshold):
    """
    Identifie les variables ayant un nombre de valeurs uniques inférieur à un seuil.
    """
    low_unique_vars = []  # Liste des variables avec peu de valeurs uniques
    unique_counts = {}  # Détail des comptes de valeurs uniques

    # Parcourir les colonnes du DataFrame
    for col in data.columns:
        unique_count = data[col].nunique()  # Compter les valeurs uniques
        if unique_count < threshold :  # Vérifier si le nombre est inférieur au seuil
            low_unique_vars.append(col)
            unique_counts[col] = unique_count
    
    print("----- Resultats -----")
    print(f"Nombre de variables avec peu de modalités: {len(low_unique_vars)}")
    print("\n Variables :")
    if low_unique_vars:
        for var in low_unique_vars:
            print(f"  - {var} (Nbre de valeurs uniques: {unique_counts[var]})")
    else:
        print("  None found.")
    print("---------------------------\n")

    return low_unique_vars, unique_counts


import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_distributions(var_list, X_train, X_test, rows=3, figsize=(8, 6)):
    """
    Affiche les distributions KDE des variables continues dans train et test en organisant les graphiques en lignes.
    
    """
    cols = rows  # Nombre de colonnes (par défaut égal au nombre de graphiques par ligne)
    num_vars = len(var_list)
    num_rows = (num_vars // cols) + int(num_vars % cols > 0)  # Calcul du nombre de lignes nécessaires
    
    fig, axes = plt.subplots(num_rows, cols, figsize=(figsize[0], figsize[1] * num_rows))
    axes = axes.flatten()  # Aplatir pour accéder aux axes facilement
    
    for i, col in enumerate(var_list):
        sns.kdeplot(X_train[col], label='Train', shade=True, ax=axes[i])
        sns.kdeplot(X_test[col], label='Test', shade=True, ax=axes[i])
        axes[i].set_title(f"Distribution de {col}")
        axes[i].legend()
    
    # Supprime les axes inutilisés s'il y a moins de variables que de sous-graphiques
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()


from scipy.stats import ks_2samp

def extract_vars_dif_dist(var_list, X_train, X_test, threshold=0.05):
    """
    Extrait les variables dont les distributions de train et test diffèrent significativement (p-value < threshold).
    """
    significant_dif_dist_vars = {}  # Stockage des variables significatives avec leurs p-valeurs
    
    for var in var_list:
        # Calcul des statistiques descriptives
        print(f"Statistiques descriptives pour {var} :")
        print(f"Train :\n{X_train[var].describe()}")
        print(f"Test :\n{X_test[var].describe()}\n")
        
        # Test de Kolmogorov-Smirnov
        ks_stat, ks_p_value = ks_2samp(X_train[var], X_test[var])
        print(f"Test de Kolmogorov-Smirnov pour {var} : p-value = {ks_p_value:.6f}")
        
        if ks_p_value < threshold:
            print(f"-> Les distributions diffèrent significativement (p-value < {threshold}).\n")
            significant_dif_dist_vars[var] = ks_p_value
        else:
            print(f"-> Les distributions sont similaires (p-value >= {threshold}).\n")
        
        print("-" * 50)
    
    return significant_dif_dist_vars


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
