
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
