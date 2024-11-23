
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
        if unique_count < threshold:  # Vérifier si le nombre est inférieur au seuil
            low_unique_vars.append(col)
            unique_counts[col] = unique_count
    
    print("----- Results Summary -----")
    print(f"Threshold for low unique values: {threshold}")
    print("\nVariables with low unique values:")
    if low_unique_vars:
        for var in low_unique_vars:
            print(f"  - {var} (Unique Count: {unique_counts[var]})")
    else:
        print("  None found.")
    print("---------------------------\n")

    return low_unique_vars