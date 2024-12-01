import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
from scipy.stats import chi2_contingency
from scipy.stats import skew, kurtosis
from collections import Counter

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

def plot_cat_vars_distributions(data, vars_list, cols=2):
    """
    Génère des graphiques montrant la distribution des modalités pour chaque variable dans une grille.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile
    
    for i, var in enumerate(vars_list):
        value_counts = data[var].value_counts(normalize=True) 
        index_values = value_counts.index.to_flat_index()  
        index_values = [str(x) for x in index_values]   
        
        # Création du graphique
        axes[i].bar(index_values, value_counts.values, color='skyblue')
        axes[i].set_ylabel('Proportion')
        axes[i].set_title(f'{var}')
        axes[i].tick_params(axis='x', rotation=45)
    
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Distribution des modalités", fontsize=16, y=1.02) 
    
    plt.tight_layout()
    plt.show()


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
    rows = math.ceil(num_vars / cols) 

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


def combined_barplot_lineplot(df, cat_vars, cible, cols=2):
    """
    Génère une grille de barplots combinés avec des lineplots pour une liste de variables catégorielles.
    """
    num_vars = len(cat_vars)
    rows = math.ceil(num_vars / cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, cat_col in enumerate(cat_vars):
        ax1 = axes[i]  # Axe pour le barplot

        # Vérifier si la variable est catégorielle et la convertir en chaîne si nécessaire
        if pd.api.types.is_categorical_dtype(df[cat_col]):
            df[cat_col] = df[cat_col].astype(str)

        # Calcul du taux de risque
        tx_rsq = (df.groupby([cat_col])[cible].mean() * 100).reset_index()

        # Calcul des effectifs
        effectifs = df[cat_col].value_counts().reset_index()
        effectifs.columns = [cat_col, "count"]

        # Fusion des données
        merged_data = effectifs.merge(tx_rsq, on=cat_col).sort_values(by=cible, ascending=True)

        # Création des graphiques
        ax2 = ax1.twinx()  # Deuxième axe pour le lineplot
        sns.barplot(data=merged_data, x=cat_col, y="count", color='grey', ax=ax1)
        sns.lineplot(data=merged_data, x=cat_col, y=cible, color='red', marker="o", ax=ax2)

        # Configuration des axes
        ax1.set_title(f"{cat_col}")
        ax1.set_xlabel("")
        ax1.set_ylabel("Effectifs")
        ax2.set_ylabel("Taux de risque (%)")
        ax1.tick_params(axis='x', rotation=45)

    # Supprimer les axes inutilisés si le nombre de variables est inférieur à la grille
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Titre général
    fig.suptitle("Barplots et Lineplots combinés pour les variables catégorielles", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

def compare_distributions_grid(X_train, X_test, var_list, cols=2):
    """
    Compare les distributions des variables continues dans Train et Test et les affiche sous forme de grille.
    """
    num_vars = len(var_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(var_list):
        # Graphique pour chaque variable
        sns.kdeplot(X_train[var], label='Train', shade=True, ax=axes[i])
        sns.kdeplot(X_test[var], label='Test', shade=True, ax=axes[i])
        axes[i].set_title(f"{var}")
        axes[i].legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    fig.suptitle("Comparaison des distributions dans l'echantillon Train et Test", fontsize=16, y=1.02) 

    plt.tight_layout()
    plt.show()


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


def plot_boxplots(data, vars_list, cols=2):
    """
    Affiche des boxplots pour chaque variable continue dans une grille.
    """
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols) 
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour itération facile

    for i, var in enumerate(vars_list):
        sns.boxplot(data=data, y=var, ax=axes[i],showfliers=False)  # Création du boxplot
        axes[i].set_title(f"Boxplot de {var}")
        axes[i].set_xlabel("")  # Pas besoin d'étiquette pour x
        axes[i].set_ylabel("Valeurs")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("Distribution des variables continues", fontsize=16, y=1.02)  # Positionner le titre légèrement au-dessus

    plt.tight_layout()
    plt.show()


import math
import matplotlib.pyplot as plt
import seaborn as sns

def plot_kde_by_target(data, vars_list, target, cols=2):
    """
    Génère des boxplots (et KDE plots optionnellement) pour chaque variable continue en fonction des valeurs cibles fournies.
    
    Parameters:
    - data (DataFrame): Les données contenant les variables.
    - vars_list (list): Liste des variables continues à tracer.
    - target (str): Nom de la variable cible.
    - cols (int): Nombre de colonnes de la grille de subplots.
    - kde (bool): Si True, génère des KDE plots en plus des boxplots.
    """
    data = data.copy()
    num_vars = len(vars_list)
    rows = math.ceil(num_vars / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()  # Aplatir les axes pour faciliter l'itération

    for i, var in enumerate(vars_list):
        sns.kdeplot(
            data=data, 
            x=var, 
            hue=target, 
            ax=axes[i], 
            fill=True, 
            palette="Set2", 
            common_norm=False, 
            alpha=0.5
        )
        axes[i].set_title(f"KDE Plot de {var} par {target}")
        axes[i].set_xlabel(target)
        axes[i].set_ylabel(var)

    # Supprimer les axes inutilisés
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"KDE des variables continues selon la variable cible", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()




def cramer_V(cat_var1,cat_var2):
    """
    Calcule le coefficient de Cramer's V pour une paire de variables catégorielles.
    """
    crosstab = np.array(pd.crosstab(cat_var1,cat_var2,rownames=None,colnames=None)) #tableau de contingence
    stat = chi2_contingency(crosstab)[0] #stat de test de khi-2
    obs=np.sum(crosstab) 
    mini = min(crosstab.shape)-1 #min entre les colonnes et ligne du tableau croisé ==> ddl
    return (np.sqrt(stat/(obs*mini)))

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

def test_freq_by_group(data, qualitative_vars, threshold=0.05):

    unique_mod_result = []

    for var in qualitative_vars:
        value_counts = data[var].value_counts(normalize=True)  # Normalisation des fréquences
        if value_counts.iloc[0] <=threshold:
            unique_mod_result.append(var)
    if len(unique_mod_result)==0:
        print("Aucune variable n'a de modalités avec moins de 5% d'effectifs.")
    return unique_mod_result

def group_by_rsq(df, cat_var,cible):
    """
    Groupe les modalités d'une variable catégorielle qui ont une fréquence inférieure à 5% 
    en fonction de leur taux de risque moyen.
    """
    grouped_classes = []
    cumulative_weight = 0
    group = []
    risk_rates = (df.groupby(cat_var)[cible].mean()).sort_values(ascending=False)
    freq_df = df.groupby(cat_var).size() / len(df)  # Fréquence des modalités
    freq_df = pd.DataFrame({'Frequence': freq_df, 'Taux de risque': risk_rates})
    freq_df = freq_df.sort_values(by='Taux de risque', ascending=False)
        
    # Affichage du DataFrame avec modalités, fréquences et taux de risque
    print(freq_df)
    for i, (interval, risk) in enumerate(risk_rates.items()):
        freq = df[df[cat_var] == interval].shape[0] / df.shape[0]
        group.append(interval)
        cumulative_weight += freq
        
        # Regrouper les classes pour que chaque groupe contienne au moins 5% de la population
        if cumulative_weight >= 0.05:
            grouped_classes.append(group)
            group = []
            cumulative_weight = 0

    # Gestion du dernier groupe (si existant)
    if group:
        last_group_weight = sum(df[df[cat_var] == g].shape[0] / df.shape[0] for g in group)
        if last_group_weight < 0.05 and grouped_classes:
            # Ajouter le dernier groupe au groupe précédent pour respecter la contrainte
            grouped_classes[-1].extend(group)
        else:
            # Ajouter le dernier groupe si la contrainte est respectée
            grouped_classes.append(group)
    return grouped_classes


import pandas as pd

def discretize_by_groups(df, cat_var, grouped_modalities):
    """
    Discrétise la variable catégorielle selon les modalités regroupées en fonction de leur taux de risque moyen.
    """
    # Créer un dictionnaire de mapping des groupes
    group_mapping = {}
    for idx, group in enumerate(grouped_modalities):
        for modality in group:
            group_mapping[modality] = f'Group_{idx + 1}'
    
    # Appliquer la discrétisation en fonction du dictionnaire
    df['Discretized'] = df[cat_var].map(group_mapping)
    
    return df['Discretized']


#Pour discrétiser une variable continue Weighted of Evidence
def iv_woe(data,target,bins=5,show_woe=False,epsilon=1e-16):
    newDF,woeDF = pd.DataFrame(),pd.DataFrame()
    cols=data.columns

    #Run WOE and IV on all independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars]))>10):
            binned_x = pd.qcut(data[ivars],bins,duplicates="drop")
            d0=pd.DataFrame({'x':binned_x,'y':data[target]})
        else:
            d0=pd.DataFrame({'x':data[ivars],'y':data[target]})

        #calculate the nb of events in each group (bin)

        d=d0.groupby("x",as_index=False).agg({"y":["count", "sum"]})
        d.columns = ["Cutoff","N","Events"]

        #calculate % of events in each group
        d['% of Events']=np.maximum(d['Events'],epsilon)/(d['Events'].sum()+epsilon)

        #calculate the non events in each group
        d['Non-Events']=d['N'] - d['Events']
        #calculate % of non-events in each group
        d['% of Non-Events']=np.maximum(d['Non-Events'],epsilon)/(d['Non-Events'].sum()+epsilon)

        #calculate WOE by taking natural log of division of % of non-events and % of events
        d['WoE'] = np.log(d['% of Events']/d['% of Non-Events'])
        d['IV'] = d['WoE']*(d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0,column="Variable",value=ivars)
        print("--------------------------------------------\n")
        print("Information value of variable " + ivars + " is " + str(round(d["IV"].sum(),6)))
        temp=pd.DataFrame({"Variable":[ivars],"IV":[d["IV"].sum()]},columns=["Variable","IV"])
        newDF=pd.concat([newDF,temp],axis=0)
        woeDF=pd.concat([woeDF,d],axis=0)

        #show woe table
        if show_woe==True:
            print(d)
    return newDF,woeDF

def discretize_with_iv_woe(X_train, cible,date, numerical_columns, bins=5, epsilon=1e-16):
    discretized_data = X_train[[date,cible]].copy()
    discretized_columns = []
    non_discretized_columns = []

    for col in numerical_columns:
        # Appliquer la fonction iv_woe pour obtenir les points de coupure
        result = iv_woe(X_train[[col] + [cible]], cible, bins=bins, show_woe=False, epsilon=epsilon)

        if result[1]["IV"].sum() != 0:  # Si l'IV n'est pas nul, discrétiser
            # Extraire les cutoffs (intervalles)
            cutoffs = result[1]["Cutoff"].unique()
            
            # Si les cutoffs sont des intervalles, extraire les bornes
            if isinstance(cutoffs[0], pd.Interval):
                bins_edges = sorted(set([interval.left for interval in cutoffs] + [interval.right for interval in cutoffs]))
            else:
                # Sinon, traiter les cutoffs comme des valeurs discrètes (par exemple pour des variables catégoriques)
                bins_edges = sorted(cutoffs)
            
            # Discrétiser la colonne en utilisant les bornes et ajouter la colonne discrétisée avec suffixe "_cut"
            discretized_data[col + "_dis"] = pd.cut(X_train[col].copy(), bins=bins_edges, include_lowest=True, duplicates='drop')
            discretized_columns.append(col + "_dis")

            print(f"Discrétisation de la colonne {col} avec les bornes: {bins_edges}")
        else:
            discretized_data[col ] = X_train[col].copy()
            non_discretized_columns.append(col)

    return discretized_data, discretized_columns, non_discretized_columns


############# Discrétisation avec la méthode ChiMerge

class Discretization:
    ''' A process that transforms quantitative data into qualitative data '''
    
    def __init__(cls):
        print('Data discretization process started')
        
    def get_new_intervals(cls, intervals, chi, min_chi):
        ''' To merge the interval based on minimum chi square value '''
        
        min_chi_index = np.where(chi == min_chi)[0][0]
        new_intervals = []
        skip = False
        done = False
        for i in range(len(intervals)):
            if skip:
                skip = False
                continue
            if i == min_chi_index and not done:
                t = intervals[i] + intervals[i+1]
                new_intervals.append([min(t), max(t)])
                skip = True
                done = True
            else:
                new_intervals.append(intervals[i])
        return new_intervals        
        
    def get_chimerge_intervals(cls, data, colName, label, max_intervals):
        '''
            1. Compute the χ 2 value for each pair of adjacent intervals
            2. Merge the pair of adjacent intervals with the lowest χ 2 value
            3. Repeat œ and  until χ 2 values of all adjacent pairs exceeds a threshold
        '''
        
        # Getting unique values of input column
        distinct_vals = np.unique(data[colName])
        
        # Getting unique values of output column
        labels = np.unique(data[label])
        
        # Initially set the value to zero for all unique output column values
        empty_count = {l: 0 for l in labels}
        intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
        while len(intervals) > max_intervals:
            chi = []
            for i in range(len(intervals)-1):
                
                # Find chi square for Interval 1
                row1 = data[data[colName].between(intervals[i][0], intervals[i][1])]
                # Find chi square for Interval 2
                row2 = data[data[colName].between(intervals[i+1][0], intervals[i+1][1])]
                total = len(row1) + len(row2)
                
                # Generate Contigency
                count_0 = np.array([v for i, v in {**empty_count, **Counter(row1[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(row2[label])}.items()])
                count_total = count_0 + count_1
                
                # Find the expected value by the following formula
                # Expected Value → ( Row Sum * Column Sum ) / Total Sum
                expected_0 = count_total*sum(count_0)/total
                expected_1 = count_total*sum(count_1)/total
                chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
                
                # Store the chi value to find minimum chi value
                chi_ = np.nan_to_num(chi_)
                chi.append(sum(chi_))
            min_chi = min(chi)
            
            intervals = cls.get_new_intervals(intervals, chi, min_chi)
        print(' Min chi square value is ' + str(min_chi))
        return intervals


def discretize_with_intervals(data, intervals_by_variable, date, cible):
    """
    Discrétise plusieurs colonnes d'un DataFrame en fonction des intervalles spécifiés dans le dictionnaire
    et retourne la liste des nouvelles variables créées.
    """
    df = data[[date, cible]].copy()
    new_variables = []  
    
    for entry in intervals_by_variable:
        variable = entry['variable']
        intervals = entry['intervals']
        labels = [
            f"[{intervals[i][0]}-{intervals[i][1]}]" if i == 0 else f"({intervals[i][0]}-{intervals[i][1]}]"
            for i in range(len(intervals))
        ] # Créer les labels pour chaque intervalle avec la borne inférieure exclue (sauf pour le premier intervalle) et la borne supérieure incluse
        
        # Nom de la nouvelle colonne
        new_col_name = f"{variable}_Dis"
        
        # Discrétisation
        df[new_col_name] = pd.cut(
            data[variable],
            bins=[intervals[0][0]] + [i[1] for i in intervals],  # Convertir intervalles en bornes
            labels=labels,
            include_lowest=True,
            right=True
        )
        
        new_variables.append(new_col_name)
    
    return df, new_variables



import pandas as pd
from scipy.stats import f_oneway

def perform_anova(df, continuous_var, target_name):
    anova_result = []
    for col in continuous_var :
        df_clean = df[[col,target_name]].dropna(axis=0)
        group=[group for _, group in df_clean.groupby(target_name)[col]]
        statistic, pvalue = stats.kruskal(*group)
        anova_result.append([col,statistic,pvalue])
    result_df = pd.DataFrame(anova_result,columns=["Columns","Stat","Pvalue"])
    return result_df

### test de Kruskall Wallis
def perform_kruskal_wallis(df, continuous_var,target_name):
    kruskal_result = []
    for col in continuous_var :
        df_clean = df[[col,target_name]].dropna(axis=0)
        group=[group for _, group in df_clean.groupby(target_name)[col]]
        statistic, pvalue = stats.kruskal(*group)
        kruskal_result.append([col,statistic,pvalue])
    result_df = pd.DataFrame(kruskal_result,columns=["Columns","Stat","Pvalue"])
    return result_df