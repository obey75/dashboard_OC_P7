import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
@st.cache
def load_data():
    # Remplacez par le chemin de vos données
    data = pd.read_pickle('full_dataset.pkl')
    data["label1"] = data["label1_combined"].apply(lambda x: x[0] if x[0] else "")
    data["label2"] = data["label2_combined"].apply(lambda x: x[0] if x[0] else "")
    return data

data = load_data()
print(data.iloc[0].label1_combined[0])

# Définir les onglets
tabs = ["Analyse Exploratoire", "Prédiction"]
page = st.sidebar.radio("Aller à", tabs)

if page == "Analyse Exploratoire":
    st.title("Analyse Exploratoire de Données")
    st.write("Cette section contient des analyses exploratoires des données.")

    # Exemple d'analyse exploratoire
    st.subheader("Distribution des catégories")
    fig, ax = plt.subplots()
    sns.countplot(x='label1', data=data, ax=ax)
    st.pyplot(fig)

    st.subheader("Distribution des sous-catégories")
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.countplot(x='label2', data=data, ax=ax)
    st.pyplot(fig)

elif page == "Prédiction":
    pass
    st.title("Prédiction avec Modèles")
    st.write("Cette section permet de faire des prédictions avec deux modèles différents.")

    # Exemple de modèles de prédiction
    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ancien modèle
    old_model = RandomForestClassifier(n_estimators=10)
    old_model.fit(X_train, y_train)
    old_predictions = old_model.predict(X_test)
    old_accuracy = accuracy_score(y_test, old_predictions)

    # Nouveau modèle
    new_model = RandomForestClassifier(n_estimators=100)
    new_model.fit(X_train, y_train)
    new_predictions = new_model.predict(X_test)
    new_accuracy = accuracy_score(y_test, new_predictions)

    st.subheader("Performance des modèles")
    st.write(f"Ancien modèle: {old_accuracy:.2f}")
    st.write(f"Nouveau modèle: {new_accuracy:.2f}")

    st.subheader("Faire une prédiction")
    input_data = st.text_input("Entrez les données pour la prédiction (séparées par des virgules)")
    if input_data:
        input_data = [float(x) for x in input_data.split(',')]
        old_pred = old_model.predict([input_data])
        new_pred = new_model.predict([input_data])
        st.write(f"Prédiction avec l'ancien modèle: {old_pred[0]}")
        st.write(f"Prédiction avec le nouveau modèle: {new_pred[0]}")