import streamlit as st
import random
import pandas as pd
from utils import load_models, load_mlb_objects
from utils import preprocess_input, predict_vit_on_img, predict_resnet_on_img, format_output_vit, format_output_resnet



# Load models, dataset and MLBs

@st.cache_resource
def cache_models():
    return load_models()


@st.cache_data
def cache_data():
    data = pd.read_pickle('full_dataset.pkl')
    return data


@st.cache_data
def cache_mlb():
    mlb_level_1, mlb_level_2 = load_mlb_objects()
    return mlb_level_1, mlb_level_2


resnet_model, vit_model = cache_models()
data = cache_data()
mlb_level_1, mlb_level_2 = cache_mlb()



tabs = ["Analyse Exploratoire", "Prédictions", "Résultats"]
page = st.sidebar.radio("Aller à", tabs)

if page == "Analyse Exploratoire":

    # T1 level
    st.title("Analyse des T1")
    st.image("plots/T1_distribution.png", caption="")
    st.text("Répartition du jeu de donneés d'entraînement :")
    st.text("'Automotive': 1,000 ; 'Family_and_relationships': 900")

    suggestions_T1 = mlb_level_1.classes_

    # User chooses a label
    with st.form("formulaire_T1"):
        T1 = st.selectbox(
            "Pour explorer les images, choisissez un T1 :",
            suggestions_T1
        )
        submit_button_T1 = st.form_submit_button("OK")

    if submit_button_T1:
        # 3 images corresponding to this label are chosen randomly and displayed
        images = list(data.loc[data['label1_combined'].apply(lambda x: T1 in x), "image_name"])
        ids = [random.randint(0, len(images)-1) for _ in range(3)]
        l_img_path = ["images/train/" + images[id] for id in ids]
        cols = st.columns(3)
        for j, img_path in enumerate(l_img_path[:3]):
            cols[j].image(img_path, caption=f"Image {j+1}", use_column_width='always')

    # T2 level Automotive
    st.title("Analyse des T2 de la classe 'Automotive'")
    st.image("plots/automotive_distribution.png", caption="")
    st.text("Répartition du jeu de donneés d'entraînement :")
    st.text("'city_car': 330 ; 'hybrid_and_electric': 325 ; 'scooters': 235")

    suggestions_T2_automotive = ["city_car", "hybrid_and_electric", "scooters"]

    with st.form("formulaire_T2_automotive"):
        T2_automotive = st.selectbox(
            "Pour explorer les images, choisissez un T2 de la classe Automotive :",
            suggestions_T2_automotive
        )
        submit_button_T2_automotive = st.form_submit_button("OK")

    if submit_button_T2_automotive:

        images = list(data.loc[data['label2_combined'].apply(lambda x: T2_automotive in x), "image_name"])
        ids = [random.randint(0, len(images)-1) for _ in range(3)]
        l_img_path = ["images/train/" + images[id] for id in ids]
        cols = st.columns(3)
        for j, img_path in enumerate(l_img_path[:3]):
            cols[j].image(img_path, caption=f"Image {j+1}", use_column_width='always')


    # T2 level Family_and_relationships
    st.title("Analyse des T2 de la classe 'Family_and_relationships'")
    st.image("plots/family_and_relationships_distribution.png", caption="")
    st.text("Répartition du jeu de donneés d'entraînement :")
    st.text("'dating': 310 ; 'parenting_babies_and_toddlers': 305 ; 'parenting_children_aged_4_11': 245 ; 'seniors': 145")

    suggestions_T2_family = ["dating", "parenting_babies_and_toddlers", "parenting_children_aged_4_11", "seniors"]

    with st.form("formulaire_T2_family"):
        T2_family = st.selectbox(
            "Pour explorer les images, choisissez un T2 de la classe 'Family_and_relationships' :",
            suggestions_T2_family
        )
        submit_button_T2_family = st.form_submit_button("OK")

    if submit_button_T2_family:

        images = list(data.loc[data['label2_combined'].apply(lambda x: T2_family in x), "image_name"])
        ids = [random.randint(0, len(images)-1) for _ in range(3)]
        l_img_path = ["images/train/" + images[id] for id in ids]
        cols = st.columns(3)
        for j, img_path in enumerate(l_img_path[:3]):
            cols[j].image(img_path, caption=f"Image {j+1}", use_column_width='always')



elif page == "Résultats":
    st.title("Scores T1")
    st.image("plots/T1_scores.png", caption="")
    st.text("Évolutions respectives pour les classes 'Family_and_relationships' et 'Automotive':")
    st.text("F1 Score: +6%, +6% ; Précision: +13%, +4% ; Rappel: -2%, +8%.")

    st.title("Scores T2 de la classe 'Automotive'")
    st.image("plots/automotive_scores.png", caption="")
    st.text("Évolutions pour les classes T2 de la classe 'Automotive':")
    st.text("F1 Score: +47% ; Précision: -1% ; Rappel: +42%.")

    st.title("Scores T2 de la classe 'Family_and_relationships'")
    st.image("plots/family_and_relationships_scores.png", caption="")
    st.text("Évolutions pour les classes T2 de la classe 'Family_and_relationships':")
    st.text("F1 Score: +36% ; Précision: +62% ; Rappel: +25%.")

    st.title("Performances")
    st.image("plots/time_performances.png", caption="")
    st.text("Évolutions des performances: ")
    st.text("Vitesse d'entraînement (durée d'une époque): -23% ;")
    st.text("Vitesse de prédiction (temps d'inférence): -3%")



elif page == "Prédictions":
    st.title("Prédiction et comparaison des modèles 'Vision Transformer' et 'ResNet'")

    # User chooses a T2 label
    suggestions_T2 = mlb_level_2.classes_
    with st.form("formulaire_T2"):
        T2 = st.selectbox(
            "Pour réaliser des prédictions sur une image, choisissez un T2 :",
            suggestions_T2
        )
        submit_button_T2 = st.form_submit_button("OK")

    if submit_button_T2:
        # Picks randomly an image corresponding to this label
        df_images = data.loc[data['label2_combined'].apply(lambda x: T2 in x), :]
        images = list(df_images["image_name"])
        id = random.randint(0, len(images)-1)
        img_path = "images/train/" + images[id]
        st.image(img_path, caption="", width=600)

        # Processes predictions with ViT and ResNet
        with st.spinner('Prédiction en cours...'):
            input= preprocess_input(img_path)
            output_vit = predict_vit_on_img(input, vit_model)
            res_vit = format_output_vit(output_vit, mlb_level_1, mlb_level_2)
            output_resnet = predict_resnet_on_img(input, resnet_model)
            res_resnet = format_output_resnet(output_resnet, mlb_level_1, mlb_level_2)

        # Displays results for ViT and ResNet
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("ViT")
            for res_T1, score_T1 in res_vit[0].items():
                st.metric(res_T1, f"{score_T1*100:.2f} %")
            for res_T2, score_T2 in res_vit[1].items():
                st.metric(res_T2, f"{score_T2*100:.2f} %")

        with col2:
            st.subheader("ResNet")
            for res_T1, score_T1 in res_resnet[0].items():
                st.metric(res_T1, f"{score_T1*100:.2f} %")
            for res_T2, score_T2 in res_resnet[1].items():
                st.metric(res_T2, f"{score_T2*100:.2f} %")
