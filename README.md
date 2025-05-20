<p>This API consists in a Streamlit dashboard used for a Proof of Concept of a Vision Transformer for Image classification.
This dashboard is written in French and respects the WCAG 2.1. criteria.</p>

<p>The first tab allows the user to explore the training dataset. It shows bar plots of the label distributions
and it shows extracts of the dataset when the user enters label names in the different forms.</p>

<p>The second tab permits to perform 2 real-time predictions on an image picked randomly among the class chosen in the form.
The image and the scores of the 2 model predictions are displayed : the Vision Transformer and the ResNet's.</p>

<p>The las tab shows plots summarizing the main progress done by the Vision Transformer compared to the ResNet model.</p>

<p>Important : Due to privacy issues, the codes used to load the models (utils.py) will not be shared to the public.
Hence, it is not possible to deploy the API on a local computer.</p>