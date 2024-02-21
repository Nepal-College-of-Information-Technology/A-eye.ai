import streamlit as st

# Set the title of the new page
st.markdown("<h1 style='text-align: center;'> A-eye.ai: An AI to Take Care of Your Eye </h1>", unsafe_allow_html=True)


# Introduction
st.markdown('<div class="introduction">Introduction</div>', unsafe_allow_html=True)

st.write("""
Early ocular illness identification is a cost-efficient and reliable technique to stop blindness brought on by conditions including diabetes, glaucoma, cataracts, and many others. The World Health Organization (WHO) estimates that there are currently at least 2.2 billion vision impairments worldwide, of which at least 1 billion might have been avoided. In order to lessen the strain of the ophthalmologist and save patients' vision from being damaged, rapid and automatic illness diagnosis is essential and urgent. Following the provision of high-quality medical eye fundus photos, computer vision and deep learning may automatically identify ocular disorders.
""")

# Dataset
st.markdown('<div class="subsection">Dataset</div>', unsafe_allow_html=True)
st.write("""
Ocular Disease Intelligent Recognition (ODIR) is a structured ophthalmic database of 5,000 patients with age, color fundus photographs from left and right eyes and diagnostic keywords from doctors.

This dataset is meant to represent ‘‘real-life’’ set of patient information collected by Shanggong Medical Technology Co., Ltd. from different hospitals/medical centers in China. In these institutions, fundus images are captured by various cameras in the market, such as Canon, Zeiss and Kowa, resulting into varied image resolutions. Annotations were labeled by trained human readers with quality control management. Patients are classified into various labels, where we have used the following 5:

- Normal (N)
- Diabetes (D)
- Glaucoma (G)
- Cataract (C)
- Other diseases/abnormalities (O)

[Link to the dataset](https://www.kaggle.com/datasets/andrewmvd/ocular-disease-recognition-odir5k)
""")

# Images Examples
st.markdown('<div class="subsection">Dataset Images Examples</div>', unsafe_allow_html=True)

row1_col1, row1_col2 = st.columns(2)
row2_col1, row2_col2 = st.columns(2)

# First Row
with row1_col1:
    st.image('images/diseases/glaucoma.jpg', caption='Glaucoma', use_column_width=True)

with row1_col2:
    st.image('images/diseases/cataract.jpg', caption='Cataract', use_column_width=True)

# Second Row
with row2_col1:
    st.image('images/diseases/diabetes.jpg', caption='Diabetes', use_column_width=True)

with row2_col2:
    st.image('images/diseases/myopia.jpg', caption='Myopia', use_column_width=True)

# ML Approaches
st.markdown('<div class="subsection">ML Approaches (accuracy)</div>', unsafe_allow_html=True)
st.write("""
- Resnet50 (Accuracy :0.60)
  ![Resnet50]

- vgg16 (Accuracy :0.75)
  ![vgg16]

- vgg19 (Accuracy :0.88)
  ![vgg19]

- Sequential(Accuracy: 0.45)
  ![CNN]
""")


