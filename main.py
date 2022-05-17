from numpy import reshape
import streamlit as st
from models import decoder_ae, decoder_vae
from funcs import improve_img, download_image
import torch

st.set_page_config(page_title='Face Generator', page_icon="🤓", layout="wide")

@st.cache(show_spinner=True)
def load_ae_model():
    #print("Loading AE model..")
    model = torch.load("model_ae.pth")
    model.eval()
    return model

@st.cache(show_spinner=True)
def load_vae_model():
    #print("Loading VAE model..")
    model = torch.load("model_vae.pth")
    model.eval()
    return model

decoder = decoder_ae
model_ae = load_ae_model()

decoder = decoder_vae
model_vae = load_vae_model()

st.title("Face Generation using Deep Learning")
st.subheader("Use this app to generate random and customized faces")
st.markdown("Inspired from [this-person-does-not-exist](https://this-person-does-not-exist.com/en)", )
st.markdown("***")

random_face, cust_face = st.columns(2)

with random_face:
    with st.expander("Random Face Generation", expanded=True):
        st.subheader("Generate Random Image")
        dist = st.radio("Choose distribution", ("Standard Normal", "Uniform Random"))
        reshaping = st.selectbox("Select Image Resolution (None denotes 64x64)", (None, 86, 128, 512, 1024),key=65)
        if dist=="Uniform Random": input = torch.rand(1, 128)
        else: input = torch.randn(1, 128)

        if st.button("Generate", key=1):
            st.info("Click generate again to get new image")
            out_img = model_vae(input)
            sharp_img = improve_img(out_img, reshaping)
            if reshaping is None:
                caption = "Generated Image (64x64)"
            else:
                caption = f"Generated Image ({reshaping}x{reshaping})"

            st.image(sharp_img, caption=caption, clamp=True, width=300)

            if st.download_button("Download⬇️", download_image(sharp_img), "Random Image.png", mime='Image/png'):
                pass


with cust_face:
    with st.expander("Customized Face Generator", expanded=True):
        st.subheader("Generate Customized Image")
        input = [[-1, -1, -1,
                    -1, 1, 1, -1, -1,
                    1, -1, -1, -1, -1,
                    1, 1, -1, -1, -1,
                    1, 1, -1, 1,
                    -1, -1, 1, -1, -1,
                    1, -1, 1, -1,
                    1, 1, -1, 1,
                    -1, 1, 1,
                    -1, 1]]
        reshaping = st.selectbox("Select Image Resolution (None denotes 64x64)", (None, 86, 128, 512, 1024), key=66)
        if reshaping is None:
                caption = "Generated Image (64x64)"
        else:
            caption = f"Generated Image ({reshaping}x{reshaping})"

        if st.button("Generate", key=2):
            input = torch.tensor(input).type(torch.FloatTensor)
            out_img = model_ae(input)
            sharp_img = improve_img(out_img)
            st.image(sharp_img, caption=caption, clamp=True, width=300)