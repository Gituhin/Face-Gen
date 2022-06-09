import streamlit as st
from models import decoder_ae, decoder_vae
from funcs import improve_img, download_image
import torch
import math

st.set_page_config(page_title='Face Generator', page_icon="Logo.png", layout="wide")
st.write('<style>div.row-widget.stRadio>  div{flex-direction:row;}</style>', unsafe_allow_html=True)        

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

st.title("Face Generator")
st.subheader("Use this app to generate random and customized fake faces")
st.markdown("***")

random_face, cust_face = st.columns(2)

with random_face:
    with st.expander("Random Face Generation", expanded=True):
        st.subheader("Generate Random Image")
        st.write("\n") #added extra spaces
        st.write("\n")
        st.write("\n")
        is_seed = st.checkbox("Use manual seed", value=False)
        st.write("\n") #added extra spaces
        st.write("\n")
        st.write("\n")
        if is_seed:
            st.info("This will generate same random image in every trial")
            seed = st.number_input("Enter seed integer", min_value=0)
            torch.manual_seed(seed)
        
        dist = st.radio("Choose Sampling distribution", ("Standard Normal", "Uniform Continuous"))
        st.write("\n") #added extra spaces
        st.write("\n")
        st.write("\n")
        reshaping = st.selectbox("Select Image Resolution in pixels", (64, 128, 256, 512, 1024),key=65)
        st.write("\n") #added extra spaces
        st.write("\n")
        st.write("\n")
        bright = st.slider("Set brightness level", min_value=0.1, max_value=2.0, step=0.1, value=1.0)
        
        if dist=="Uniform Continuous": input = torch.rand(1, 128)
        else: input = torch.randn(1, 128)
        st.write("\n") #added extra spaces
        st.write("\n")
        st.write("\n")
        if st.button("Generate", key=1):
            if not is_seed:
                st.info("Click generate again to get new image")
            out_img = model_vae(input)
            sharp_img = improve_img(out_img, reshaping, bright)
            st.image(sharp_img, caption=f"Generated Image ({reshaping} x {reshaping} px)",
                    clamp=True, width=int(math.log2(reshaping))*45)
            st.text('\n\n')
            if st.download_button("Download⬇️", download_image(sharp_img), "Random Image.png", mime='Image/png'):
                pass

with cust_face:
    with st.expander("Customized Face Generator", expanded=True):
        st.subheader("Generate Customized faces from attributes")

        gender = st.radio("Select Gender", ("Male", "Female"))

        attr = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
                'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
                'Black', 'Blonde', 'Blurry', 'Brown', 'Bushy_Eyebrows',
                'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray',
                'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
                'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
                'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
                'Smiling', 'Straight Hair', 'Wavy Hair', 'Wearing Earrings',
                'Wearing Hat', 'Wearing_Lipstick', 'Wearing Necklace',
                'Wearing Necktie', 'Young']
        
        attr_dic = dict(zip(attr, 40*[-1]))

        hair = st.radio("Hair", ('Black', 'Blonde', 'Brown', 'Gray', 'Bald'))
        hair_type = st.radio("Hair Type", ("Straight Hair", "Wavy Hair"))
        eyeglass = st.checkbox("Eyeglasses")
        smile = st.checkbox("Smiling")
        fat = st.checkbox("Make fat")
        attractive = st.checkbox("Attractive and young")
    
        if gender == "Male":
            make_true = ['Male'] # use this to change dict values
            facial_hair = st.checkbox("Beard and Mustache") # bool
            attr = st.multiselect("Apparels",
                    ['Wearing Hat', 'Wearing Necktie', 'Sideburns'])

            #start editing dict
            make_true.extend([hair, hair_type])
            make_true.extend(attr)
            if eyeglass: make_true.append("Eyeglasses")
            if facial_hair:make_true.extend(["Goatee", "Mustache"])    
            if smile: make_true.extend(["Mouth_Slightly_Open", "Smiling"])
            if fat: make_true.extend(["Big_Lips", "Big_Nose", "Chubby", "Double_Chin"])
            if attractive:
                make_true.extend(["Young", "Attractive", "Oval_Face", "Narrow_Eyes", "Pointy_Nose", "Bushy_Eyebrows"])
                if "Big_Nose" in make_true:
                    make_true.remove("Big_Nose")

            if not facial_hair: make_true.append("No_Beard")

            for i in make_true: attr_dic[i] = 1

        else: #female
            make_true = []
            makeup = st.checkbox("Makeup")
            arched_eyebrows = st.checkbox("Arched Eyebrows")
            attr = st.multiselect("Apparels",
                    ['Wearing Hat', 'Wearing Necktie', 'Wearing Necklace', 'Wearing Earrings'])
            #start editing dict
            make_true.extend([hair, hair_type])
            make_true.extend(attr)
            if eyeglass: make_true.append("Eyeglasses")   
            if smile: make_true.extend(["Mouth_Slightly_Open", "Smiling"])
            if fat: make_true.extend(["Big_Lips", "Big_Nose", "Chubby", "Double_Chin"])
            if makeup: make_true.extend(['Heavy_Makeup', 'Wearing_Lipstick', 'Rosy_Cheeks', 'High_Cheekbones'])
            if arched_eyebrows: make_true.append("Arched_Eyebrows")
            if attractive:
                make_true.extend(["Young", "Attractive", "Oval_Face", "Narrow_Eyes", "Pointy_Nose"])
                if "Big_Nose" in make_true:
                    make_true.remove("Big_Nose")

            for i in make_true: attr_dic[i] = 1
            attr_dic['No_Beard'] = 1
            

        input = [list(attr_dic.values())]
        reshaping = st.selectbox("Select Image Resolution in Pixels", (64, 128, 256, 512, 1024), key=66)
        bright = st.slider("Set brightness level", min_value=0.1, max_value=2.0, step=0.1, value=1.0, key=2)

        if st.button("Generate", key=2):
            input = torch.tensor(input).type(torch.FloatTensor)
            out_img = model_ae(input)
            sharp_img = improve_img(out_img, reshaping, bright)
            st.image(sharp_img, caption=f"Generated Image ({reshaping} x {reshaping}px)", clamp=True, width=int(math.log2(reshaping))*45)
            if st.download_button("Download⬇️", download_image(sharp_img), "Random Image.png", mime='Image/png'):
                pass

st.markdown("***")
st.markdown("***")
st.text('\n\n')
st.subheader("Assests and Sources")
st.write('''This project is inspired from Nvidia Labs [this-person-does-not-exist](https://this-person-does-not-exist.com/en). The models are trained on [CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
            using a single pytorch's XLA device (Accelerated Linear Algebra Compiler) for 3-3.5 cloud TPU hours.''')
st.write("To access source code visit Github Repository [Gituhin/Face-Gen](https://github.com/Gituhin/Face-Gen/tree/master)")
st.write("To know more in details visit [Website/FG](https://sites.google.com/view/tuhinsubhrade/worksprojects/face-generator)")
st.markdown("***")
st.text("\n\n")
st.text("sudo generatef --Gituhin == 2022😉")