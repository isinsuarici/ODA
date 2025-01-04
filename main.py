import streamlit as st
import params
import helper_funcs
from pathlib import Path

def main():
    # Page configuration
    st.set_page_config(
        page_title="Parasitic Egg Detection using YOLOv8",
        page_icon="logo.png",
        layout="wide"
    )

    # Custom CSS to hide Streamlit's default elements
    hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                header {visibility: hidden;}
                </style>
                """
    st.markdown(hide_st_style, unsafe_allow_html=True)

    # Title and header
    st.markdown(
    """
    <style>
        .title {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .title img {
            width: 100px;  # Adjust logo size as needed
            height: 100px;
            margin-right: 10px;  # Space between logo and title
        }
    </style>
    <div class="title">
        <img src="https://images.pexels.com/photos/29834764/pexels-photo-29834764.png?auto=compress&cs=tinysrgb&w=600&lazy=load"/>
        <h1 style="color: #d1bcce;">Parasitic Object Detection in Human Stool Using YOLOv8</h1>
    </div>
    """,
    unsafe_allow_html=True,
)

    st.markdown("---")

    # Load YOLO model
    model_path = Path(params.MODEL_DIR)  # Update with your model path
    model = helper_funcs.load_model(model_path)

    if model:
        helper_funcs.detect_objects_in_image(model)
    else:
        st.error("Model could not be loaded. Check the path or file.")

    st.markdown("---")
    st.markdown(
        """
        <style>
            .about-card {
                text-align: center;  /* Center text inside the card */
            }
            .about-card h2 {
                color: #a6b0b9;  /* Muted text color */
            }
            .about-card p {
                font-size: 16px;  /* Adjust text size */
                line-height: 1.6;  /* Improve readability */
                color: #a6b0b9;  /* Dark text for contrast */
            }
            .about-card a {
                color: #007bff;  /* Link color */
                text-decoration: none;
            }
            .about-card a:hover {
                text-decoration: underline;  /* Underline on hover */
            }
        </style>
        <div class="about-card">
            <h2>About</h2>
            <p>This application employs the YOLOv8 model to detect specific parasitic objects in human stool samples, offering a faster and more efficient tool to assist healthcare professionals in diagnosis.</p>
            <p>The model is trained to recognize 34 parasitic species, covering a diverse range of protozoa and helminths. Among the protozoa, it can detect Entamoeba histolytica (E. histolytica), Entamoeba coli (E. coli), Endolimax nana (E. nana), Iodamoeba butschlii (I. butschlii), Blastocystis spp., and Giardia duodenalis (G. duodenalis).</p>
            <p>The helminths include Ascaris lumbricoides (fertilized, unfertilized, and decorticated eggs), Enterobius vermicularis (E. vermicularis), Trichuris trichiura (T. trichiura), hookworm, Strongyloides stercoralis (S. stercoralis), Trichostrongylus orientalis (T. orientalis), Toxocara spp., Capillaria philippinensis (C. philippinensis), Fasciolopsis buski (F. buski), Echinostoma spp., Haplorchis spp., Gastrodiscoides hominis (G. hominis), Schistosoma japonicum (S. japonicum), Schistosoma mekongi (S. mekongi), Schistosoma mansoni (S. mansoni),  Schistosoma haematobium (S. haematobium), Opisthorchis viverrini (O. viverrini), Eurytrema pancreaticum (E. pancreaticum), Fasciola spp., Paragonimus spp., Taenia spp., Hymenolepis nana (H. nana), Hymenolepis diminuta (H. diminuta), Dipylidium caninum (D. caninum), Diphyllobothrium latum (D. latum), and Spirometra spp.</p>
            <h3>Disclaimer:</h3>
            <p>This tool is for educational and research purposes only and should not be used for clinical diagnosis. Always consult a healthcare professional for accurate results.</p>
            <h3>Contact Information:</h3>
            <p>For inquiries or feedback, please email:</p>
            <p>Rojelyn Laguinan, <a href="mailto:rlaguinan01491@usep.edu.ph">rlaguinan01491@usep.edu.ph</a>
            <p>Remart Maynantay, <a href="mailto:rsmaynantay01498@usep.edu.ph">rsmaynantay01498@usep.edu.ph</a>
        </div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
