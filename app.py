import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import seaborn as sns
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora
from collections import defaultdict
from sklearn.decomposition import LatentDirichletAllocation
from streamlit_option_menu import option_menu
from PIL import Image
st.set_option('deprecation.showPyplotGlobalUse', False)

# Kamus untuk mapping nama baru
mapping = {
    'daya_tarik': 'Daya Tarik',
    'amenitas': 'Amenitas',
    'aksesibilitas': 'Aksesibilitas',
    'citra': 'Citra',
    'harga': 'Harga',
    'sdm': 'SDM'
}

# load clean dataset
data_clean = pd.read_pickle('data_clean.pickle')
data_raw = pd.read_pickle('data_raw.pickle')

# Mengubah nama kolom
data_clean = data_clean.rename(columns=mapping)
data_raw = data_raw.rename(columns=mapping)

# Membaca kembali objek CountVectorizer dari file
with open('count_vectorizer.pickle', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Memuat gambar dari file JPEG
gambar = Image.open('bg.jpeg')

# Membuat Word Cloud Untuk Background
text = " ".join(review for review in data_clean.text)
wordcloud = WordCloud(background_color="white").generate(text)

# Menampilkan Word Cloud di Streamlit
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis("off")

# Menggunakan BytesIO untuk menyimpan gambar
image_stream = BytesIO()
plt.savefig(image_stream, format="png")
plt.close(fig)

#Load File Evaluasi
with open('all_evaluations.pickle', 'rb') as evaluations_file:
    all_evaluations = pickle.load(evaluations_file)

# Loop untuk mengubah kunci
for model, features in all_evaluations.items():
    for feature in list(features.keys()):
        features[mapping[feature]] = features.pop(feature)

# Membuat DataFrame dari data evaluasi
df_evaluations = pd.DataFrame.from_dict({(model, aspect, metric): all_evaluations[model][aspect][metric] 
                                         for model in all_evaluations.keys() 
                                         for aspect in all_evaluations[model].keys() 
                                         for metric in all_evaluations[model][aspect].keys()},
                                        orient='index', columns=['Score'])

# Predict sentimen baru
def predict_sentiment(new_text, model_type='naive_bayes'):
    # Membaca kembali model dari file pickle
    with open(f'{model_type}.pickle', 'rb') as model_file:
        models = pickle.load(model_file)

    # Membaca kembali objek CountVectorizer dari file
    #with open('count_vectorizer.pickle', 'rb') as vectorizer_file:
        #vectorizer = pickle.load(vectorizer_file)

    # Mengonversi teks menggunakan CountVectorizer
    new_text_vec = vectorizer.transform([new_text])

    # Melakukan prediksi sentimen untuk setiap kategori dan model
    predictions = {}

    for category, model in models.items():
        sentiment = model.predict(new_text_vec)
        predictions[f"{model_type.capitalize()} ({category}) "] = sentiment[0]

    return predictions

# Fungsi untuk menampilkan hasil prediksi dalam bentuk tabel
def show_prediction_table(predictions):
    st.subheader("Hasil Prediksi Sentimen:")
    st.write(predictions)

# Fungsi untuk menampilkan diagram batang rangkuman prediksi
def show_summary_plot(predictions):
    st.subheader("Rangkuman Prediksi:")
    for category in ['daya_tarik', 'amenitas', 'aksesibilitas', 'citra', 'harga', 'sdm']:
        plt.figure(figsize=(8, 5))
        sns.countplot(x=f'{category}_sentiment', data=predictions, palette='viridis')
        plt.title(f'Rangkuman Prediksi {category.capitalize()}')
        plt.xlabel('Sentiment')
        plt.ylabel('Jumlah')
        st.pyplot()


# WordCloud untuk LDA (Per Aspek)
def generate_wordcloud(sentiment_column, display_positive=True, display_negative=True):
    if display_negative:
        text_neg = ' '.join(data_clean[data_clean[sentiment_column] == 'Negatif']['text'])  # Negative sentiment
        st.subheader(f"Word Cloud for Negative Sentiment in {sentiment_column}")
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(text_neg)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

    if display_positive:
        text_pos = ' '.join(data_clean[data_clean[sentiment_column] == 'Positif']['text'])  # Positive sentiment
        st.subheader(f"Word Cloud for Positive Sentiment in {sentiment_column}")
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(text_pos)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

# WordCloud untuk LDA (Semua Aspek)
def generate_combined_wordcloud(display_positive=True, display_negative=True):
    text = ' '.join(data_clean['text'])
    st.subheader("Word Cloud of Topic Modeling")
    
    if display_negative:
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.subheader("Negative Sentiment")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_neg, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

    if display_positive:
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.subheader("Positive Sentiment")
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud_pos, interpolation='bilinear')
        plt.axis('off')
        st.pyplot()

mapping_model = {
    'Logistic Regression': 'lr_models',
    'Naive Bayes': 'nb_models',
    'SVM': 'svm_models',
    'Random Forest': 'rf_models',
    'Decision Tree': 'dt_models'    
}


# Streamlit App
#selected_tab = st.sidebar.radio("Choose a tab", ['Business Case','Model Performance', 'Predict Sentimen', 'Topic Modeling', 'Dataset', 'About Me'])

with st.sidebar:
    selected_tab = option_menu(
        menu_title = "Main Menu",
        options=['Business Case','Model Performance', 'Predict Sentimen', 'Topic Modeling', 'Dataset', 'About Me'],
        icons=["house", "file-bar-graph", "filter-circle", "diagram-3", "database", "person"],
        menu_icon="cast",  # optional
        default_index=0,  # optional
        styles={

                "icon": {"color": "orange"},
                "nav-link": {
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )

if selected_tab == 'Business Case':
    st.subheader("Exploring Tourist Reviews: Aspect-Based Sentiment Analysis and Topic Modeling for Destination Enhancement")
    st.caption("Analisis Sentimen Berbasis Aspek dan Pemodelan Topik Untuk Peningkatan Pelayanan Destinasi Pariwisata")
    tab_background, tab_problem, tab_data = st.tabs(["Background", "Problem Scope", "Data Understanding"])
    with tab_background:
        st.subheader("Project Idea")
        st.markdown(""" <div style="text-align: justify">
                    <b style="color:green;">Evaluasi kualitas layanan</b> pariwisata di candi borobudur dapat dilakukan dengan <b style="color:green;">memanfaatkan ulasan internet</b>. Oleh karena itu, pada project ini akan dikembangkan sebuah model <b style="color:green;">analisis sentimen berbasis aspek</b> periwisata untuk <b style="color:green;">evaluasi pelayanan</b> serta <b style="color:green;">topik modeling</b> untuk <b style="color:green;">identifikasi topik-topik</b> terkait aspek pelayanan tersebut.
                    </div>
                    """, unsafe_allow_html=True)
        # Menampilkan gambar di aplikasi Streamlit
        st.image(gambar, caption='Mindmap', use_column_width=True)    
          
        st.header("WordCloud")
        # Menampilkan gambar menggunakan st.image
        fig
        
    with tab_problem:
        st.subheader("Problem Scope")
        st.markdown(""" <div style="text-align: justify">
                    Pada project ini akan digunakan Data ulasan yang berasal dari hasil scraping pada platform google maps terkait destinasi wisata candi borobudur. Selanjutnya, dataset tersebut akan di latih dengan beberapa model machine learning untuk mencari model dengan preforma terbaik. Model dengan performa terbaik akan digunakan untuk membuat analisis sentimen berbasis aspek. Kemudian dilakukan topic modeling dengan LDA pada masing-masing aspek untuk menemukan topik-topik yang berpengaruh pada aspek tersebut.
                    </div>
                    """, unsafe_allow_html=True)

        st.subheader("Business Impact")
        st.markdown("""
        <div style="text-align: justify">
        <ol>
        <li> Peningkatan Pelayanan: Hasil proyek ini dapat digunakan sebagai referensi untuk meningkatkan kualitas pelayanan di sekitar objek pariwisata. Dengan menganalisis ulasan dari pengunjung, pengelola dapat mengidentifikasi aspek-aspek yang perlu diperbaiki dalam layanan mereka. Ini dapat mencakup masalah seperti keramahan staf, kebersihan, fasilitas, dan lainnya. Dengan informasi ini, pengelola dapat melakukan perbaikan yang lebih terarah.</li>
        <li> Informasi yang Lebih Mendalam: Analisis ulasan yang mencakup aspek dan topik terkait dapat membantu pengelola untuk menggali informasi yang lebih mendalam tentang apa yang perlu ditingkatkan dalam layanan mereka. Ini tidak hanya mencakup masalah umum, tetapi juga detail-detail yang mungkin terlewatkan tanpa tinjauan yang cermat.</li>
        <li> Efisiensi Anggaran: Proyek ini juga dapat membantu pengelola menghemat anggaran yang biasanya digunakan untuk mengukur kepuasan pelanggan secara tradisional. Metode tradisional seringkali melibatkan survei yang mahal dan sumber daya lainnya. Dengan menggunakan ulasan online yang sudah ada, pengelola dapat mendapatkan wawasan yang sama atau lebih baik tanpa biaya tambahan.</li>
        <li> Peningkatan Reputasi: Dengan mengambil tindakan yang sesuai berdasarkan ulasan, pengelola dapat meningkatkan reputasi destinasi mereka. Ulasan positif dari pengunjung akan menarik lebih banyak pelanggan potensial, sementara perbaikan yang dilakukan akan mengurangi keluhan dan ulasan negatif.</li>
        <li> Pengambilan Keputusan yang Lebih Baik: Informasi yang diberikan oleh hasil proyek ini dapat menjadi panduan berharga untuk pengelola destinasi dalam mengambil keputusan terkait perbaikan layanan, pengembangan fasilitas, atau strategi pemasaran.</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)


    with tab_data:
        st.subheader("Dataset")        
        st.markdown("""
        <div style="text-align: justify">
        Data hasil scraping kemudian di anotasi dengan oleh tiga orang anotator berbeda agar penilaian sentimen tidak subjektif. Metode penentuan label akhir yaitu apakah positif, negatif, netral untuk setiap aspek dilakukan dengan voting nilai terbanyak.</li>
        Dataset yang telah di anotasi mengandung informasi sebagai berikut:
            <ol>
            <li>Nomor: Urutan ulasan</li>
            <li>Text: Ulasan dalam bahasa Indonesia</li>
            <li>DayaTarik: Aspek ini dinilai berdasarkan komentar tentang daya tarik lingkungan candi, termasuk objek candi, arsitektur, sejarah, keindahan alam, taman, dan berbagai acara seperti Prambanan Jazz Festival, Borobudur Marathon, atau festival budaya.</li>
            <li>Amenitas: Aspek ini dievaluasi berdasarkan kenyamanan pengunjung di lingkungan candi, termasuk ketersediaan air, pepohonan, akses telekomunikasi, fasilitas sanitasi, tempat sampah, pos keamanan, parkir, tempat ibadah, penginapan, restoran, transportasi, fasilitas kesehatan, penyewaan sepeda, toko oleh-oleh, informasi pariwisata, dan penyewaan peralatan.</li>
            <li>Aksesibilitas: Aspek ini berfokus pada aksesibilitas menuju dan di dalam candi. Melibatkan komentar terkait infrastruktur jalan, bandara, jalur kereta api, jarak, waktu tempuh, pintu keluar, jalur khusus untuk disabilitas dan orang tua di dalam candi, transportasi umum, dan kebijakan hewan peliharaan di dalam candi.</li>
            <li>Citra: Aspek ini dinilai berdasarkan citra yang terbentuk di sekitar candi, termasuk komentar tentang kebersihan lingkungan, perilaku ramah dari pengunjung dan pedagang, serta suasana dan cuaca di sekitar candi.</li>
            <li>Harga: Aspek ini dinilai berdasarkan komentar terkait aspek finansial kunjungan ke candi, melibatkan harga tiket masuk, biaya transportasi, tarif pemandu, akomodasi, makanan, aktivitas di sekitar candi, dan biaya parkir.</li>
            <li>SDM: Aspek ini dievaluasi berdasarkan komentar terkait kemampuan, pelayanan, dan keramahan staf candi, termasuk petugas tiket masuk, keamanan, kebersihan, pemandu wisata, dan manajemen. Komentar dapat mencakup keterampilan bahasa, jumlah staf, respons terhadap pertanyaan atau keluhan pengunjung, serta keramahan dan kualitas keahlian staf.</li>
        </ol>
        Dataset disadur dari <a href="https://github.com/dian9395/dataset-analisis-sentimen-berbasis-aspek-dan-pemodelan-topik">GitHub</a> dengan jurnal <a href="https://jurnal.pnj.ac.id/index.php/multinetics/article/view/5056">MultiNetics</a>.</li>
        </div>
        """, unsafe_allow_html=True)

elif selected_tab == 'Model Performance':
    # Streamlit App
    st.header("Model Evaluation")
    # Sidebar
    st.text("Select View")
    table_data = []
    for model, aspek_dict in all_evaluations.items():
        for aspek, metrics in aspek_dict.items():
            table_data.append({
                'Model': model,
                'Aspek': aspek,
                'Accuracy': metrics['accuracy']
            })

        df_table = pd.DataFrame(table_data)

    tab1, tab2 = st.tabs(["Tabel", "Plot"])

    # Tampilan Tabel
    with tab1:
       # Create DataFrame for table
       # Display table
        st.table(df_table.pivot_table(index='Model', columns='Aspek', values='Accuracy', aggfunc='mean', margins=True, margins_name='Total'))

    with tab2:
        model_list = list(all_evaluations.keys())
        selected_model = st.selectbox("Pilih Model", ["Semua Model"] + model_list)

        aspek_list = list(all_evaluations[model_list[0]].keys())
        selected_aspek = st.selectbox("Pilih Aspek", ["Semua Aspek"] + aspek_list)

        # Filter data based on user selection
        if selected_model == "Semua Model":
            models_to_plot = model_list
        else:
            models_to_plot = [selected_model]

        if selected_aspek == "Semua Aspek":
            aspek_to_plot = aspek_list
        else:
            aspek_to_plot = [selected_aspek]

        # Create a DataFrame for seaborn
        data = {'Model': [], 'Aspek': [], 'Metric': [], 'Value': []}
        for model in models_to_plot:
            for aspek in aspek_to_plot:
                for metric, value in all_evaluations[model][aspek].items():
                    data['Model'].append(model)
                    data['Aspek'].append(aspek)
                    data['Metric'].append(metric)
                    data['Value'].append(value)

        df = pd.DataFrame(data)

        # Plot using seaborn
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Metric', y='Value', hue='Model', data=df, ci=None)
        plt.title(f"Evaluasi Model untuk {', '.join(aspek_to_plot)}")
        plt.xlabel("Metric Evaluasi")
        plt.ylabel("Nilai Evaluasi")
        st.pyplot(plt)

    # Kesimpulan
    st.subheader("Kesimpulan:")
    # 1. Model terbaik berdasarkan accuracy
    st.write("1. Model terbaik berdasarkan accuracy adalah: Logistic Regression")
    st.write("2. Model terbaik untuk mengukur aspek Aksesibilitas: Logistic Regression")
    st.write("3. Model terbaik untuk mengukur aspek Amenitas: Logistic Regression")
    st.write("4. Model terbaik untuk mengukur aspek Citra: SVM")
    st.write("5. Model terbaik untuk mengukur aspek Daya Tarik: Logistic Regression")
    st.write("6. Model terbaik untuk mengukur aspek Harga: Logistic Regression")
    st.write("7. Model terbaik untuk mengukur aspek SDM: SVM")


elif selected_tab == 'Predict Sentimen':
    st.header("Predict Sentiment")

    # Pilihan yang ditampilkan dalam dropdown
    options_display = list(mapping_model.keys())

    # Pilihan yang diterima oleh selectbox
    options_value = list(mapping_model.values())

    # Selectbox
    model_type = st.selectbox("Pilih Model Machine Learning:", options_display)

    # Mendapatkan nilai yang sesuai dengan pilihan yang dipilih
    selected_value = mapping_model[model_type]

    # Input teks
    tab_comment, tab_csv = st.tabs(["Write Comment", "Upload CSV"])
    with tab_comment:
        new_text = st.text_area("Masukkan kalimat untuk diprediksi sentimennya:")

        # Tombol untuk melakukan prediksi
        if st.button("Prediksi Sentimen"):
        
            if new_text:
                predictions = predict_sentiment(new_text, model_type=selected_value)
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen:")
                for model_category, sentiment in predictions.items():
                    st.write(f"{model_category}: {sentiment}")
            else:
                st.warning("Masukkan kalimat terlebih dahulu.")
    with tab_csv:
        # Input file CSV
        uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])

        # Tombol untuk melakukan prediksi
        if st.button("Prediksi File") and uploaded_file:
        # Membaca file CSV
            df_new = pd.read_csv(uploaded_file)
            # Mengonversi teks menggunakan CountVectorizer
            X = vectorizer.transform(df_new['text'])

            # Membaca kembali model dari file pickle
            with open(f'{selected_value}.pickle', 'rb') as model_file:
                models = pickle.load(model_file)

            # Mengevaluasi model
            predictions = pd.DataFrame({'id': df_new['id'], 'text': df_new['text']})
            for category, model in models.items():
                sentiment = model.predict(X)
                predictions[f'{category}_sentiment'] = sentiment

            # Menampilkan hasil prediksi
            show_prediction_table(predictions)

            # Menampilkan diagram batang rangkuman prediksi
            show_summary_plot(predictions)



elif selected_tab == 'Topic Modeling':
    def main():
        st.header("Topic Modeling dengan LDA")

        # Select option
        selected_aspect = st.selectbox("Select Aspect", ('Semua Aspek', 'Daya Tarik', 'Amenitas', 'Aksesibilitas', 'Citra', 'Harga', 'SDM'))
        
        if selected_aspect == 'Semua Aspek':
            # Display positive and/or negative sentiment word clouds
            display_negative = st.checkbox("Display Negative Sentiment", value=True)
            display_positive = st.checkbox("Display Positive Sentiment", value=True)
        
            # Generate and display word cloud for combined text
            generate_combined_wordcloud(display_positive, display_negative)
        
        else:
        
            # Display positive and/or negative sentiment word clouds
            display_negative = st.checkbox("Display Negative Sentiment", value=True)
            display_positive = st.checkbox("Display Positive Sentiment", value=True)

        
            # Generate and display word cloud
            generate_wordcloud(selected_aspect, display_positive, display_negative)

    if __name__ == "__main__":
        main()

elif selected_tab == 'Dataset':
    # Create a DataFrame
    data_tampil = pd.DataFrame(data_raw)

    # Streamlit App
    st.header('Dataset Candi Borobudur')

    # Pilihan aspek
    selected_aspek = st.selectbox('Pilih Aspek:', ['Semua Aspek'] + list(data_tampil.columns[2:]))

    # Pilihan sentimen
    selected_sentimen = st.selectbox('Pilih Sentimen:', ['Semua Sentimen', 'Positif', 'Negatif'])

    # Filter DataFrame berdasarkan pilihan
    if selected_aspek != 'Semua Aspek':
        data_tampil = data_tampil[['lokasi', 'text', selected_aspek]]

    if selected_sentimen != 'Semua Sentimen':
        if selected_aspek == 'Semua Aspek':
            data_tampil = data_tampil[data_tampil.apply(lambda row: row.str.contains(selected_sentimen, case=False) if row.name in data_tampil.columns[2:] else True, axis=1)]
        else:
            data_tampil = data_tampil[data_tampil[selected_aspek] == selected_sentimen]

    # Tampilkan DataFrame
    st.dataframe(data_tampil)

elif selected_tab == 'About Me':
    # Judul halaman
    st.header("About Me")

    st.write("- **Nama Lengkap:** Jayadi Butar Butar")
    st.write("- **Alamat:** Jakarta, Indonesia")
    st.write("- **Email:** jayadidetormentor@gmail.com")

    # Summary
    st.subheader("Summary")
    st.markdown(""" <div style="text-align: justify">    
    I'm a motivated Data Professional with a strong background in scientific research, Data Science, and Machine Learning. 
    Proficient in Python, Rstudio, SQL, and Spreadsheets, I excel in qualitative and quantitative research. 
    My dynamic academic journey honed my strategic thinking, leadership, and problem-solving skills.
    I derive valuable insights from complex datasets and excel in presenting them for data-driven decision-making. 
    Passionate about Statistics, Data Science, and AI, I aim to make a significant impact in the world of data.
    Eager to learn and grow, I stay updated with the latest industry advancements through active training and self-directed learning. 
    My goal is to leverage my skills in data analysis and machine learning to solve complex problems and achieve meaningful outcomes.
    If you're looking for a dedicated and analytical team player thriving in a data-driven environment, let's connect for outstanding results.
    """, unsafe_allow_html=True)


    # Tautan ke Akun Sosial Media
    st.subheader("Projects Portofolio")
    st.write("- [LinkedIn](https://www.linkedin.com/in/jayadib/)")
    st.write("- [rPubs](https://rpubs.com/JayadiB/)")
    st.write("- [GitHub](https://github.com/Jay4di)")
