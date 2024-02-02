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
st.set_option('deprecation.showPyplotGlobalUse', False)

# load clean dataset
data_clean = pd.read_pickle('data_clean.pickle')
data_raw = pd.read_pickle('data_raw.pickle')

# Membaca kembali objek CountVectorizer dari file
with open('count_vectorizer.pickle', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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
    st.markdown(""" ## Background
    Untuk meningkatkan kualitas pariwisata, pendapat dan ulasan pengunjung dianggap sebagai wawasan berharga untuk meningkatkan layanan. Dengan kemajuan teknologi informasi, ulasan pengunjung dapat ditemukan di internet dan platform media sosial. Namun, ada dua tantangan dalam memanfaatkan data ulasan online. Tantangan pertama adalah bagaimana cara mendapatkan sejumlah besar data ulasan yang tersebar di internet. Tantangan kedua adalah bagaimana mengolah data ulasan ini untuk mengambil wawasan yang berguna. Untuk masalah pertama, teknik web scraping adalah salah satu metode yang umum digunakan saat ini. Ini melibatkan ekstraksi data dari halaman web atau dokumen lain dengan tujuan mengambil informasi yang terstruktur atau tidak terstruktur dan menyimpannya dalam format yang dapat digunakan. Sementara itu, untuk tantangan kedua, metode berpikir kritis dapat digunakan oleh manusia untuk mengatasi masalah ini dengan membaca dan merangkum ulasan. Namun, ini menjadi tantangan baru ketika ulasan dihasilkan dengan cepat (velocity) dan dalam jumlah besar (volume). Ini berarti mengandalkan metode berpikir kritis saja mungkin tidak cukup. Selain itu, pertimbangan lainnya adalah bahwa teknik analisis manusia seringkali bersifat "subjektif" dan dapat berbeda tergantung pada sudut pandang penilai.
    ## Project Idea
    Evaluasi kualitas layanan pariwisata di candi borobudur dapat dilakukan dengan memanfaatkan ulasan internet. Oleh karena itu, pada project ini akan dikembangkan sebuah model analisis sentimen berbasis aspek periwisata untuk evaluasi pelayanan serta topik modeling untuk identifikasi topik-topik terkait aspek pelayanan tersebut.
    ## Problem Scope
    Pada project ini akan digunakan Data ulasan yang berasal dari hasil scraping pada platform google maps terkait destinasi wisata candi borobudur. Selanjutnya, dataset tersebut akan di latih dengan beberapa model machine learning untuk mencari model dengan preforma terbaik. Model dengan performa terbaik akan digunakan untuk membuat analisis sentimen berbasis aspek. Kemudian dilakukan topic modeling dengan LDA pada masing-masing aspek untuk menemukan topik-topik yang berpengaruh pada aspek tersebut.

    ## Dataset
    Data hasil scraping kemudian di anotasi dengan oleh tiga orang anotator berbeda agar penilaian sentimen tidak subjektif. Metode penentuan label akhir yaitu apakah positif, negatif, netral untuk setiap aspek dilakukan dengan voting nilai terbanyak. 
    Dataset yang telah di anotasi mengandung informasi sebagai berikut:
    - Nomor     : Urutan ulasan
    - Ulasan    : Ulasan dalam bahasa indonesia
    - DayaTarik : Aspek ini dinilai berdasarkan komentar tentang daya tarik lingkungan candi, termasuk objek candi, arsitektur, sejarah, keindahan alam, taman, dan berbagai acara seperti Prambanan Jazz Festival, Borobudur Marathon, atau festival budaya.
    - Amenitas  : Aspek ini dievaluasi berdasarkan kenyamanan pengunjung di lingkungan candi, termasuk ketersediaan air, pepohonan, akses telekomunikasi, fasilitas sanitasi, tempat sampah, pos keamanan, parkir, tempat ibadah, penginapan, restoran, transportasi, fasilitas kesehatan, penyewaan sepeda, toko oleh-oleh, informasi pariwisata, dan penyewaan peralatan
    - Aksesibilitas : Aspek ini berfokus pada aksesibilitas menuju dan di dalam candi. Melibatkan komentar terkait infrastruktur jalan, bandara, jalur kereta api, jarak, waktu tempuh, pintu keluar, jalur khusus untuk disabilitas dan orangtua di dalam candi, transportasi umum, dan kebijakan hewan peliharaan di dalam candi.
    - Citra     : Aspek ini dinilai berdasarkan citra yang terbentuk di sekitar candi, termasuk komentar tentang kebersihan lingkungan, perilaku ramah dari pengunjung dan pedagang, serta suasana dan cuaca di sekitar candi.
    - Harga     : Aspek ini dinilai berdasarkan komentar terkait aspek finansial kunjungan ke candi, melibatkan harga tiket masuk, biaya transportasi, tarif pemandu, akomodasi, makanan, aktivitas di sekitar candi, dan biaya parkir.
    - SDM   : Aspek ini dievaluasi berdasarkan komentar terkait kemampuan, pelayanan, dan keramahan staf candi, termasuk petugas tiket masuk, keamanan, kebersihan, pemandu wisata, dan manajemen. Komentar dapat mencakup keterampilan bahasa, jumlah staf, respons terhadap pertanyaan atau keluhan pengunjung, serta keramahan dan kualitas keahlian staf.
    
    Dataset disadur dari https://github.com/dian9395/dataset-analisis-sentimen-berbasis-aspek-dan-pemodelan-topik dengan jurnal https://jurnal.pnj.ac.id/index.php/multinetics/article/view/5056.
    ## Business Impact 
    Untuk Untuk pengelola destinasi wisata, hasil proyek ini dapat memiliki beberapa manfaat yang signifikan:
    1. Peningkatan Pelayanan: Hasil proyek ini dapat digunakan sebagai referensi untuk meningkatkan kualitas pelayanan di sekitar objek pariwisata. Dengan menganalisis ulasan dari pengunjung, pengelola dapat mengidentifikasi aspek-aspek yang perlu diperbaiki dalam layanan mereka. Ini dapat mencakup masalah seperti keramahan staf, kebersihan, fasilitas, dan lainnya. Dengan informasi ini, pengelola dapat melakukan perbaikan yang lebih terarah.
    2. Informasi yang Lebih Mendalam: Analisis ulasan yang mencakup aspek dan topik terkait dapat membantu pengelola untuk menggali informasi yang lebih mendalam tentang apa yang perlu ditingkatkan dalam layanan mereka. Ini tidak hanya mencakup masalah umum, tetapi juga detail-detail yang mungkin terlewatkan tanpa tinjauan yang cermat.
    3. Efisiensi Anggaran: Proyek ini juga dapat membantu pengelola menghemat anggaran yang biasanya digunakan untuk mengukur kepuasan pelanggan secara tradisional. Metode tradisional seringkali melibatkan survei yang mahal dan sumber daya lainnya. Dengan menggunakan ulasan online yang sudah ada, pengelola dapat mendapatkan wawasan yang sama atau lebih baik tanpa biaya tambahan.
    4. Peningkatan Reputasi: Dengan mengambil tindakan yang sesuai berdasarkan ulasan, pengelola dapat meningkatkan reputasi destinasi mereka. Ulasan positif dari pengunjung akan menarik lebih banyak pelanggan potensial, sementara perbaikan yang dilakukan akan mengurangi keluhan dan ulasan negatif.
    5. Pengambilan Keputusan yang Lebih Baik: Informasi yang diberikan oleh hasil proyek ini dapat menjadi panduan berharga untuk pengelola destinasi dalam mengambil keputusan terkait perbaikan layanan, pengembangan fasilitas, atau strategi pemasaran.
    
    Secara keseluruhan, hasil proyek ini dapat memberikan wawasan berharga kepada pengelola destinasi wisata, membantu mereka meningkatkan pengalaman pengunjung, menghemat biaya, dan meningkatkan reputasi destinasi mereka.
    """)
    st.header("WordCloud")
    # Menampilkan gambar menggunakan st.image
    st.image(image_stream.getvalue(), use_column_width=True)

elif selected_tab == 'Model Performance':
    # Streamlit App
    st.title("Model Evaluation Metrics")
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
    st.write("### Prediksi Sentiment ###")
    # Input teks
    # Select option
    #option_predict = st.radio("Select Option", ['Write Comment', 'Upload CSV'])
    tab_comment, tab_csv = st.tabs(["Write Comment", "Upload CSV"])
    with tab_comment:
        new_text = st.text_area("Masukkan kalimat untuk diprediksi sentimennya:")

        # Pilihan model
        model_type = st.selectbox("Pilih Model Machine Learning:", ['nb_models', 'svm_models', 'rf_models', 'dt_models', 'lr_models'])

        # Tombol untuk melakukan prediksi
        if st.button("Prediksi Sentimen"):
        
            if new_text:
                predictions = predict_sentiment(new_text, model_type=model_type)
                # Menampilkan hasil prediksi
                st.subheader("Hasil Prediksi Sentimen:")
                for model_category, sentiment in predictions.items():
                    st.write(f"{model_category}: {sentiment}")
            else:
                st.warning("Masukkan kalimat terlebih dahulu.")
    with tab_csv:
        # Input file CSV
        uploaded_file = st.file_uploader("Upload File CSV", type=["csv"])
        # Pilihan model
        model_type = st.selectbox("Pilih Model Machine Learning:", ['nb', 'svm', 'rf', 'dt', 'lr'])

        # Tombol untuk melakukan prediksi
        if st.button("Prediksi File") and uploaded_file:
        # Membaca file CSV
            df_new = pd.read_csv(uploaded_file)
            # Mengonversi teks menggunakan CountVectorizer
            X = vectorizer.transform(df_new['text'])

            # Membaca kembali model dari file pickle
            with open(f'{model_type}_models.pickle', 'rb') as model_file:
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
        st.title("Topic Modeling dengan LDA")

        # Select option
        option = st.radio("Select Option", ['All Aspects', 'Select Aspect'], horizontal =True)
        
        if option == 'Select Aspect':
            # Select aspect
            selected_aspect = st.selectbox("Select Aspect", data_clean.columns[2:])
        
            # Display positive and/or negative sentiment word clouds
            display_positive = st.checkbox("Display Positive Sentiment", value=True)
            display_negative = st.checkbox("Display Negative Sentiment", value=True)
        
            # Generate and display word cloud
            generate_wordcloud(selected_aspect, display_positive, display_negative)
        else:
            # Display positive and/or negative sentiment word clouds
            display_positive = st.checkbox("Display Positive Sentiment", value=True)
            display_negative = st.checkbox("Display Negative Sentiment", value=True)
        
            # Generate and display word cloud for combined text
            generate_combined_wordcloud(display_positive, display_negative)

    if __name__ == "__main__":
        main()

elif selected_tab == 'Dataset':
    # Create a DataFrame
    data_tampil = pd.DataFrame(data_raw)

    # Streamlit App
    st.title('Dataset Candi Borobudur')

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
    st.subheader("About Me")

    st.write("- **Nama Lengkap:** Jayadi Butar Butar")
    st.write("- **Alamat:** Jakarta, Indonesia")
    st.write("- **Email:** jayadidetormentor@gmail.com")

    # Summary
    st.subheader("Summary")
    st.write("""
    I'm a motivated Data Professional with a strong background in scientific research, Data Science, and Machine Learning. 
    Proficient in Python, Rstudio, SQL, and Spreadsheets, I excel in qualitative and quantitative research. 
    My dynamic academic journey honed my strategic thinking, leadership, and problem-solving skills.

    I derive valuable insights from complex datasets and excel in presenting them for data-driven decision-making. 
    Passionate about Statistics, Data Science, and AI, I aim to make a significant impact in the world of data.

    Eager to learn and grow, I stay updated with the latest industry advancements through active training and self-directed learning. 
    My goal is to leverage my skills in data analysis and machine learning to solve complex problems and achieve meaningful outcomes.
    
    If you're looking for a dedicated and analytical team player thriving in a data-driven environment, let's connect for outstanding results.
    """)


    # Tautan ke Akun Sosial Media
    st.subheader("Projects Portofolio")
    st.write("- [LinkedIn](https://www.linkedin.com/in/jayadib/)")
    st.write("- [rPubs](https://rpubs.com/JayadiB/)")
    st.write("- [GitHub](https://github.com/Jay4di)")
