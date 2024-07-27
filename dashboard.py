import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# Fungsi untuk memuat data
def load_data():
    day_df = pd.read_csv("day.csv")
    hour_df = pd.read_csv("hour.csv")
    
    day_df['dteday'] = pd.to_datetime(day_df['dteday'])
    hour_df['dteday'] = pd.to_datetime(hour_df['dteday'])
    
    day_df.drop_duplicates(inplace=True)
    hour_df.drop_duplicates(inplace=True)
    
    return day_df, hour_df

# Memuat data
day_df, hour_df = load_data()

# Tampilan informasi data
st.title("Analisis Penyewaan Sepeda")

st.subheader("Informasi Data Harian")
st.write(day_df.info())
st.write(day_df.describe())

st.subheader("Informasi Data Per Jam")
st.write(hour_df.info())
st.write(hour_df.describe())

# Visualisasi data
st.subheader("Visualisasi Distribusi Jumlah Penyewaan Sepeda")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(day_df['cnt'], bins=30, kde=True, color='blue', ax=ax)
ax.set_title('Distribusi Jumlah Penyewaan Sepeda')
ax.set_xlabel('Jumlah Penyewaan')
ax.set_ylabel('Frekuensi')
st.pyplot(fig)

st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Musim")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='season', y='cnt', data=day_df, palette='Set2', ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Musim')
ax.set_xlabel('Musim')
ax.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig)

st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Hari dalam Seminggu")
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='weekday', y='cnt', data=day_df, palette='coolwarm', ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Hari dalam Seminggu')
ax.set_xlabel('Hari dalam Seminggu')
ax.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig)

st.subheader("Pengaruh Suhu terhadap Jumlah Penyewaan Sepeda")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x='temp', y='cnt', data=day_df, color='red', ax=ax)
ax.set_title('Pengaruh Suhu terhadap Jumlah Penyewaan Sepeda')
ax.set_xlabel('Suhu (°C)')
ax.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig)

st.subheader("Jumlah Penyewaan Sepeda Berdasarkan Hari Kerja dan Hari Libur")
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='workingday', y='cnt', data=day_df, palette='Paired', ax=ax)
ax.set_title('Jumlah Penyewaan Sepeda Berdasarkan Hari Kerja dan Hari Libur')
ax.set_xlabel('Hari Kerja (0 = Tidak, 1 = Ya)')
ax.set_ylabel('Jumlah Penyewaan')
st.pyplot(fig)

st.subheader("Pairplot untuk Fitur Terkait Cuaca")

pairplot = sns.pairplot(hour_df, vars=['temp', 'atemp', 'hum', 'windspeed', 'cnt'], kind='reg', plot_kws={'line_kws':{'color':'red'}})
plt.figure(figsize=(10, 6))
pairplot.fig.suptitle("Pairplot untuk Fitur Terkait Cuaca", y=1.02)  # Adjust title position
st.pyplot(pairplot.fig)


# Matriks korelasi dan heatmap
correlation_matrix = hour_df[['temp', 'atemp', 'hum', 'windspeed', 'cnt']].corr()
st.subheader("Matriks Korelasi untuk Faktor Cuaca dan Penyewaan Sepeda")
st.write(correlation_matrix)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Matriks Korelasi untuk Faktor Cuaca dan Penyewaan Sepeda')
st.pyplot(fig)

# Analisis T-Statistik
st.subheader("Rata-rata Penyewaan Sepeda: Hari Kerja vs Hari Libur")
workingday_mean = day_df.groupby('workingday')['cnt'].mean().reset_index()

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='workingday', y='cnt', data=workingday_mean, ax=ax)
ax.set_title('Rata-rata Penyewaan Sepeda: Hari Kerja vs Hari Libur')
ax.set_xlabel('Hari Kerja')
ax.set_ylabel('Rata-rata Jumlah')
ax.set_xticklabels(['Hari Libur', 'Hari Kerja'])
st.pyplot(fig)

workingday_rentals = day_df[day_df['workingday'] == 1]['cnt']
holiday_rentals = day_df[day_df['workingday'] == 0]['cnt']
t_stat, p_val = ttest_ind(workingday_rentals, holiday_rentals)

st.write(f"T-Statistik: {t_stat}, P-Value: {p_val}")

# Kesimpulan
st.subheader("Kesimpulan")
st.write("""
### Conclution pertanyaan 1:

1. Suhu (temp dan atemp): Memiliki korelasi positif moderat dengan jumlah penyewaan sepeda (cnt). Ini berarti lebih banyak sepeda disewa pada suhu yang lebih tinggi.
2. Kelembaban (hum): Memiliki korelasi negatif dengan jumlah penyewaan sepeda. Ini berarti lebih sedikit sepeda disewa ketika kelembaban lebih tinggi.
3. Kecepatan Angin (windspeed): Memiliki korelasi yang sangat lemah dengan jumlah penyewaan sepeda, sehingga pengaruhnya bisa dianggap minimal.
Secara keseluruhan, faktor cuaca yang paling signifikan yang mempengaruhi jumlah penyewaan sepeda adalah suhu dan kelembaban.

### Conclution pertanyaan 2:

Tidak ada perbedaan yang signifikan secara statistik dalam jumlah penyewaan sepeda antara hari kerja dan hari libur. Meskipun terdapat perbedaan kecil dalam rata-rata jumlah penyewaan sepeda antara kedua hari tersebut, perbedaan tersebut tidak cukup besar untuk dianggap signifikan berdasarkan uji statistik yang dilakukan (dengan T-Statistik sebesar 1.6543 dan P-Value sebesar 0.0984).

Dengan kata lain, jumlah penyewaan sepeda cenderung konsisten antara hari kerja dan hari libur.
""")