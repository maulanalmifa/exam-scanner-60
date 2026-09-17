import cv2
import numpy as np
import streamlit as st
from PIL import Image

def order_points(pts):
    """
    Mengurutkan 4 titik koordinat dengan urutan: 
    Kiri-Atas, Kanan-Atas, Kanan-Bawah, Kiri-Bawah.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Kiri-Atas memiliki jumlah koordinat (x+y) terkecil
    # Kanan-Bawah memiliki jumlah koordinat (x+y) terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Kanan-Atas memiliki selisih (y-x) terkecil
    # Kiri-Bawah memiliki selisih (y-x) terbesar
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def four_point_transform(image, pts):
    """
    Melakukan 'Warp Perspective' untuk meratakan gambar berdasarkan 4 titik.
    """
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # Hitung lebar gambar baru
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # Hitung tinggi gambar baru
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Tentukan koordinat tujuan untuk meratakan gambar
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # Hitung matriks transformasi dan terapkan ke gambar
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped

def detect_and_warp(image_array):
    """
    Mendeteksi 4 marker sudut dan meratakan lembar jawaban.
    """
    # 1. Konversi ke Grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # 2. Gaussian Blur untuk mengurangi noise (bayangan/tekstur kertas)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Edge Detection (Deteksi Tepi)
    edged = cv2.Canny(blurred, 75, 200)

    # 4. Temukan Kontur
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. Cari 4 marker berdasarkan bentuk persegi dan luasnya
    markers = []
    for c in contours:
        # Hitung keliling dan aproksimasi bentuk poligon
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Marker harus berupa persegi (4 titik) dan cukup besar
        if len(approx) == 4 and cv2.contourArea(c) > 500:
            # Hitung bounding box untuk memastikan bentuknya mirip persegi (rasio aspek ~1)
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            if 0.8 <= aspect_ratio <= 1.2:
                # Simpan titik tengah dari marker tersebut
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    markers.append([cX, cY])

    # Jika sistem menemukan tepat 4 marker, lakukan transformasi
    if len(markers) == 4:
        pts = np.array(markers, dtype="float32")
        warped_image = four_point_transform(image_array, pts)
        
        # Gambar titik merah di posisi marker pada gambar asli untuk visualisasi debug
        debug_image = image_array.copy()
        for marker in markers:
            cv2.circle(debug_image, tuple(marker), 15, (255, 0, 0), -1)
            
        return debug_image, warped_image, True
    else:
        return image_array, None, False
        
def process_answers(warped_img):
    """
    Mengekstrak bulatan jawaban secara berurutan dan membaca pilihan siswa.
    """
    # 1. Standarisasi Ukuran (Proporsi A4)
    warped_img = cv2.resize(warped_img, (800, 1131))
    
    # 2. Binarization
    gray = cv2.cvtColor(warped_img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # 3. Potong ROI Area Jawaban (40% dari atas sampai 92% ke bawah)
    h_img = warped_img.shape[0]
    y_start = int(h_img * 0.40)
    y_end = int(h_img * 0.92) 
    
    answer_roi_thresh = thresh[y_start:y_end, :]
    answer_roi_color = warped_img[y_start:y_end, :].copy()
    
    # 4. Cari kontur di area jawaban
    contours, _ = cv2.findContours(answer_roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bubbles = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(c)
        
        # Filter ketat: Hanya ambil bulatan OMR yang valid
        if 15 <= w <= 32 and 15 <= h <= 32 and 0.8 <= aspect_ratio <= 1.2:
            if 150 <= area <= 900:
                bubbles.append(c)
                
    # Validasi: Wajib menemukan tepat 200 bulatan (40 soal x 5 opsi)
    if len(bubbles) != 300:
        cv2.drawContours(answer_roi_color, bubbles, -1, (255, 0, 0), 2)
        return answer_roi_thresh, answer_roi_color, False, f"Gagal mengekstrak. Terdeteksi {len(bubbles)} bulatan, seharusnya 200."
        
    # --- LOGIKA SORTING (MENGURUTKAN BULATAN) ---
    # Urutkan seluruh bulatan dari kiri ke kanan berdasarkan sumbu X
    bubbles = sorted(bubbles, key=lambda b: cv2.boundingRect(b)[0])
    
    # Pecah menjadi 4 kolom berurutan (karena 200 bulatan / 4 kolom = 50 per kolom)
    columns = [bubbles[0:75], bubbles[75:150], bubbles[150:225], bubbles[225:300]]
    
    student_answers = []
    
    for col_idx, col_bubbles in enumerate(columns):
        # Dalam satu kolom, urutkan dari atas ke bawah berdasarkan sumbu Y
        col_bubbles = sorted(col_bubbles, key=lambda b: cv2.boundingRect(b)[1])
        
        # Iterasi per baris soal (kelompokkan 5 bulatan)
        for i in range(0, 75, 5):
            question_bubbles = col_bubbles[i:i+5]
            # Urutkan opsi A, B, C, D, E dari kiri ke kanan (sumbu X)
            question_bubbles = sorted(question_bubbles, key=lambda b: cv2.boundingRect(b)[0])

            # --- GANTI BAGIAN INI ---
            pixels_count = []
            for bubble in question_bubbles:
                # Dapatkan koordinat kotak yang mengelilingi bulatan
                (x, y, w, h) = cv2.boundingRect(bubble)
                
                # PANGKAS (SHRINK) 3 piksel dari setiap sisi untuk membuang garis tepi lingkaran
                shrink = 3
                inner_roi = answer_roi_thresh[y+shrink : y+h-shrink, x+shrink : x+w-shrink]
                
                # Hitung total piksel putih (tinta) HANYA di area tengah
                total_pixels = cv2.countNonZero(inner_roi)
                
                # Hitung kepadatan (Persentase area tengah yang tertutup tinta)
                area_tengah = inner_roi.shape[0] * inner_roi.shape[1]
                
                if area_tengah > 0:
                    kepadatan = total_pixels / area_tengah
                else:
                    kepadatan = 0
                    
                pixels_count.append(kepadatan)
                
            # Ambang batas minimal 40% area TENGAH harus tertutup tinta
            MIN_KEPADATAN = 0.40  
            
            # Cari bulatan dengan kepadatan tinta tertinggi
            max_kepadatan = max(pixels_count)
            
            # Tentukan Nomor Soal 
            soal_no = (col_idx * 15) + (i // 5) + 1
            
            # Evaluasi Jawaban
            if max_kepadatan > MIN_KEPADATAN:
                # Jika kepadatan melebihi 40%, berarti valid dijawab
                answered_index = pixels_count.index(max_kepadatan)
                jawaban = ['A', 'B', 'C', 'D', 'E'][answered_index]
                
                # Lingkari jawaban yang terpilih dengan warna hijau
                cv2.drawContours(answer_roi_color, [question_bubbles[answered_index]], -1, (0, 255, 0), 2)
            else:
                # Jika tengahnya kosong melompong (kepadatan < 40%), berarti kosong
                jawaban = "-"
                
            # MASUKKAN DATA
            student_answers.append((soal_no, jawaban))
            # ------------------------
            
    # Pastikan hasil akhir diurutkan murni berdasarkan Nomor Soal 1-40
    student_answers = sorted(student_answers, key=lambda x: x[0])
    
    return answer_roi_thresh, answer_roi_color, True, student_answers   

# --- ANTARMUKA STREAMLIT ---
st.set_page_config(layout="wide")
st.title("Sistem Koreksi LJK Otomatis (60 Soal)")

# ================= SIDEBAR: KUNCI JAWABAN =================
st.sidebar.header("⚙️ Konfigurasi")
csv_file = st.sidebar.file_uploader("1. Unggah CSV Kunci Jawaban", type=["csv"])

# --- SIDEBAR : TEMPLATE ---
st.sidebar.subheader("📥 Download Template")
st.sidebar.write("Belum punya formatnya? Unduh template LJK dan Kunci Jawaban di bawah ini:")

# Menggunakan format tombol link (lebih rapi dan modern)
st.sidebar.link_button("📄 Template LJK (HTML)", "https://github.com/maulanalmifa/exam-scanner-60/blob/5dc98daf0153bc3b19c99bd0e27b2d34599ec4d8/template-ljk-60.html", use_container_width=True)
st.sidebar.link_button("📊 Template Kunci (CSV)", "https://github.com/maulanalmifa/exam-scanner-60/blob/5dc98daf0153bc3b19c99bd0e27b2d34599ec4d8/kunci.csv", use_container_width=True)

# ================= SIDEBAR: BOBOT NILAI ==============
st.sidebar.markdown("---")
st.sidebar.header("⚖️ Pengaturan Bobot Nilai")
bobot_benar = st.sidebar.number_input("Poin jika BENAR", value=1.0, step=0.5, format="%.1f")
bobot_salah = st.sidebar.number_input("Poin jika SALAH", value=0.0, step=0.5, format="%.1f")
bobot_kosong = st.sidebar.number_input("Poin jika KOSONG", value=0.0, step=0.5, format="%.1f")
st.sidebar.info("💡 Tip: Gunakan angka minus (misal -1) pada kolom SALAH untuk menerapkan sistem penalti UTBK/SNBT.")
# ------------------------------------------------

kunci_df = None
if csv_file is not None:
    import pandas as pd
    kunci_df = pd.read_csv(csv_file)
    # Standarisasi nama kolom untuk berjaga-jaga
    kunci_df.columns = ["Nomor Soal", "Kunci Jawaban"]
    
    st.sidebar.success("Kunci Jawaban Berhasil Dimuat!")
    with st.sidebar.expander("Lihat Kunci Jawaban"):
        st.dataframe(kunci_df, hide_index=True)
else:
    st.sidebar.warning("⚠️ Silakan unggah file CSV Kunci Jawaban terlebih dahulu.")

# ================= AREA UTAMA: INPUT LJK =================
st.write("2. Posisikan LJK di tengah layar, pastikan pencahayaan cukup.")
tab1, tab2 = st.tabs(["📸 Kamera", "📁 Unggah File"])
image_data = None

with tab1:
    camera_file = st.camera_input("Ambil Foto LJK")
    if camera_file is not None:
        image_data = camera_file

with tab2:
    uploaded_file = st.file_uploader("Atau pilih file LJK dari perangkat...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_data = uploaded_file

# ================= PROSES EVALUASI =================
if image_data is not None:
    if kunci_df is None:
        st.error("Kunci Jawaban belum diunggah! Masukkan file CSV di bilah samping (Sidebar) kiri terlebih dahulu.")
    else:
        # Konversi gambar
        image_pil = Image.open(image_data)
        image_np = np.array(image_pil)
        
        # 1. Deteksi dan ratakan gambar
        debug_img, warped_img, success = detect_and_warp(image_np)
        
        if success:
            # 2. Ekstrak jawaban
            mask_img, result_img, ans_success, data = process_answers(warped_img)
            
            if ans_success:
                st.success("✅ Gambar berhasil diratakan dan jawaban berhasil diekstrak!")
                
                # Ubah ke Dataframe Pandas
                df_jawaban = pd.DataFrame(data, columns=["Nomor Soal", "Jawaban Siswa"])
                
                # 3. PROSES PENILAIAN (SCORING)
                # Gabungkan tabel jawaban siswa dengan tabel kunci jawaban
                hasil_df = pd.merge(df_jawaban, kunci_df, on="Nomor Soal")
                
                # Fungsi mengecek benar/salah
                def cek_status(row):
                    if str(row["Jawaban Siswa"]).strip().upper() == str(row["Kunci Jawaban"]).strip().upper():
                        return "Benar"
                    elif row["Jawaban Siswa"] == "-":
                        return "Kosong"
                    else:
                        return "Salah"
                
                hasil_df["Status"] = hasil_df.apply(cek_status, axis=1)
                
                # Kalkulasi Statistik
                total_soal = len(hasil_df)
                jumlah_benar = len(hasil_df[hasil_df["Status"] == "Benar"])
                jumlah_salah = len(hasil_df[hasil_df["Status"] == "Salah"])
                jumlah_kosong = len(hasil_df[hasil_df["Status"] == "Kosong"])
                
                # --- RUMUS SKOR DINAMIS BARU ---
                # Hitung akumulasi poin berdasarkan input di sidebar
                skor_akhir = (jumlah_benar * bobot_benar) + (jumlah_salah * bobot_salah) + (jumlah_kosong * bobot_kosong)
                skor_maksimal = total_soal * bobot_benar # Asumsi poin maksimal didapat jika benar semua

                st.divider()
                
                # Menampilkan visual dan detail analisis
                col_vis, col_tabel = st.columns([1, 1])
                
                with col_vis:
                    st.write("🔍 Visualisasi Pembacaan")
                    st.image(result_img, use_container_width=True)
                    
                # 4. TAMPILKAN HASIL
                st.divider()
                st.subheader("🏆 Ringkasan Nilai Siswa")
                
                # Menampilkan Score Card
                col_score1, col_score2, col_score3, col_score4 = st.columns(4)
                col_score1.metric("SKOR AKHIR", f"{skor_akhir:.2f}")
                col_score2.metric("Benar", jumlah_benar)
                col_score3.metric("Salah", jumlah_salah)
                col_score4.metric("Kosong", jumlah_kosong)
                    
                with col_tabel:
                    st.write("📊 Detail Analisis Butir Soal")
                    # Styling dataframe agar warna barisnya hijau kalau benar, merah kalau salah
                    def color_status(val):
                        if val == 'Benar':
                            color = 'rgba(0, 255, 0, 0.2)'
                        elif val == 'Salah':
                            color = 'rgba(255, 0, 0, 0.2)'
                        else:
                            color = 'rgba(255, 255, 0, 0.2)'
                        return f'background-color: {color}'

                    styled_df = hasil_df.style.map(color_status, subset=['Status'])
                    st.dataframe(styled_df, hide_index=True, height=500)
                    
            else:
                st.error(data) # Tampilkan pesan gagal ekstraksi jawaban
        else:
            st.error("❌ Gagal menemukan 4 marker. Pastikan foto tidak terpotong dan pencahayaan terang.")
