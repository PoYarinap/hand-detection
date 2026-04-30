import cv2
import mediapipe as mp
import pyttsx3
import threading

# 1. Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# 2. Inisialisasi Text-to-Speech (Suara)
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Kecepatan bicara

def speak(text):
    """Fungsi untuk menjalankan suara di thread terpisah agar video tidak macet."""
    def run_speech():
        try:
            engine.say(text)
            engine.runAndWait()
        except:
            pass
    
    # Jalankan sebagai thread agar tidak memblokir loop utama kamera
    thread = threading.Thread(target=run_speech)
    thread.start()

# Variabel untuk melacak status suara agar tidak berulang terus-menerus
last_msg = ""

# 3. Jalankan Kamera
cap = cv2.VideoCapture(0)

print("Program berjalan... Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Gagal mengambil gambar dari kamera.")
        break

    # Balik gambar secara horizontal (mirror) dan konversi warna ke RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Proses deteksi tangan
    results = hands.process(image_rgb)

    message = ""
    fingers_status = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar titik-titik tangan di layar
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Daftar koordinat landmark
            lm = hand_landmarks.landmark

            # Logika Deteksi Jari Terbuka (Sederhana)
            # Jempol: Cek sumbu X
            if lm[4].x < lm[3].x: 
                fingers_status.append("Jempol")

            # Jari lain: Cek sumbu Y (Tip < PIP joint)
            if lm[8].y < lm[6].y: fingers_status.append("Telunjuk")
            if lm[12].y < lm[10].y: fingers_status.append("Tengah")
            if lm[16].y < lm[14].y: fingers_status.append("Manis")
            if lm[20].y < lm[18].y: fingers_status.append("Kelingking")

            # --- LOGIKA PESAN KHUSUS ---
            if fingers_status == ["Tengah"]:
                message = "Jari Tengah"
            elif len(fingers_status) == 5:
                message = "Halo Weslay"
            elif len(fingers_status) > 0:
                message = ", ".join(fingers_status)
            else:
                message = "Tangan Terdeteksi"

            # --- SUARA (TTS) ---
            if message != "" and message != last_msg:
                speak(message)
                last_msg = message

    # Tampilkan teks di layar
    cv2.putText(image, f"Status: {message}", (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Detection System', image)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
