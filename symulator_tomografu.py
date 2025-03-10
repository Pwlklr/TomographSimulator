import tkinter as tk
from tkinter import ttk, filedialog, Label, Button, Canvas, Checkbutton, IntVar, DoubleVar, StringVar, Entry, simpledialog
from PIL import Image, ImageTk
import numpy as np
from numpy.fft import fft, ifft, fftfreq
import time


class TomographSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Symulator Tomografu Komputerowego")

        # Ramka na wczytywanie obrazu i parametry
        self.control_frame = tk.Frame(root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        # Przycisk do wczytywania obrazu
        self.load_button = Button(self.control_frame, text="Wczytaj obraz", command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)

        # Pola parametryczne
        Label(self.control_frame, text="∆α (stopnie):").grid(row=0, column=1, padx=5, pady=5)
        self.delta_alpha_var = DoubleVar(value=1.0)
        self.delta_alpha_entry = Entry(self.control_frame, textvariable=self.delta_alpha_var, width=5)
        self.delta_alpha_entry.grid(row=0, column=2, padx=10, pady=10)

        Label(self.control_frame, text="Liczba detektorów:").grid(row=0, column=3, padx=5, pady=5)
        self.num_detectors_var = IntVar(value=250)
        self.num_detectors_entry = Entry(self.control_frame, textvariable=self.num_detectors_var, width=5)
        self.num_detectors_entry.grid(row=0, column=4, padx=5, pady=5)

        Label(self.control_frame, text="Rozpiętość (l):").grid(row=0, column=5, padx=5, pady=5)
        self.span_var = DoubleVar(value=180.0)
        self.span_entry = Entry(self.control_frame, textvariable=self.span_var, width=5)
        self.span_entry.grid(row=0, column=6, padx=5, pady=5)

        # Wybór filtra
        # Etykieta obok listy
        tk.Label(self.control_frame, text="Filtr:").grid(row=0, column=7, padx=5, pady=5)

        # Zmienna przechowująca wybraną wartość (string)
        self.filter_var = tk.StringVar()

        # Definiujemy Combobox z czterema opcjami
        self.filter_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.filter_var,
            values=["0 - brak", "1 - filtr z wykładu", "2 - ramp_filter", "3 - shepp_logan_filter"],
            state="readonly"
        )
        self.filter_combobox.grid(row=0, column=8, padx=5, pady=5)
        self.filter_combobox.current(0)  # Ustawiamy domyślnie pierwszą opcję ("0 - brak")

        # Przycisk pokazywania kroków pośrednich
        self.show_steps = IntVar(value=0)
        self.steps_check = Checkbutton(self.control_frame, text="Włącz pokazywanie kroków pośrednich", variable=self.show_steps)
        self.steps_check.grid(row=0, column=9, padx=5, pady=5)

        # Przycisk włączenia przyspieszonej odwrotnej transformaty
        self.accelerate = IntVar(value=0)
        self.accelerate_check = Checkbutton(self.control_frame, text="Przyspiesz obliczenie odwrotnej transformaty",
                                       variable=self.accelerate)
        self.accelerate_check.grid(row=0, column=10, padx=5, pady=5)

        # Przycisk startu generacji
        self.start_button = Button(self.control_frame, text="Generuj", command=self.generate)
        self.start_button.grid(row=0, column=11, padx=5, pady=5)

        # Ramka na wynik RMSE i zapis do DICOM
        self.result_frame = tk.Frame(root)
        self.result_frame.pack(after=self.control_frame, fill=tk.X, pady=5)

        # Pole do wyświetlania błędu średniokwadratowego
        Label(self.result_frame, text="Błąd rekonstrukcji:").grid(row=0, column=0, padx=5, pady=5)
        self.error_var = tk.StringVar()
        self.error_entry = Entry(self.result_frame, textvariable=self.error_var, state="readonly", width=10)
        self.error_entry.grid(row=0, column=1, padx=5, pady=5)

        # Przycisk generowania pliku DICOM
        self.start_button = Button(self.result_frame, text="Zapisz do pliku DICOM", command=self.load_to_dicom)
        self.start_button.grid(row=0, column=2, padx=5, pady=5)


        # Ramka do wyświetlania obrazów
        self.image_frame = tk.Frame(root)
        self.image_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # Canvas do wyświetlania oryginalnego obrazu
        self.original_picture_canvas = Canvas(self.image_frame, width=400, height=400, bg="black")
        self.original_picture_canvas.grid(row=0, column=0, padx=15, pady=10)
        Label(self.image_frame, text="Oryginalny obraz").grid(row=1, column=0)

        # Canvas do wyświetlania sinogramu
        self.sinogram_canvas = Canvas(self.image_frame, width=400, height=400, bg="black")
        self.sinogram_canvas.grid(row=0, column=1, padx=15, pady=10)
        Label(self.image_frame, text="Sinogram").grid(row=1, column=1)

        # Canvas do wyświetlania obrazu po odwrotnej transformacie\n
        self.reconstruction_canvas = Canvas(self.image_frame, width=400, height=400, bg="black")
        self.reconstruction_canvas.grid(row=0, column=2, padx=15, pady=10)
        Label(self.image_frame, text="Obraz po odwrotnej transformacie").grid(row=1, column=2)



        self.image = None
        self.image_array = None
        self.sinogram_image = None
        self.reconstructed_image = None

    def load_to_dicom(self):
        if not hasattr(self, 'reconstructed_array') or self.reconstructed_array is None:
            from tkinter import messagebox
            messagebox.showerror("Błąd", "Brak zrekonstruowanego obrazu do zapisu.")
            return
        else:
            import pydicom
            from pydicom.dataset import Dataset, FileDataset
            import datetime
            import os

            # Pobierz nazwę pliku do zapisu
            filename = filedialog.asksaveasfilename(
                defaultextension=".dcm",
                filetypes=[("DICOM files", "*.dcm")],
                title="Zapisz plik DICOM"
            )
            if not filename:
                return

            patientname = simpledialog.askstring("Input", "Wprowadź imię pacjenta",
                                            parent=root)
            patientID = simpledialog.askstring("Input", "Wprowadź ID pacjenta",
                                            parent=root)

            file_meta = pydicom.Dataset()
            file_meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
            file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
            file_meta.ImplementationClassUID = pydicom.uid.PYDICOM_IMPLEMENTATION_UID

            ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)
            dt = datetime.datetime.now()
            ds.ContentDate = dt.strftime("%Y%m%d")
            ds.ContentTime = dt.strftime("%H%M%S")
            ds.Modality = "OT"  # Other
            ds.PatientName = patientname
            ds.PatientID = patientID
            ds.StudyInstanceUID = pydicom.uid.generate_uid()
            ds.SeriesInstanceUID = pydicom.uid.generate_uid()
            ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
            ds.SOPClassUID = file_meta.MediaStorageSOPClassUID

            # Konwersja obrazu do uint8 (normalizacja)
            image_array = (self.reconstructed_array - self.reconstructed_array.min()) / \
                          (self.reconstructed_array.max() - self.reconstructed_array.min()) * 255
            image_array = image_array.astype(np.uint8)

            # Ustawiamy informacje o obrazie – używamy image_array, a nie self.reconstructed_array
            ds.Rows, ds.Columns = image_array.shape
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.SamplesPerPixel = 1
            ds.BitsAllocated = 8
            ds.BitsStored = 8
            ds.HighBit = 7
            ds.PixelRepresentation = 0
            ds.PixelData = image_array.tobytes()

            ds.save_as(filename, write_like_original=False)
            from tkinter import messagebox
            messagebox.showinfo("DICOM", f"Plik DICOM został zapisany jako:\n{os.path.basename(filename)}")

    def load_dicom_image(self, file_path):
        import pydicom
        #  Odczytanie pliku DICOM za pomocą pydicom
        dicom_data = pydicom.dcmread(file_path)

        # Pobranie danych pikseli z DICOM (zawierają one obraz)
        image_data = dicom_data.pixel_array

        # Przekształcenie danych pikseli na obraz w skali szarości ('L')
        image = Image.fromarray(image_data).convert("L")
        self.image_array = np.array(image)

        # Przeskalowanie obrazu, aby pasował do rozmiaru Canvas (zachowanie proporcji)
        max_canvas_size = 400
        img_width, img_height = image.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        display_image = image.resize(new_size)  # Zmieniony obraz do wyświetlania
        self.image = ImageTk.PhotoImage(display_image)

        # Usunięcie poprzedniego obrazu z Canvas i wyświetlenie nowego
        self.original_picture_canvas.delete("all")
        self.original_picture_canvas.create_image(200, 200, image=self.image, anchor=tk.CENTER)

        # Włączenie przycisku "Start"
        self.start_button.config(state=tk.NORMAL)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp;*.dcm")])
        if not file_path:
            return
        if file_path.endswith(".dcm"):
            self.load_dicom_image(file_path)
            return
        from PIL import Image
        image = Image.open(file_path).convert("L")  # Skala szarości
        self.image_array = np.array(image)
        self.image = ImageTk.PhotoImage(image)


        # żeby poprawnie wyświetlić obraz w Canvas jest on skalowany - jest to kopia niewykorzysywana do obliczeń
        max_canvas_size = 400
        img_width, img_height = image.size
        scale = min(max_canvas_size / img_width, max_canvas_size / img_height)
        new_size = (int(img_width * scale), int(img_height * scale))

        display_image = image.resize(new_size)  # Skopiowana wersja do wyświetlania
        self.image = ImageTk.PhotoImage(display_image)

        # Wyśrodkowanie obrazu na Canvas
        self.original_picture_canvas.delete("all")
        self.original_picture_canvas.create_image(200, 200, image=self.image, anchor=tk.CENTER)


        self.start_button.config(state=tk.NORMAL)


    def create_filter(self, kernel_size = 21):
        h = np.zeros(kernel_size)

        h[0] = 1
        for i in range(kernel_size):
            if i % 2:
                h[i] = (-4 / (np.pi ** 2)) / (i ** 2)
        return h

    def ramp_filter(self, proj):
        n = len(proj)
        # Transformacja do dziedziny częstotliwości
        proj_fft = fft(proj)
        # Tablica częstotliwości
        freqs = fftfreq(n)  # wartości od -0.5.. do 0.5

        # Ram-Lak = |ω| (tutaj normalizujemy tak, by max częstotliwość = 1)
        filter_vals = np.abs(freqs)

        # Mnożenie w dziedzinie częstotliwości
        filtered_fft = proj_fft * filter_vals

        # Powrót do dziedziny czasu
        return np.real(ifft(filtered_fft))

    def shepp_logan_filter(self, proj, cutoff=0.5):
        n = len(proj)

        # Transformata Fouriera sygnału
        proj_fft = fft(proj)
        freqs = fftfreq(n)  # Dla n punktów da wartości od -0.5 do +0.5 (co 1/n)

        # --- 1. Rampa (|omega|)
        ramp = np.abs(freqs)

        # --- 2. Shepp-Logan => dodatkowy czynnik sin(x)/x
        #     x = pi * f / (2 * cutoff);  f w przedziale [-0.5, 0.5]
        x = np.pi * freqs / (2 * cutoff)
        shepp_logan = np.ones_like(freqs)
        mask_nonzero = np.abs(x) > 1e-9
        shepp_logan[mask_nonzero] = np.sin(x[mask_nonzero]) / (x[mask_nonzero])

        # --- 3. Okno Hann
        hann_window = np.zeros_like(freqs)
        mask_in_cutoff = np.abs(freqs) <= cutoff
        # Wewnątrz pasma:
        hann_window[mask_in_cutoff] = 0.5 + 0.5 * np.cos(
            np.pi * freqs[mask_in_cutoff] / cutoff
        )

        # --- 4. Połączenie: filtr = ramp * Shepp-Logan * okno Hann
        filter_vals = ramp * shepp_logan * hann_window
        filtered_fft = proj_fft * filter_vals
        filtered_proj = np.real(ifft(filtered_fft))

        return filtered_proj

    # Przetwarzanie całego sinogramu
    def filter_sinogram(self, sinogram, filter_type):
        filtered_sinogram = np.zeros_like(sinogram)
        for i in range(sinogram.shape[0]):
            if filter_type == 'ramp':
                filtered_sinogram[i, :] = self.ramp_filter(sinogram[i, :])
            elif filter_type == 'shepp-logan':
                filtered_sinogram[i, :] = self.shepp_logan_filter(sinogram[i, :])
        return filtered_sinogram

    def bresenham_algorithm(self, x0, y0, x1, y1):
        points = []
        dx = abs(x0 - x1)
        dy = abs(y0 - y1)
        sx = 1 if x0 < x1 else -1   # sprawdzanie kierunku kroku (prawo 1 lewo -1)
        sy = 1 if y0 < y1 else -1   # 1 dół -1 góra
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def print_sinogram(self, sino):
        # Normalizacja do przedziału 0-255
        sinogram_unnormalized = (sino - sino.min()) / (sino.max() - sino.min()) * 255
        sinogram_image = Image.fromarray(sinogram_unnormalized.astype(np.uint8))
        sinogram_resized = sinogram_image.resize((400, 400))
        self.sinogram_image = ImageTk.PhotoImage(sinogram_resized)

        # Aktualizacja wyświetlania
        self.sinogram_canvas.delete("all")
        self.sinogram_canvas.create_image(200, 200, image=self.sinogram_image, anchor=tk.CENTER)
        self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
        time.sleep(0.01)  # Opóźnienie

    def radon_transform(self, span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows):

        sino = np.zeros((num_angles, num_detectors))

        for i, angle in enumerate(np.linspace(0, span, num_angles, endpoint=False)):
            theta = np.deg2rad(angle)

            # Kierunek promienia
            dir_x = np.cos(theta)
            dir_y = np.sin(theta)
            # Promień normalny do dir_x, dir_y
            perp_x = -dir_y
            perp_y = dir_x

            for j in range(num_detectors):
                # offset przesuwa detektor wzdłuż wektora prostopadłego
                offset = (j - num_detectors // 2) * detector_spacing

                # Środek j-tego promienia (punkt na prostopadłej)
                ray_center_x = center_x + offset * perp_x
                ray_center_y = center_y + offset * perp_y

                # Emiter = ray_center - promień okręgu w kierunku -dir
                emitter_x = ray_center_x - max_radius * dir_x
                emitter_y = ray_center_y - max_radius * dir_y

                # Detektor = ray_center + promień okręgu w kierunku dir
                det_x = ray_center_x + max_radius * dir_x
                det_y = ray_center_y + max_radius * dir_y

                # Oblicz punkty na linii
                line_points = self.bresenham_algorithm(int(emitter_x), int(emitter_y), int(det_x), int(det_y))
                sino[i, j] = sum(self.image_array[y, x] for x, y in line_points if 0 <= x < cols and 0 <= y < rows)
        return sino

    def steps_radon_transform(self, span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows):
        delay = 0.05  # Czas opóźnienia w sekundach

        sino = np.zeros((num_angles, num_detectors))

        for i, angle in enumerate(np.linspace(0, span, num_angles, endpoint=False)):
            theta = np.deg2rad(angle)

            # Kierunek promienia
            dir_x = np.cos(theta)
            dir_y = np.sin(theta)
            # Promień normalny do dir_x, dir_y
            perp_x = -dir_y
            perp_y = dir_x

            for j in range(num_detectors):
                # offset przesuwa detektor wzdłuż wektora prostopadłego
                offset = (j - num_detectors // 2) * detector_spacing

                # Środek j-tego promienia (punkt na prostopadłej)
                ray_center_x = center_x + offset * perp_x
                ray_center_y = center_y + offset * perp_y

                # Emiter = ray_center - promień okręgu w kierunku -dir
                emitter_x = ray_center_x - max_radius * dir_x
                emitter_y = ray_center_y - max_radius * dir_y

                # Detektor = ray_center + promień okręgu w kierunku dir
                det_x = ray_center_x + max_radius * dir_x
                det_y = ray_center_y + max_radius * dir_y

                # Oblicz punkty na linii
                line_points = self.bresenham_algorithm(int(emitter_x), int(emitter_y), int(det_x), int(det_y))
                sino[i, j] = sum(
                    self.image_array[y, x] for x, y in line_points if 0 <= x < cols and 0 <= y < rows
                )

            self.print_sinogram(sino)

        return sino

    def reverseRadonTransform(self, rows, cols, delta_alpha, center_x, center_y, num_angles, max_radius, detector_spacing, num_detectors, sinogram):
        print("Implementacja odwrotnej transformaty Radona")

        reverse_transformed = np.zeros((rows, cols))
        delta_theta = np.deg2rad(delta_alpha)

        for x in range(rows):
            for y in range(cols):
                value = 0
                relative_x = x - center_x
                relative_y = center_y - y

                for N in range(num_angles):
                    theta = np.deg2rad(N * delta_alpha)
                    t = (relative_x * np.cos(theta) + relative_y * np.sin(theta))

                    # z racji na to, że t może przyjmować niedyskretne wartości należy przeprowadzić interpolację liniową między 2 najbliższymi wartościami w sinogramie
                    g0 = 0
                    det_number = (t + max_radius) / detector_spacing
                    det_closest = int(np.floor(det_number))
                    det_second = min(det_closest + 1, num_detectors - 1)

                    # Liniowa interpolacja – sprawdzamy, czy indeks mieści się w zakresie
                    if det_number < 0 or det_number > num_detectors - 1:
                        g0 = 0
                    else:
                        j0 = int(np.floor(det_number))
                        j1 = min(j0 + 1, num_detectors - 1)
                        alpha = det_number - j0
                        g0 = (1 - alpha) * sinogram[N, j0] + alpha * sinogram[N, j1]
                    value += delta_theta * g0
                reverse_transformed[x][y] = (1 / (2 * np.pi)) * value

        print("Wyświetlanie wynikowego obrazu")
        # Przekształcenie do przedziału 0-255
        reverse_transformed_unnormalised = (reverse_transformed - reverse_transformed.min()) / (reverse_transformed.max() - reverse_transformed.min()) * 255
        reverse_transformed_image = Image.fromarray(reverse_transformed_unnormalised.astype(np.uint8))
        reverse_transformed_resized = reverse_transformed_image.resize((400, 400))
        reconstructed_img = ImageTk.PhotoImage(reverse_transformed_resized)

        # Aktualizacja wyświetlania w canvasie rekonstrukcji
        self.reconstruction_canvas.delete("all")
        self.reconstruction_canvas.create_image(200, 200, image=reconstructed_img, anchor=tk.CENTER)
        self.reconstructed_image = reconstructed_img

        return reverse_transformed_unnormalised


    def vectorisedReverseTransform(self, rows, cols, span, delta_alpha, center_x, center_y, num_angles, max_radius, detector_spacing, num_detectors, sinogram):
        print("Implementacja wektoryzowanej odwrotnej transformaty Radona")

        # Przygotowanie siatki współrzędnych dla obrazu
        xx, yy = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
        relative_x = xx - center_x  # macierz rozmiaru (rows, cols)
        relative_y = center_y - yy

        # Tablica kątów w radianach
        angles = np.deg2rad(np.arange(0, span, delta_alpha))
        dtheta = np.deg2rad(delta_alpha)

        # Inicjalizacja rekonstrukcji
        reconstructed = np.zeros((rows, cols))

        # Wektorowe przetwarzanie dla każdego kąta
        for i, theta in enumerate(angles):
            # Obliczenie wartości t dla wszystkich pikseli jednocześnie
            t = relative_x * np.cos(theta) + relative_y * np.sin(theta)
            # Przekształcenie wartości t na indeksy detektorów (detektory: -max_radius do max_radius)
            det_index = (t + max_radius) / detector_spacing

            # Liniowa interpolacja – obliczamy indeksy dolne i górne
            j0 = np.floor(det_index).astype(np.int64)
            j1 = j0 + 1

            # Ograniczamy indeksy do zakresu [0, num_detectors-1]
            j0 = np.clip(j0, 0, num_detectors - 1)
            j1 = np.clip(j1, 0, num_detectors - 1)

            # Współczynnik interpolacji
            alpha = det_index - j0

            # Pobieramy odpowiednią projekcję (wiersz sinogramu) dla kąta theta
            try:
                proj = sinogram[i, :]  # kształt: (num_detectors,)
            except IndexError:
                pass
            # Aby operować wektorowo, rozszerzamy projekcję do rozmiaru obrazu:
            # Możemy wykorzystać fancy indexing – sinogram[i, j0] daje macierz (rows, cols)
            proj_val = (1 - alpha) * proj[j0] + alpha * proj[j1]

            # Dodajemy wkład dla danego kąta (pamiętaj o kroku kątowym)
            reconstructed += dtheta * proj_val

        # Finalna normalizacja
        reconstructed /= (2 * np.pi)

        # Skalowanie do przedziału 0-255 i konwersja do obrazu
        reconstructed_unnorm = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min()) * 255
        reconstructed_img = Image.fromarray(reconstructed_unnorm.astype(np.uint8))
        reconstructed_resized = reconstructed_img.resize((400, 400))
        reconstructed_photo = ImageTk.PhotoImage(reconstructed_resized)

        # Aktualizacja wyświetlania w canvasie rekonstrukcji
        self.reconstruction_canvas.delete("all")
        self.reconstruction_canvas.create_image(200, 200, image=reconstructed_photo, anchor=tk.CENTER)
        self.reconstructed_image = reconstructed_photo  # zachowujemy referencję

        return reconstructed_unnorm


    def calculateAvgError(self, rows, cols):
        original = self.image_array
        # normalizuje oryginalny obraz do tego samego formatu
        original = (original - original.min()) / (original.max() - original.min()) * 255
        reconstructed_image = self.reconstructed_array
        sum_errors = 0
        for i in range (rows):
            for j in range(cols):
                sum_errors += (original[i][j] - reconstructed_image[i][j]) ** 2
        return (sum_errors / (rows * cols)) ** (1/2)
    def generate(self):
        # Pobranie parametrów
        delta_alpha = self.delta_alpha_var.get()
        num_detectors = self.num_detectors_var.get()
        span = self.span_var.get()
        selected_filter_str = self.filter_var.get()  # np. "2 - ramp_filter"
        selected_filter_num = int(selected_filter_str.split("-")[0].strip())  # np. 2
        show_steps = bool(self.show_steps.get())
        accelerate = bool(self.accelerate.get())
        print(f"Parametry: ∆α={delta_alpha}, n={num_detectors}, l={span}, filtr={selected_filter_str}, kroki pośrednie={show_steps}, accelerate{accelerate}")

        num_angles = int(span / delta_alpha)
        sinogram = np.zeros((num_angles, num_detectors)) # macierz reprezentująca sinogram

        rows, cols = self.image_array.shape
        center_x, center_y = cols // 2, rows // 2   # środek obrazu
        max_radius = min(center_x, center_y)
        detector_spacing = 2 * max_radius / num_detectors

        print(sinogram)

        if show_steps:
            if selected_filter_num:
                print("kroki + filtr")
                sinogram = self.steps_radon_transform(span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows)

                if selected_filter_num == 1:
                    print("Wybrano: filtr z wykładu")
                    # stosowanie filtru
                    filtr = self.create_filter()
                    for i in range(sinogram.shape[0]):  # Iteracja po rzędach sinogramu
                        sinogram[i, :] = np.convolve(sinogram[i, :], filtr, mode='same')

                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)
                elif selected_filter_num == 2:
                    print("Wybrano: ramp_filter")
                    sinogram = self.filter_sinogram(sinogram, 'ramp')
                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)
                else:
                    print("Wybrano: shepp_logan_filter")
                    sinogram = self.filter_sinogram(sinogram, 'shepp-logan')
                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)

            else:
                print("pokazuje kroki, bez filtru")
                sinogram = self.steps_radon_transform(span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows)

        else:
            if selected_filter_num:
                sinogram = self.radon_transform(span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows)

                if selected_filter_num == 1:
                    print("Wybrano: filtr z wykładu")
                    # stosowanie filtru
                    filtr = self.create_filter()
                    for i in range(sinogram.shape[0]):  # Iteracja po rzędach sinogramu
                        sinogram[i, :] = np.convolve(sinogram[i, :], filtr, mode='same')

                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)
                elif selected_filter_num == 2:
                    print("Wybrano: ramp_filter")
                    sinogram = self.filter_sinogram(sinogram, 'ramp')
                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)
                else:
                    print("Wybrano: shepp_logan_filter")
                    sinogram = self.filter_sinogram(sinogram, 'shepp-logan')
                    self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
                    time.sleep(1)  # Opóźnienie
                    self.print_sinogram(sinogram)
            else:
                print("no kroki no filtr")
                sinogram =self.radon_transform(span, num_angles, num_detectors, detector_spacing, center_x, center_y, max_radius, cols, rows)
            self.sinogram_canvas.update_idletasks()  # Odświeżenie interfejsu Tkintera
            time.sleep(0.01)  # Opóźnienie
            self.print_sinogram(sinogram)

        if accelerate:
            self.reconstructed_array = self.vectorisedReverseTransform(rows, cols, span, delta_alpha, center_x, center_y, num_angles, max_radius, detector_spacing, num_detectors, sinogram)
        else:
            self.reconstructed_array = self.reverseRadonTransform(rows, cols, delta_alpha, center_x, center_y, num_angles, max_radius, detector_spacing, num_detectors, sinogram)

        blad = self.calculateAvgError(rows, cols)
        self.error_var.set(f"{blad:.4f}")









if __name__ == "__main__":
    root = tk.Tk()
    app = TomographSimulator(root)
    root.mainloop()
