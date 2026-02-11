import os
import cv2
from preprocesare_planta import preprocesare_planta

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def asigura_director(path):
    if not os.path.exists(path):
        os.makedirs(path)

def proceseaza_folder(folder_raw, folder_processed):
    """
    Procesează toate imaginile dintr-un folder și le salvează în folderul procesat.
    """
    asigura_director(folder_processed)

    for fisier in os.listdir(folder_raw):
        if fisier.lower().endswith((".jpg", ".jpeg", ".png")):
            cale_raw = os.path.join(folder_raw, fisier)
            cale_out = os.path.join(folder_processed, fisier)
            if cale_out.endswith(".jpg"):
                cale_out = cale_out[:-4] + ".png" 

            print(f"Procesez: {cale_raw}")

            try:
                img_finala = preprocesare_planta(cale_raw, afiseaza=False)
                cv2.imwrite(cale_out, img_finala)
            except Exception as e:
                print(f"Eroare la {cale_raw}: {e}")

    print(f"✔ Folder procesat: {folder_raw} → {folder_processed}")

def proceseaza_toate_plantele():
    """
    Parcurge toate subfolderele din data/raw și procesează imaginile corespunzător.
    """
    for planta in os.listdir(RAW_DIR):
        folder_raw = os.path.join(RAW_DIR, planta)

        if os.path.isdir(folder_raw):
            folder_processed = os.path.join(PROCESSED_DIR, planta)
            proceseaza_folder(folder_raw, folder_processed)

if __name__ == "__main__":
    proceseaza_toate_plantele()
    print("✔ Toate imaginile au fost procesate!")
