import cv2
import numpy as np

def remove_background_grabcut(img):
    """Elimină fundalul folosind GrabCut cu bounding box adaptiv."""
    h, w = img.shape[:2]
    rect = (int(w*0.05), int(h*0.25), int(w*0.90), int(h*0.70))

    mask = np.zeros((h, w), np.uint8)
    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img, mask, rect, bgModel, fgModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 0) | (mask == 2), 0, 255).astype("uint8")
    rezultat = cv2.bitwise_and(img, img, mask=mask2)

    return rezultat, mask2


def crop_to_content(mask, img):
    """Decupează imaginea exact pe zona plantei, fără margini negre."""
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    return img[y:y+h, x:x+w]


def resize_keep_ratio(img, target=150):
    """Resize la 150x150 păstrând aspect ratio (letterboxing)."""
    h, w = img.shape[:2]
    scale = target / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))

    result = np.zeros((target, target, 3), dtype=np.uint8)
    x_offset = (target - new_w) // 2
    y_offset = (target - new_h) // 2

    result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return result


def preprocesare_planta(path_imagine, afiseaza=False, dim=(500, 500)):
    """Pipeline curat: eliminare fundal -> crop -> resize."""

    img = cv2.imread(path_imagine)
    if img is None:
        raise ValueError("Imaginea nu a putut fi încărcată. Verifică path-ul!")

    # Redimensionare inițială pentru stabilitate
    img = cv2.resize(img, dim)

    # 1. Scoatem background-ul
    img_no_bg, mask_bg = remove_background_grabcut(img)

    # 2. Decupăm doar planta
    cropped = crop_to_content(mask_bg, img_no_bg)

    # 3. Imagine finală 150x150 păstrând forma
    final_img = resize_keep_ratio(cropped, 150)

    # Afișăm DOAR original + final
    if afiseaza:
        cv2.imshow("Imagine originală", img)
        cv2.imshow("Imagine finală fără background (150x150)", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_img
