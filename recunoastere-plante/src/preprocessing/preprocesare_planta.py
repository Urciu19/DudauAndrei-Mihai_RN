import cv2
import numpy as np


def _dynamic_rect_from_hsv(img_bgr):
    """
    Încearcă să găsească zona plantei pe baza saturației/verdeții (HSV),
    apoi returnează un bounding rect "umflat" (padding).
    Dacă nu reușește, întoarce None.
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Praguri "safe" pentru vegetație (nu perfecte, dar bune ca inițializare)
    # Ajustate să prindă și plante mai puțin verzi (AI images / lumină ciudată)
    lower = np.array([15, 25, 25], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Curățare
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Dacă e prea puțin foreground, abandonăm
    if cv2.countNonZero(mask) < int(0.01 * h * w):
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Ia cel mai mare contur (cel mai probabil planta)
    c = max(contours, key=cv2.contourArea)
    x, y, rw, rh = cv2.boundingRect(c)

    # Padding (umflăm dreptunghiul ca să nu tăiem frunze)
    pad_x = int(0.08 * w)
    pad_y = int(0.08 * h)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w - 1, x + rw + pad_x)
    y2 = min(h - 1, y + rh + pad_y)

    rect = (x1, y1, x2 - x1, y2 - y1)

    # Rect invalid? (rare)
    if rect[2] <= 2 or rect[3] <= 2:
        return None

    return rect


def remove_background_grabcut(img_bgr, iterations=8):
    """
    Elimină fundalul folosind GrabCut:
    - încearcă rect dinamic (din HSV)
    - dacă nu merge, folosește fallback rect procentual
    - folosește și o mască inițială (seed) pentru stabilitate
    Returnează: (img_no_bg_bgr, mask_255)
    """
    h, w = img_bgr.shape[:2]

    # 1) rect dinamic (preferat)
    rect = _dynamic_rect_from_hsv(img_bgr)

    # 2) fallback dacă nu găsim rect dinamic
    if rect is None:
        rect = (int(w * 0.05), int(h * 0.15), int(w * 0.90), int(h * 0.80))

    # Inițializăm masca:
    # - default: probabil background
    mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    x, y, rw, rh = rect
    x2, y2 = x + rw, y + rh

    # În interiorul rect-ului: probabil foreground
    mask[y:y2, x:x2] = cv2.GC_PR_FGD

    # Un mic nucleu central: sigur foreground (ajută să nu „mănânce” obiectul)
    cx1 = x + int(0.25 * rw)
    cy1 = y + int(0.25 * rh)
    cx2 = x + int(0.75 * rw)
    cy2 = y + int(0.75 * rh)
    mask[cy1:cy2, cx1:cx2] = cv2.GC_FGD

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    # IMPORTANT: inițializăm cu mască (mai stabil decât doar rect)
    cv2.grabCut(img_bgr, mask, None, bgModel, fgModel, iterations, cv2.GC_INIT_WITH_MASK)

    # mască finală binară: 255 = plantă, 0 = fundal
    mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    # Dacă masca e goală (rare), fallback: nu scoatem nimic
    if cv2.countNonZero(mask2) < 10:
        mask2 = np.full((h, w), 255, dtype=np.uint8)

    rezultat = cv2.bitwise_and(img_bgr, img_bgr, mask=mask2)
    return rezultat, mask2


def crop_to_content(mask_255, img_bgr):
    """
    Decupează imaginea pe zona non-zero din mască.
    Dacă masca e goală, întoarce imaginea originală.
    """
    coords = cv2.findNonZero(mask_255)
    if coords is None:
        return img_bgr

    x, y, w, h = cv2.boundingRect(coords)

    # fallback dacă boundingRect e suspect
    if w <= 2 or h <= 2:
        return img_bgr

    return img_bgr[y:y + h, x:x + w]


def resize_keep_ratio(img_bgr, target=150):
    """
    Resize la target x target păstrând aspect ratio (letterbox pe fundal negru).
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((target, target, 3), dtype=np.uint8)

    scale = target / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))

    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

    result = np.zeros((target, target, 3), dtype=np.uint8)
    x_offset = (target - new_w) // 2
    y_offset = (target - new_h) // 2

    result[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return result


def preprocesare_planta(path_imagine, afiseaza=False, dim=(500, 500)):
    """
    Pipeline: resize stabil -> GrabCut (mai robust) -> crop -> resize 150x150 păstrând forma.
    Returnează imaginea finală BGR (pentru cv2.imwrite).
    """
    img = cv2.imread(path_imagine)
    if img is None:
        raise ValueError(f"Imaginea nu a putut fi încărcată: {path_imagine}")

    # Resize inițial (stabilizează parametrii pentru GrabCut)
    img_resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # 1) Scoatem background-ul (GrabCut robust)
    img_no_bg, mask_bg = remove_background_grabcut(img_resized, iterations=8)

    # 2) Decupăm doar zona plantei
    cropped = crop_to_content(mask_bg, img_no_bg)

    # 3) Resize final la 150x150 păstrând forma
    final_img = resize_keep_ratio(cropped, target=150)

    # Afișăm doar original (resized) + final
    if afiseaza:
        cv2.imshow("Imagine originală (resize pentru stabilitate)", img_resized)
        cv2.imshow("Imagine finală fără background (150x150)", final_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return final_img