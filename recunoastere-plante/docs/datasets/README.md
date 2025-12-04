## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care se analizează și se preprocesează setul de date necesar proiectului „Rețele Neuronale". Scopul etapei este pregătirea corectă a datelor pentru instruirea modelului RN, respectând bunele practici privind calitatea, consistența și reproductibilitatea datelor.

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```
recunoastere-plante/
│
├── README.md                     # descriere generală, instrucțiuni de folosire, documentație
│
├── config/                       # fișiere de configurare (parametri, setări, hyperparametri)
│
├── docs/                         # documentație auxiliară
│   └── datasets/                 # descriere seturi de date, surse, diagrame
│
├── data/                         # structura completă a datelor
│   ├── raw/                      # date brute (imaginile originale, neprelucrate)
│   │   ├── clopotel/            # specia 1
│   │   ├── feriga/              # specia 2
│   │   ├── floarecolt/          # specia 3
│   │   └── patlagina/           # specia 4
│   │
│   ├── processed/                # imaginile procesate (fără background, crop, resize 150x150px)
│   │   ├── clopotel/            # rezultatele preprocesării
│   │   ├── feriga/
│   │   ├── floarecolt/
│   │   └── patlagina/
│   │
│   ├── train/                   # setul folosit pentru antrenarea rețelei neurale
│   ├── validation/              # set folosit pentru validare în timpul antrenării
│   └── test/                    # set final pentru testarea performanței modelului
│
├── src/                         # codul sursă principal al proiectului
│   ├── preprocessing/           # module de preprocesare a imaginilor
│   │   ├── preprocesare_planta.py   # funcția principală: eliminare background, crop, resize
│   │   └── batch_process.py         # procesare automată pentru toate imaginile din data/raw
│   │
│   ├── data_acquisition/        # dacă există scripturi pentru colectare/organizare date
│   │   └── placeholder.txt
│   │
│   └── neural_network/          # implementarea viitoare a modelului ML (CNN)
│       └── placeholder.txt
│
└── requirements.txt             # lista dependențelor Python necesare proiectului

```

---

##  2. Descrierea Setului de Date

Setul de date utilizat în acest proiect este compus din imagini reprezentând diverse specii de plante, fiecare specie fiind organizată în propriul său subfolder. Imaginile sunt generate cu ajutorul unui model AI, ceea ce prezintă avantajul că nu implică probleme legate de drepturile de autor și permite controlul calității vizuale și al variabilității setului.

Fișierele sunt salvate preponderent în format JPG, un format compact care ocupă mai puțin spațiu față de PNG, menținând totuși un nivel suficient de bun al detaliilor pentru sarcinile de analiză vizuală și clasificare. După colectare, imaginile brute sunt procesate automat printr-un pipeline dedicat, care include eliminarea fundalului, decuparea obiectului principal (planta) și redimensionarea tuturor imaginilor la dimensiunea standard de 150×150 px.

Structura uniformizată și preprocesarea consecventă a datelor asigură o bază solidă pentru etapele ulterioare de antrenare și evaluare a modelelor de recunoaștere vizuală.

### 2.1 Sursa datelor

* **Origine:** Drone terestre, fotografii, camera telefonului, in mare parte orice dispozitiv capabil sa faca o poza, inclusiv poze generate de A.I(daca este cazul)
* **Modul de achiziție:**  Senzori reali /  Fișier extern /  Generare programatică
* **Perioada / condițiile colectării:**  Noiembrie 2025 - Decembrie 2025, condiții experimentale specifice

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** 18
* **Număr de caracteristici (features):** 5
* **Tipuri de date:**  Imagini
* **Format fișiere:**  PNG / JPG 

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip**          | **Unitate** | **Descriere**                                      | **Domeniu valori** |
|--------------------|------------------|-------------|----------------------------------------------------|---------------------|
| specia_1           | imagine (JPG/PNG) | pixeli      | Clopoțel *(Campanula patula)*                      | 150 × 150 px        |
| specia_2           | imagine (JPG/PNG) | pixeli      | Ferigă *(Polypodiopsida)*                          | 150 × 150 px        |
| specia_3           | imagine (JPG/PNG) | pixeli      | Floare de colț *(Leontopodium alpinum)*            | 150 × 150 px        |
| specia_4           | imagine (JPG/PNG) | pixeli      | Pătlagină *(Plantago major)*                       | 150 × 150 px        |
| specia_5           | imagine (JPG/PNG) | pixeli      | Păpădie *(Taraxacum officinale)*                   | 150 × 150 px        |


**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic
**Structura datelor din fisierul data/raw**
```
data/raw/
│
├── clopotel     → 3 imagini
├── feriga       → 3 imagini
├── floarecolt   → 3 imagini
├── papadie      → 6 imagini
└── patlagina    → 3 imagini

```
**Structura datelor din fisierul data/processed**

```
data/processed/
│
├── clopotel     → 3 imagini procesate
├── feriga       → 3 imagini procesate
├── floarecolt   → 3 imagini procesate
├── papadie      → 6 imagini procesate
└── patlagina    → 3 imagini procesate

```

### 3.1 Statistici descriptive aplicate

Pe baza datelor furnizate:

* **Număr total de imagini brute:** 18

* **Număr total de imagini procesate:** 18

* **Repartizarea pe clase:** clopoțel → 3 imagini
                             ferigă → 3 imagini
                             floare de colț → 3 imagini
                             pătlagină → 3 imagini
                             păpădie → 6 imagini

Dataset-ul conține acum 5 clase distincte, cu un ușor dezechilibru:
→ Papadie are de două ori mai multe imagini decât celelalte clase.

Au fost evaluate următoarele caracteristici:

* **format fișier:** JPG (majoritar)
* **dimensiuni brute variabile**
* **dimensiuni procesate standardizate:** 150×150 px
* **distribuții ale culorilor pe canalele RGB**
* **ariile bounding box-urilor după segmentare**
* **raport planta/fundal înainte și după preprocesare**

### 3.2 Analiza calității datelor

Conform analizei structurii reale:

**Disponibilitate și consistență**

* Nu există fișiere lipsă sau corupte.
* Structura raw și processed este coerentă și completă.

**Calitatea imaginilor generate AI**

* Imaginile sunt clare, bine definite, fără probleme majore de compresie.

**Standardizare în procesare**

În data/processed/, toate imaginile sunt:

* 150×150 px
* plante centrate corect
* fundal eliminat cu succes (GrabCut + crop)

**Echilibru parțial în clase**

* În timp ce majoritatea speciilor au 3 imagini, papadie are 6, ceea ce introduce un mic dezechilibru.

**Rezoluții brute variabile**

Imaginile inițiale au rezoluții diferite, dar pipeline-ul rezolvă această problemă prin:

* resize inițial
* eliminare fundal
* crop adaptiv
* resize final 150×150 px

### 3.3 Probleme identificate

* **Clasă cu număr dublu de imagini (papadie)**

Aceasta poate influența modelul spre a învăța mai bine clasa cu mai multe exemple.
  → Soluție: Augmentare pe celelalte clase.

* **Variabilitate mare de rezoluții brute**

Unele imagini AI pot fi foarte mari, altele mici.
  → Pipeline-ul controlează complet această variabilitate.

* **Lipsa datelor naturale**

Fiind imagini generate AI, nu reflectă complexitatea completă a fotografiilor reale din natură.
  → Recomandare: adăugarea de imagini reale în dataset.

* **Fundaluri artificiale**

AI generează fundaluri uneori prea uniforme sau prea artistice.
  → Segmentarea prin GrabCut reduce problema, dar nu o elimină total.

---

##  4. Preprocesarea Datelor

Preprocesarea este un pas esențial în proiectele de clasificare a imaginilor, deoarece garantează consistența vizuală, elimină informația inutilă (fundalul), normalizează dimensiunea imaginilor și pregătește setul de date pentru antrenarea rețelei neuronale.

### 4.1 Curățarea datelor

Pentru setul de date actual (imagini brute generate AI), au fost aplicate următoarele proceduri:

**Eliminarea duplicatelor**

* Imaginile au fost verificate vizual și structural pentru a identifica eventuale duplicate generate automat. În setul actual nu au fost identificate duplicări.

**Verificarea imaginilor corupte**

Toate fișierele din data/raw/ au fost citite cu OpenCV pentru a confirma integritatea. Nu există imagini corupte.

**Tratarea valorilor lipsă**

În cazul imaginilor, valorile lipsă se referă la:
  * fișiere inexistente
  * fișiere neîncărcabile
  * imagini cu background complet uniform (neinformativ)

În datasetul furnizat nu au fost identificate astfel de cazuri.

**Curățarea fundalului (principalul pas de igienizare vizuală)**

Fundalul a fost eliminat prin pipeline-ul:
  * GrabCut pentru segmentarea plantei
  * Mask binar (0 = fundal, 255 = plantă)
  * Decupare exactă (crop la bounding box)

Acest pas îmbunătățește semnificativ calitatea setului pentru clasificare.

### 4.2 Transformarea caracteristicilor

Spre deosebire de tabele, imaginile necesită transformări specifice Computer Vision.

**Redimensionare la format standard**

* Toate imaginile sunt scalate la dimensiunea fixă 150×150 px, păstrând raportul de aspect (cu letterboxing dacă este necesar).

**Normalizarea valorilor pixelilor**

Imaginile pot fi normalizate ulterior la:
  * [0, 1] pentru modele CNN
  * SAU standardizate cu mean/std calculate DOAR pe train

(normalizarea se va aplica în etapa de model, nu în folderele procesate)

**Eliminarea background-ului**

Transformare majoră pentru claritatea claselor:
  * reduce variația inutilă a fundalului
  * scoate în evidență forma și textura plantei
  * îmbunătățește acuratețea clasificării

**Ajustarea dezechilibrului de clasă (dacă este cazul)**

În setul actual:
  * fiecare clasă are 3 imagini
  * papadie are 6 imagini

Pentru un model neural, recomandăm:
  * augmentare a claselor mai mici (flip, rotire, zoom, blur)
  * SAU sub-sampling al clasei mai mari, în funcție de strategie

### 4.3 Structurarea seturilor de date

După preprocesare, datele sunt pregătite pentru împărțirea în:

* **train (70–80%)**

* **validation (10–15%)**

* **test (10–15%)**

***Principii respectate***

**Stratificare**

* Fiecare set conține proporții similare de imagini pentru fiecare specie.

***Fără scurgere de informație (data leakage)***

  * Imaginile procesate nu influențează ulterior transformările altor imagini.

  * Normalizarea trebuie calculată din train și aplicată identic pe val/test.

***Separarea clară a folderelor***
```
data/train/
data/validation/
data/test/
```

### 4.4 Salvarea rezultatelor preprocesării

Rezultatele sunt salvate astfel:

**data/processed/**

Conține:
* imaginile cropuite și standardizate la 150×150 px
* fundal eliminat
* plante centrate

Structura este identică cu data/raw/.

**Seturile finale pentru învățare**
Vor fi salvate în:
```
data/train/
data/validation/
data/test/

```
Configurații opționale

Parametrii de preprocesare pot fi salvați în:

```
config/preprocessing_config.json

```
Exemple de parametri:
  * dimensiune imagine
  * tip normalizare
  * număr iterații GrabCut
  * marginile bounding box-ului procentual

---

##  5. Fișiere Generate în Această Etapă

* `data/raw/` – date brute
* `data/processed/` – date curățate & transformate
* `data/train/`, `data/validation/`, `data/test/` – seturi finale
* `src/preprocessing/` – codul de preprocesare
* `data/README.md` – descrierea dataset-ului

---

##  6. Stare Etapă (de completat de student)

- [X] Structură repository configurată
- [ ] Dataset analizat (EDA realizată)
- [ ] Date preprocesate
- [ ] Seturi train/val/test generate
- [ ] Documentație actualizată în README + `data/README.md`

---
