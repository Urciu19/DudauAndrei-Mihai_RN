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

* **Număr total de observații:** 12
* **Număr de caracteristici (features):** 4
* **Tipuri de date:**  Imagini
* **Format fișiere:**  PNG / JPG 

### 2.3 Descrierea fiecărei caracteristici

| **Caracteristică** | **Tip**          | **Unitate** | **Descriere**                                      | **Domeniu valori** |
|--------------------|------------------|-------------|----------------------------------------------------|---------------------|
| specia_1           | imagine (JPG/PNG) | pixeli      | Clopoțel *(Campanula patula)*                      | 150 × 150 px        |
| specia_2           | imagine (JPG/PNG) | pixeli      | Ferigă *(Polypodiopsida)*                          | 150 × 150 px        |
| specia_3           | imagine (JPG/PNG) | pixeli      | Floare de colț *(Leontopodium alpinum)*            | 150 × 150 px        |
| specia_4           | imagine (JPG/PNG) | pixeli      | Pătlagină *(Plantago major)*                       | 150 × 150 px        |


**Fișier recomandat:**  `data/README.md`

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

* **Medie, mediană, deviație standard**
* **Min–max și quartile**
* **Distribuții pe caracteristici** (histograme)
* **Identificarea outlierilor** (IQR / percentile)

### 3.2 Analiza calității datelor

* **Detectarea valorilor lipsă** (% pe coloană)
* **Detectarea valorilor inconsistente sau eronate**
* **Identificarea caracteristicilor redundante sau puternic corelate**

### 3.3 Probleme identificate

* [exemplu] Feature X are 8% valori lipsă
* [exemplu] Distribuția feature Y este puternic neuniformă
* [exemplu] Variabilitate ridicată în clase (class imbalance)

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

* **Eliminare duplicatelor**
* **Tratarea valorilor lipsă:**
  * Feature A: imputare cu mediană
  * Feature B: eliminare (30% valori lipsă)
* **Tratarea outlierilor:** IQR / limitare percentile

### 4.2 Transformarea caracteristicilor

* **Normalizare:** Min–Max / Standardizare
* **Encoding pentru variabile categoriale**
* **Ajustarea dezechilibrului de clasă** (dacă este cazul)

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

* Date preprocesate în `data/processed/`
* Seturi train/val/test în foldere dedicate
* Parametrii de preprocesare în `config/preprocessing_config.*` (opțional)

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
