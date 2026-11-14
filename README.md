# CIFAR-10: klasifikacija slika neuronskim mrežama (PyTorch, MLflow, Optuna)

Domaći zadatak iz predmeta Veštačka inteligencija sa primenama. Skup podataka je [CIFAR-10 sa Kaggle-a](https://www.kaggle.com/competitions/cifar-10/data): 50.000 označenih RGB slika 32x32 u 10 klasa (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

U radu je upoređeno 5 verzija modela (MLP, CNN, CNN sa BatchNorm i Dropout, ista mreža sa augmentacijom, mali ResNet), svaka sa ručno izabranim i sa optimizovanim hiperparametrima, uz 5-fold cross-validaciju i ocenu na odvojenom test skupu.

## Struktura repozitorijuma

```
configs/
  mlp.yaml, cnn.yaml, cnn_bn.yaml, cnn_bn_aug.yaml, resnet.yaml   ručni hiperparametri
  tuned/                       konfiguracije sa hiperparametrima iz Optune (pravi ih tune.py)
  split_seed42.csv             podela na trainval i test (seed 42)
  normalization.json           mean i std po RGB kanalu
src/
  data.py                      raspakivanje, validacija, split
  models.py                    arhitekture mreža
  train.py                     5-fold cross-validacija, MLflow logovanje
  tune.py                      optimizacija hiperparametara (Optuna)
  utils.py                     seed-ovi i evidencija okruženja
notebooks/
  01_eda.ipynb                 analiza podataka
  02_comparison.ipynb          poređenje modela
results/
  eda/                         grafikoni iz analize podataka
  runs/<ime>/                  rezultati treniranja jedne verzije
  tuning/<ime>/                rezultati optimizacije jednog modela
  comparison/                  tabele i grafikoni iz poređenja
logs/                          tekstualni logovi treniranja i tuninga
mlruns/                        MLflow evidencija
requirements.txt               glavni paketi
requirements-lock.txt          tačne verzije paketa korišćene u eksperimentima
```

## Instalacija

Korišćeno je Python 3.12.7.

```
python -m venv .venv
.venv\Scripts\Activate.ps1            # Windows PowerShell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Za tačnu reprodukciju verzija:

```
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Podaci

Na stranici takmičenja (potrebno je prihvatiti pravila preko "Join Competition") skinuti samo dva fajla i staviti ih u `data/raw/`:

- `train.7z`
- `trainLabels.csv`

Fajl `test.7z` nije potreban: nema oznake, pa se test skup izdvaja iz označenih podataka.

Zatim pripremiti podatke:

```
python src/data.py
```

Skripta raspakuje slike u `data/images/`, proveri da nema duplikata ID-jeva, nedostajućih vrednosti ni nepoznatih klasa i da je svaka slika 32x32x3. Pravi `data/processed/cifar10.npz` i stratifikovanu podelu u `configs/split_seed42.csv`: 45.000 slika za trening i validaciju (trainval) i 5.000 za test. Folder `data/` nije u repou. Podela i `configs/normalization.json` jesu u repou, pa svako dobija iste vrednosti.

## Analiza podataka

Otvoriti `notebooks/01_eda.ipynb`: distribucija klasa, primeri slika, statistika piksela, histogrami, korelacija kanala, provera nedostajućih vrednosti i duplikata. Ćelija sa statistikom piksela upisuje `configs/normalization.json`.

## Modeli i konfiguracije

| Verzija | Arhitektura |
|---|---|
| `mlp` | Spljoštena slika, dva skrivena sloja (512 i 256 neurona) |
| `cnn` | 3 bloka (konvolucija, ReLU, max pooling) sa 32, 64 i 128 filtera, pa 2 linearna sloja |
| `cnn_bn` | Ista osnova, uz BatchNorm posle konvolucija i Dropout (0,3) |
| `cnn_bn_aug` | Isti model kao `cnn_bn`, trenira se sa augmentacijom (horizontalno okretanje, nasumično isecanje) |
| `resnet` | Mali ResNet: 3 stupnja po jedan rezidualni blok (32, 64 i 128 kanala) |

Ručni hiperparametri za sve verzije: Adam, `lr` 0,001, `batch_size` 128, `weight_decay` 0, 10 epoha, 5 fold-ova, seed 42. Svaka verzija ima i optimizovanu varijantu (`_tuned`).

## Treniranje i optimizacija: redosled i komande

Rad ima tri koraka. Prvi (ručni hiperparametri) je polazna tačka (baseline) i nije zahtev zadatka, a optimizacija hiperparametara i treniranje optimizovanih verzija su glavni deo rada.

Sve komande se pokreću iz korena projekta sa aktiviranim `.venv`. Svaka od njih ima i opciju `--quick`, koja pokreće brz test na malom podskupu (par sekundi do minut). Brzi test upisuje u `results/quick/` i `mlruns_quick/`, koji nisu u repou.

Svako treniranje radi stratifikovanu 5-fold cross-validaciju nad trainval delom (45.000 slika). Za svaki fold se trenira novi model, meri na validacionom delu i na test skupu (5.000 slika). Rezultati idu u:

- `results/runs/<ime>/`: `summary.json` (prosek i std), `fold_metrics.csv`, `history.csv` (gubitak po epohi), `confusion_matrix_fold*.csv`, `oof_predictions.npz` (predikcije za ROC/PR), `config.yaml` i `env_info.json`
- `logs/<ime>.log`
- `mlruns/`: MLflow evidencija

U MLflow se loguju hiperparametri, gubitak i tačnost po epohi, metrike po fold-u (tačnost, macro F1, precision, recall, ROC AUC, PR AUC, test tačnost i F1), vreme treniranja, vreme inferencije po slici, veličina modela, vrh RAM-a i iskorišćenost CPU-a. Uz svaki run se čuvaju verzije biblioteka, podaci o mašini i commit koda.

### Korak 1: treniranje sa ručnim hiperparametrima (polazna tačka)

Pokreće se jednom od komandi ispod, po jedna za svaku verziju modela. Hiperparametri su isti za sve: Adam, `lr` 0,001, `batch_size` 128, 10 epoha. Ove verzije nemaju test ocenu, jer je ona dodata u `train.py` kasnije.

```
# MLP (oko 2 minuta)
python src/train.py --config configs/mlp.yaml

# CNN (oko 12 minuta)
python src/train.py --config configs/cnn.yaml

# CNN sa BatchNorm i Dropout (oko 15 minuta)
python src/train.py --config configs/cnn_bn.yaml

# CNN sa BatchNorm, Dropout i augmentacijom (oko 17 minuta)
python src/train.py --config configs/cnn_bn_aug.yaml

# ResNet (oko 50 minuta)
python src/train.py --config configs/resnet.yaml
```

### Korak 2: optimizacija hiperparametara (Optuna)

Pokreće se jednom od komandi ispod, po jedna za svaki model. Optuna pretražuje `optimizer` (Adam ili SGD), `lr`, `batch_size` (64, 128, 256), `weight_decay` i, za modele sa Dropout-om, `dropout`. Svaki pokušaj je ocenjen istom 5-fold cross-validacijom, 10 epoha, na svih 45.000 slika. Podrazumevano je 8 pokušaja (5 nasumičnih i 3 vođena prethodnim rezultatima), a mogu se menjati opcijama `--trials` i `--startup`. Test skup se u optimizaciji ne koristi.

```
# MLP (oko 20 minuta)
python src/tune.py --config configs/mlp.yaml

# CNN (oko 1,5 do 2 sata)
python src/tune.py --config configs/cnn.yaml

# CNN sa BatchNorm i Dropout (oko 2 sata)
python src/tune.py --config configs/cnn_bn.yaml

# CNN sa BatchNorm, Dropout i augmentacijom (oko 2,5 sata)
python src/tune.py --config configs/cnn_bn_aug.yaml

# ResNet (oko 7 sati)
python src/tune.py --config configs/resnet.yaml
```

Izlaz: `results/tuning/<ime>/` (`trials.csv`, `best_params.json`), `logs/tune_<ime>.log` i `configs/tuned/<ime>_tuned.yaml`.

### Korak 3: treniranje i testiranje optimizovanih verzija

Pokreće se jednom od komandi ispod, po jedna za svaki model, sa najboljim hiperparametrima iz koraka 2. Konfiguracije `configs/tuned/*.yaml` već postoje u repou, pa se ovaj korak može pokrenuti i bez ponovne optimizacije. Ista komanda i trenira i testira: na kraju svakog fold-a model se ocenjuje na test skupu (5.000 slika), koji se nikad ne koristi za treniranje.

```
# MLP (oko 3 minuta)
python src/train.py --config configs/tuned/mlp_tuned.yaml

# CNN (oko 12 minuta)
python src/train.py --config configs/tuned/cnn_tuned.yaml

# CNN sa BatchNorm i Dropout (oko 15 minuta)
python src/train.py --config configs/tuned/cnn_bn_tuned.yaml

# CNN sa BatchNorm, Dropout i augmentacijom (oko 15 minuta)
python src/train.py --config configs/tuned/cnn_bn_aug_tuned.yaml

# ResNet (oko 52 minuta)
python src/train.py --config configs/tuned/resnet_tuned.yaml
```

Navedena vremena su okvirna i odnose se na CPU opisan na kraju dokumenta.

## MLflow interfejs (opciono)

Interfejs za pregled svih eksperimenata u browseru.

```
$env:MLFLOW_ALLOW_FILE_STORE = "true"      # PowerShell
mlflow ui --backend-store-uri mlruns
```

Napomena: putanje do artefakata u `mlruns/` zapisane su na računaru na kome su eksperimenti rađeni, pa se artefakti možda neće otvoriti na drugom računaru. Parametri i metrike se vide. Isti artefakti su i u `results/runs/`.

## Poređenje

Otvoriti `notebooks/02_comparison.ipynb`. Tabele i grafikoni se čuvaju u `results/comparison/`.

## Rezultati

Optimizovane verzije. CV je prosek preko 5 validacionih fold-ova (± std), test je prosek 5 fold-modela na istih 5.000 test slika.

| Verzija | CV tačnost | Test tačnost | Vreme treniranja | Inferencija (ms/slici) | Parametri | CV pre optimizacije |
|---|---|---|---|---|---|---|
| `mlp_tuned` | 0,5329 ± 0,0053 | 0,5332 | 190 s | 0,013 | 1.707.274 | 0,5140 |
| `cnn_tuned` | 0,7268 ± 0,0096 | 0,7320 | 698 s | 0,209 | 620.362 | 0,7272 |
| `cnn_bn_tuned` | 0,7579 ± 0,0147 | 0,7610 | 921 s | 0,222 | 620.586 | 0,7687 |
| `cnn_bn_aug_tuned` | 0,7437 ± 0,0041 | 0,7457 | 886 s | 0,218 | 620.586 | 0,7435 |
| `resnet_tuned` | 0,7631 ± 0,0074 | 0,7688 | 3128 s | 0,613 | 308.650 | 0,7316 |

Ručne verzije nemaju test ocenu, jer su trenirane pre nego što je u `train.py` dodata. Svi detalji (metrike po fold-u, F1, resursi, learning curves, konfuzione matrice, ROC i PR krive) su u `notebooks/02_comparison.ipynb`.

Zaključci: konvolucione mreže su oko 20 procentnih poena bolje od MLP-a. ResNet i CNN+BN su na test skupu praktično izjednačeni (razlika manja od 1 poena), a ResNet se trenira oko tri puta duže. Optimizacija je pomogla MLP-u i ResNet-u, a kod `cnn_bn` je ispala malo lošija od ručnih vrednosti: ručna kombinacija nije bila među probanim, a pretraga je mala (8 pokušaja).

## Napomene o rezultatima

- Hiperparametri su birani po rezultatu cross-validacije (najbolja od 8 probanih kombinacija), pa je taj rezultat blago optimističan. Zato postoji odvojen test skup od 5.000 slika, koji nije korišćen ni u treningu ni u optimizaciji.
- Test ocena: svih 5 modela iz fold-ova (svaki treniran na 36.000 slika) ocenjen je na istih 5.000 test slika, a navodi se prosek. Nije istreniran jedan model na svih 45.000 slika.
- Learning curves prate gubitak po epohama na trening i validacionim podacima. Konfuzione matrice i ROC/PR krive računate su nad validacionim delovima fold-ova (slike koje model nije video tokom treniranja), a ne nad test skupom, jer test predikcije nisu čuvane.
- Težine istreniranih modela se ne čuvaju, pa se za novu ocenu model mora ponovo istrenirati.
- Optimizacija ima samo 8 pokušaja po modelu.

## Reproduktivnost i hardver

- Seed 42 (`random`, `numpy`, `torch`), isti split i isti fold-ovi za sve verzije.
- Verzije biblioteka i podaci o mašini se čuvaju uz svaki run (`env_info.json`, MLflow), a tačne verzije paketa su u `requirements-lock.txt`.
- Sva merenja su rađena na CPU-u: AMD Ryzen (12 niti), Windows 11, Python 3.12.7, PyTorch 2.15.0+cpu. Vremena treniranja i inferencije zavise od hardvera.

## Git istorija

Rad je razvijan kroz grane: `eda` (podaci i analiza), `models` (arhitekture i pipeline), `exp/mlp`, `exp/cnn`, `exp/cnn-bn`, `exp/augmentation`, `exp/resnet` (eksperimenti sa ručnim hiperparametrima), `exp/tuning` (optimizacija i test ocena), `exp/tuned-models` (treniranje optimizovanih verzija) i `comparison` (poređenje). Svaka grana je spojena u `main`.
