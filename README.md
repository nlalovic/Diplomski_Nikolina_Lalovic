# Upravljanje robotskom rukom na osnovu praćenja pogleda

Programsko rješenje razvijeno u okviru diplomskog rada *Razvoj asistivnog softvera za upravljanje robotskom rukom za manipulaciju na osnovu praćenja pogleda i detekcije namjere korisnika*, Fakultet tehničkih nauka, Univerzitet u Novom Sadu, 2026.

Autor: Nikolina Lalović
Mentor: dr Mirko Raković, redovni profesor

## O projektu

Korisnik nosi naočare za praćenje pogleda i bira predmet sa radne površine tako što ga posmatra. Kada se pogled zadrži na istom predmetu duže od zadatog praga, sistem prepoznaje o kojem je predmetu riječ, određuje njegov položaj i orijentaciju u prostoru i robotskoj ruci zadaje zadatak da ga uhvati i odloži u jednu od zona za odlaganje.

Sistem pripada grupi sistema zasnovanih na pogledu usmjerenom na objekat. Korisnik ne upravlja pokretima ruke, već samo pokazuje šta želi, a sve ostalo, od prepoznavanja predmeta do formiranja komande za hvatanje, sistem izvršava samostalno.

## Princip rada

Naočare *Pupil Core* daju dvije vrste podataka: položaj pogleda korisnika i sliku scene koju korisnik posmatra. Na slici scene model *YOLO11* prepoznaje predmete, a položaj pogleda se poredi sa okvirima prepoznatih predmeta kako bi se odredilo koji predmet korisnik trenutno gleda. Da prirodno pomjeranje pogleda ne bi izazivalo neželjene izbore, izbor se potvrđuje tek kada pogled ostane na istom predmetu 2 s.

Okvir dobijen detekcijom nije dovoljno precizan za hvatanje, pa se poza predmeta određuje pomoću *ArUco* markera postavljenog na njega. Dobijena poza izražena je u koordinatnom sistemu kamere, koja se pomjera zajedno sa korisnikovom glavom. Zato se položaj predmeta najprije izražava u odnosu na radnu površinu, čiji su uglovi označeni referentnim markerima, a zatim se jednom izmjerenom transformacijom prevodi u koordinatni sistem robota. Na taj način pomjeranje glave ne utiče na rezultat i nije potrebno poznavati položaj kamere u odnosu na robotsku ruku.

Na kraju se formira komanda za hvatanje, koja se šalje kontroleru robotske ruke *xArm 7*. Ruka prilazi predmetu odozgo, hvata ga, prenosi i odlaže u prvu slobodnu zonu za odlaganje.

## Korišćena oprema i biblioteke

Oprema:
- naočare za praćenje pogleda *Pupil Core*, sa programom *Pupil Capture*;
- robotska ruka *xArm 7* sa hvataljkom *xArm Gripper*;
- odštampani *ArUco* markeri (rječnik `DICT_4X4_100`) - četiri referentna markera u uglovima radne površine (ID 21-24) i po jedan marker na svakom predmetu (ID 1-3).

Biblioteke:
- *OpenCV* - obrada slike, detekcija *ArUco* markera i procjena njihove poze;
- *Ultralytics* - model *YOLO11*, dodatno obučen za tri predmeta korišćena u radu;
- *NumPy* - rad sa matricama transformacije;
- *ZeroMQ* i *MessagePack* - komunikacija sa programom *Pupil Capture*;
- *xArm Python SDK* - upravljanje robotskom rukom.

## Sadržaj repozitorijuma

Aplikacija je podijeljena na module, tako da svaki modul obavlja jedan zaokružen dio zadatka:

- `pupil_core.py` - glavni program koji povezuje sve module i sadrži glavnu petlju aplikacije;
- `config.py` - sve podesive vrijednosti sistema na jednom mjestu: pragovi pouzdanosti, trajanje zadržavanja pogleda, identifikatori i dimenzije markera, geometrija radne površine, parametri kamere, dimenzije predmeta i režim rada ruke;
- `pupilcore.py` - veza sa programom *Pupil Capture*, prijem podataka o pogledu i slike scene;
- `detection.py` - prevođenje tačke pogleda u koordinate slike i određivanje predmeta na koji je pogled usmjeren;
- `dwell.py` - mjerenje vremena zadržavanja pogleda i prikaz trake napretka;
- `aruco.py` - detekcija markera, procjena poze predmeta i radne površine, kratkotrajno pamćenje posljednjih detekcija;
- `transformacije.py` - rad sa homogenim matricama transformacije i usrednjavanje poza;
- `xarm_kontrola.py` - veza sa kontrolerom robotske ruke, upravljanje hvataljkom i izvršavanje ciklusa hvatanja.

Pored same aplikacije, u repozitorijumu se nalaze:

- `kalibracija_umeyama.py` - program za određivanje transformacije između radne površine i robota na osnovu parova tačaka;
- `best.pt` - obučeni model za detekciju predmeta (šolja, džojstik, plišana igračka);
- `charuco_kalibracija.json` i `robot_table_kalibracija_umeyama.json` - rezultati kalibracije kamere i kalibracije radne površine i robota;

## Pokretanje

Aplikacija je razvijana i ispitivana sa verzijom *Python* 3.9. Potrebne biblioteke instaliraju se naredbom:

```
pip install -r requirements.txt
pip install xarm-python-sdk
```

Prije pokretanja potrebno je pokrenuti *Pupil Capture* i izvršiti kalibraciju pogleda. Ako se koristi robotska ruka, računar mora biti u istoj mreži sa kontrolerom, a njegova adresa upisana u `xarm_kontrola.py`.

Režim rada ruke bira se u `config.py` promjenom vrijednosti `ARM_MODE`:

- `MOCK` - ruka se ne pokreće, komande se samo ispisuju u komandnoj liniji, pa se ostatak sistema može isprobati i bez robota;
- `HOVER` - ruka dolazi iznad predmeta i zaustavlja se, bez hvatanja;
- `HVAT` - izvršava se kompletan ciklus hvatanja i odlaganja.

Aplikacija se zatim pokreće naredbom:

```
python pupil_core.py
```

Tokom rada prikazuje se slika sa kamere scene, na kojoj su označeni prepoznati predmeti, detektovani markeri, trenutna tačka pogleda i napredak zadržavanja pogleda. Taster `r` oslobađa zone za odlaganje, a taster `q` zaustavlja aplikaciju.

## Prilagođavanje drugoj postavci

Vrijednosti u `config.py` odgovaraju postavci korišćenoj u laboratoriji. Pri promjeni opreme ili rasporeda potrebno je ponovo odrediti:

- unutrašnje parametre kamere scene (važe samo za rezoluciju 1280x720);
- dimenzije odštampanih markera i položaje referentnih markera na radnoj površini;
- transformaciju između radne površine i robota (`T_ROBOT_TABLE`), pokretanjem programa `kalibracija_umeyama.py`;
- dimenzije predmeta, kao i položaje zona za odlaganje.

Novi predmet zahtijeva dodatno obučavanje modela za detekciju, marker postavljen na predmet i unos njegovih dimenzija u `config.py`.

## Rezultati

Obučeni model za detekciju predmeta ostvaruje *mAP50* od 0,991. Pri pragu zadržavanja pogleda od 2 s uspješnost izbora predmeta iznosila je 95 %. U završnom ispitivanju kompletnog sistema uspješno je izvedeno 46 od 60 ciklusa hvatanja i odlaganja, odnosno približno 77 %. Detaljan opis ispitivanja i analiza rezultata nalaze se u tekstu rada.
