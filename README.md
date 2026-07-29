# MuJoCo → Isaac: Headless-100er-Studie und Visualisierung (Version 2.0.1)

Version 2.0.1 setzt auf dem funktionierenden v1.9-Hotfix auf. Das physikalische Grundmodell
bleibt bytegleich: Geometrien, Kollisionskörper, FixedJoint, Bewegungsbahn,
Feder-/Dämpferwerte, Rotation und Kontaktparameter wurden nicht verändert.
Neu sind ein verbindlicher sichtbarer Präsentationsmodus, feste 16:9-Kameras
und ein automatischer PNG-Export für Start- und Endzustand.

Dieses Projekt uebertraegt die bereitgestellte MuJoCo-Basis in eine
konfigurierbare Isaac-Sim-Struktur fuer zwei sichtbare beziehungsweise bis zu
100 gleichzeitige Montageversuche. Die Isaac-Sim-Oberfläche dient als
Visualisierung; eine eigene Bedien-GUI wird nicht benötigt.

## Umgesetzter Stand

- sichtbarer `--visual-demo`-Modus mit KET12 und USB
- zwei UR5e-CAD-Darstellungen mit bewegten Finray-Greifern
- feste Ansichten `overview`, `ket12` und `usb`
- sechs automatische PNGs: Start und Ende aus allen drei Ansichten
- standardmäßig 1920 × 1080 mit `RaytracedLighting`
- optionales `PathTracing`
- Rendering-Manifest mit Kamera, Auflösung, Dateigröße und SHA-256
- verbindlicher `--headless-study-100`-Modus
- exakt 50 KET12- und 50 USB-Umgebungen in einer PhysX-Szene
- 50 Parametergruppen mit je zwei Wiederholungen
- Vorabprüfung des 100er-Plans ohne Isaac Sim
- SHA-256-Schutz für 15 physikalisch relevante v1.8-Dateien
- Laufmanifest mit Status `RUNNING`, `COMPLETED` oder `FAILED`
- automatische Qualitätsprüfung der 100 Ergebniszeilen
- JSON-/Markdown-Gesamtbericht und Aufgabenstatistik
- frei wählbarer Ergebnisordner über `--output-dir`
- echtes MuJoCo–Isaac-Mapping aus `sim.py` und `robot_15.xml`
- KET12 und USB als zwei parallele Montageaufgaben
- pro Umgebung eigene Steifigkeit, Dämpfung, Reibung und Buchsenversatz
- 100er-Plan: 2 Aufgaben × 5 Steifigkeiten × 5 Versätze × 2 Wiederholungen
- D6-Ersatzmodell mit drei Translationen und drei Rotationen
- kombinierte Greiferrahmenmasse 0,004 kg aus zwei MuJoCo-Fingern à 0,002 kg
- keine gesperrte Rotation in der MuJoCo-Baseline
- headless Ergebnisaggregation als Einzel- und Gruppen-CSV
- viewerloser MuJoCo-Einzelreferenzlauf und automatischer Vergleich
- originale Finray-Aussenmesh als Visualisierung
- originale MuJoCo-Greiferhalter `gripper_mount_left/right.obj`
- exakte MJCF-Transformationen der Halter und Finray-Finger
- aufgabenspezifische MuJoCo-Schließpositionen: KET12 4,7 mm, USB 4,8 mm
- originale USB-Male-/USB-Female-Visualmeshes
- originales NIST-Board als KET12-Aufnahme
- originale UR5e-Linkmeshes mit aufgabenspezifischer IK-Debugpose
- Zwei-Umgebungs-Debugansicht fuer die Zwischenpraesentation
- stabiler FixedJoint-Griff des bereits aufgenommenen Steckers

## Wesentliche Korrektur gegenueber Schritt 5

MuJoCo sperrt die Rotationen nicht. Die Laufzeitwerte sind:

```text
Translation lokal:  x=1.2 N/mm, y=52.4 N/mm, z=100 N/mm
Rotation lokal:     x=100, y=100, z=1 N*m/rad
Dämpfung:            Translation 10 N*s/m, Rotation 1 N*m*s/rad
```

Nach dem Koordinatenmapping `mj(x,y,z) -> isaac(X,Z,Y)` gilt:

```text
Isaac transX/transY/transZ = 1 200 / 100 000 / 52 400 N/m
Isaac rotX/rotY/rotZ       = 100 / 1 / 100 N*m/rad
```

Die Rotation um die Isaac-Einfuehrachse Z bleibt also beweglich, wird aber
mit 100 N*m/rad gefuehrt. Eine gesperrte Z-Rotation ist nur ein spaeterer
Ablationsfall. Da USD Winkel-Drives pro Grad beschreibt, rechnet der Code die
Rotationssteifigkeit und -daempfung beim Schreiben des D6-Joints mit
`pi/180` von pro Radiant auf pro Grad um.

## Start

Sichtbare Isaac-Demo mit automatischen Renderings:

```bash
cd /home/aut_isaac/Downloads/isaac_mujoco_parallel_v2_0_1_visualization_fixed
chmod +x run_visual_demo.sh
./run_visual_demo.sh
```

Oder direkt:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py \
  --visual-demo \
  --output-dir results/visual_demo_run1
```

Die vollständige Rendering-Anleitung steht in
`docs/VISUALIZATION.md`.

Vorabprüfung ohne Isaac Sim:

```bash
python3 tools/prepare_headless_study.py
python3 -m unittest discover -s tests -v
```

Verbindlicher 100er-Lauf:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py \
  --headless-study-100
```

Alternativ über das Startskript:

```bash
chmod +x run_headless_100.sh
./run_headless_100.sh
```

Falls Isaac Sim an einem anderen Ort installiert ist:

```bash
ISAAC_SIM_PYTHON=/anderer/pfad/python.sh ./run_headless_100.sh
```

Vorhandene Laufdaten werden nicht stillschweigend überschrieben. Für eine
bewusste Wiederholung im gleichen Ausgabeordner:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py \
  --headless-study-100 \
  --overwrite-results
```

Zwei Aufgaben sichtbar:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py --visual-demo
```

Die beiden UR5e stehen dabei 1,40 m auseinander. Die Standardkamera zeigt
beide Roboter. Fuer eine Montage-Nahansicht kann derselbe Lauf so gestartet
werden:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py \
  --debug-two-envs \
  --camera-view ket12

/home/aut_isaac/isaac_sim/python.sh main.py \
  --debug-two-envs \
  --camera-view usb
```

Der Start zeigt im Terminal sechs Setup-Phasen. Die Montageobjekte werden
bereits beleuchtet und mit einer festen Projektkamera dargestellt, bevor die
umfangreicheren UR5e-CAD-Meshes geladen werden. Während des CAD-Imports bleibt
das Isaac-Fenster ansprechbar.

Nach dem 7,5 Sekunden langen Versuch bleibt die Debugansicht am Endzustand
offen. Sie wird erst geschlossen, wenn das Isaac-Sim-Fenster geschlossen wird.

Freier Headless-Lauf mit eigener Umgebungszahl:

```bash
/home/aut_isaac/isaac_sim/python.sh main.py \
  --headless \
  --num-envs 100
```

Einzelvergleich:

```bash
python3 tools/run_mujoco_single_trial.py

/home/aut_isaac/isaac_sim/python.sh main.py \
  --headless \
  --single-validation \
  --task KET12 \
  --physics-dt 5e-5

python3 tools/compare_validation.py \
  results/mujoco_single_trial.csv \
  results/environment_results.csv
```

## Wo Parameter geaendert werden

Nur `project_config.py`:

- `MujocoCompliance` und `IsaacCompliance`: 6D-Nachgiebigkeit
- `ContactConfig`: Reibung und compliant contact
- `TASKS`: Geometrie, Greifweite, TCP-Abstand, Suche, Einsetztiefe und
  Erfolgsschwellen
- `StudyConfig`: Aufgaben, Steifigkeiten, Versätze, Wiederholungen

`parameter_sweep.py` weist jedem Environment einen eigenen Parametersatz zu.

## Ausgaben

```text
results/headless_100/parameter_plan_100.csv
results/headless_100/parameter_plan_check.json
results/headless_100/physics_baseline_check.json
results/headless_100/environment_results.csv
results/headless_100/aggregate_results.csv
results/headless_100/study_summary.json
results/headless_100/automatic_evaluation.json
results/headless_100/automatic_evaluation.md
results/headless_100/run_manifest.json
```

`SUCCESS` und die vier physikalischen Fehlerklassen sind gültige
Studienergebnisse. Das Qualitätsgate schlägt nur bei fehlenden oder doppelten
Umgebungen, unbekannten Ergebnisklassen, nicht-endlichen Messwerten oder
`INVALID_SIMULATION_STATE` fehl.

## Technische Entscheidung zu den CAD-Assets

Die Finray-Aussenform, die beiden Greiferhalter, USB Male/Female, das
NIST-Board und die UR5e-Linkmeshes stammen direkt aus dem bereitgestellten
MuJoCo-Projekt. Ihre genauen Dateinamen sind in `ASSET_SOURCES.md`
dokumentiert. Fuer die 100er-Physik bleiben die Kollisionskoerper bewusst
einfach. Damit wird in Version 1.8 nur die sichtbare Geometrie originalgetreuer,
nicht ungeprueft das Kontaktmodell verändert. Das visuelle Finray-Netz wird
nicht als deformierbares Rippenmodell simuliert; sein mechanisches Verhalten
wird durch das 6D-Feder-Dämpfer-Ersatzmodell beschrieben. Der UR5e ist in der
Debugansicht eine CAD-/Strukturvisualisierung. Seine Basis bleibt auf der
Werkebene; fuer
jeden Bahnwegpunkt ist eine aufgabenspezifische IK-Pose hinterlegt. Die
Gelenkwinkel werden zeitgleich zur TCP-Bahn interpoliert, sodass Arm, TCP und
bewegter Finray-Greifer sichtbar zusammenbleiben. Die doppelten
Finray-Geometrien des MJCF-Imports werden ausgeblendet. Der sichtbare
Finray-Greifer sitzt nur am bewegten Ersatzgreiferrahmen. Die TCP-Bahn bleibt
vorgegeben, damit nicht gleichzeitig Roboterregler und Nachgiebigkeit variiert
werden.

Die Suchbewegung selbst bleibt aus MuJoCo uebernommen. Danach folgt eine
explizite Zentrier- und Setzphase. Sie war in v1.2 nicht vorhanden; dadurch
endeten die beiden Nullversatz-Debuglaeufe geometrisch noch 5,1 mm
beziehungsweise 11,2 mm neben der Buchsenmitte und konnten nicht als Erfolg
klassifiziert werden.

## Physikalische Abgrenzung des Greifens

Der Stecker startet als bereits gegriffenes Bauteil und ist mit einem
FixedJoint am nachgiebigen Greifrahmen befestigt. Das verhindert, dass eine
ungepruefte 0,1-mm-Kontaktueberdeckung der vereinfachten Pads beim
Simulationsstart vom PhysX-Solver aufgeloest wird und den Stecker
herausschleudert. Die untersuchte Nachgiebigkeit bleibt vollstaendig im
6D-D6-Modell zwischen Soll-TCP und Greifrahmen erhalten. Ein spaeterer
Greifverlustversuch erfordert dagegen ein separat kalibriertes Finger-/
Kontaktmodell und ist nicht Teil dieser Baseline.

Beide Finray-Visualmeshes und beide Greiferhalter verwenden ihre
Transformationen direkt aus `robot_15.xml`. Die geschlossenen Slide-Joints
stehen wie in `sim.py` bei KET12 auf ±4,7 mm und bei USB auf ±4,8 mm. Im
berechneten Endzustand besitzt der tiefste Finray-Punkt 9,743 mm Abstand zur
Oberseite des NIST-Boards (KET12) und 17,760 mm zur USB-Buchsenoberkante.
Bei KET12 bleibt der einzusetzende Stecker ein Quader, weil auch Richards
MuJoCo-Referenz KET12 als MuJoCo-Box und nicht als separates CAD-Mesh
modelliert.

## Dokumentation

- `docs/MUJOCO_ISAAC_MAPPING.md`: jede direkte, umgerechnete oder zu
  kalibrierende Groesse
- `docs/VALIDATION.md`: reproduzierbarer Einzelvergleich
- `docs/ARCHITECTURE.md`: Programmstruktur fuer die Zwischenpraesentation
- `ASSET_SOURCES.md`: verwendete MuJoCo-Visualassets und Abgrenzung zu GitHub

## Laufzeit-Hinweis

Die Python-Kernlogik ist ohne Isaac testbar. Ein echter PhysX-Lauf muss mit
der Isaac-Sim-Installation auf Artemis ausgefuehrt werden. Vor der
100er-Auswertung ist der Einzelvergleich mit `5e-5 s` verpflichtend; danach
wird die schnellere Studien-Schrittweite `1/1000 s` nur verwendet, wenn ein
Zeitschritt-Konvergenztest keine relevante Ergebnisänderung zeigt.

Eine genaue Bedienfolge und die Bedeutung aller Ausgaben stehen in
`docs/HEADLESS_100.md`.
