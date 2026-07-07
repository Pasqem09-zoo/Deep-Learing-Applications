Lab 3 - Deep Reinforcement Learning

Questo progetto contiene una versione modulare di REINFORCE e gli esperimenti del lab.
Il notebook e' tenuto solo come riferimento per le indicazioni del prof.

Struttura attuale
lab3/
├── main.py
├── config.py
├── src/
│   ├── networks.py
│   ├── reinforce.py
│   └── utils.py
├── experiments/
│   ├── DLA-Lab3-cartpole.ipynb
│   └── DLA-Lab3-DRL.ipynb
├── checkpoints/
├── requirements.txt
└── README.md

Obiettivi
- Far funzionare REINFORCE su CartPole.
- Aggiungere checkpoint e valutazione periodica (media reward e lunghezza episodio).
- Estendere la base di codice per altri esercizi del lab.

How to run
1) Installa i requirements
	pip install -r lab3/requirements.txt

2) Avvia il training
	python lab3/main.py

Note
- I parametri principali sono in config.py (tutti in maiuscolo).
- I checkpoint finiscono in lab3/checkpoints/.


