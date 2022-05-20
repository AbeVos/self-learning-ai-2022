# Huiswerk Opdracht 3

Deze huiswerkopdracht bevat vragen en programmeeropdrachten.
Bij elke opdracht staat het aantal punten dat er voor te verdienen is.
Iedere student moet de opdrachten zelfstandig uitvoeren.
Volg de volgende stappen bij het inleveren van de opdrachten:

- Maak een map met daarin:
  - Een PDF-document met alle vragen en antwoorden.
  - Je Python-bestanden en evt. bijgeleverde scripts (die zullen nodig zijn om de code uit te voeren).
- Comprimeer de map als een zip-bestand met de naam `HW3-{voornaam}_{achternaam}.zip`.
- Stuur het bestand op naar abe_vos@msn.com met het onderwerp `HW3 {voornaam} {achternaam}`.

De deadline is donderdag 2 juni om 23:59.
Vragen over het huiswerk kun je sturen naar abe_vos@msn.com.

## Functie benadering

In deze opdracht gaan we Sarsa met een lineaire Q-functie trainen om de CartPole omgeving op te lossen.

### CartPole omgeving

In de CartPole omgeving moet de agent leren een stok te balanceren door een kar naar links en rechts te bewegen.
Net zoals eerdere omgevingen, kunnen we deze als volgt aanmaken:

```python
env = gym.make("CartPole-v1")
```

De staat omschrijft de huidige positie en snelheid van de kar en de hoek en draaisnelheid van de stok.
De omgeving heeft twee acties; links en rechts.
De episode eindigt wanneer de kar te veel opzij schuift of wanneer de stok omvalt (exacte grenswaardes staan in de documentatie).
Elk frame waarin de stok overeind wordt gehouden levert een beloning van 1 op.

In de [documentatie](https://www.gymlibrary.ml/environments/classic_control/cart_pole/) staan meer details over deze omgeving.

### Lineaire Q-functie (1 punt)

Onze Q-functie gaat nu niet meer een dictionary zijn, maar een lineaire parametrische functie.
Dit is een functie die het dot-product tussen een parameter-vector en staat-vector berekent.
Het dot-product voor twee vectoren kan als volgt worden geschreven in Python:

```python
A = [1, 2, 3]
B = [3, 2, 1]
result = 0

for a, b in zip(A, B):  # Loop door a en b tegelijk
	result += a * b
```

We kunnen ook numpy `array`s gebruiken als vectoren Python:

```python
import numpy as np
a = np.array([1, 2, 3])  # Zet lijst om in vector
b = np.zeros(5)  # Maak vector met 5 nullen
c = np.ones(5)  # Maak vector met 5 enen
d = np.arange(5)  # Maak vector van [0, 1, 2, 3, 4]
```

Berekeningen op reeksen waar we normaal gesproken for-loops voor nodig hebben, kunnen we nu met simpele operators doen.

```numpy
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])

a + b  # Vector met de som van corresponderende elementen
a * b  # Vector met product van corresponderende elementen
5 * a  # Vermenigvuldig alle elementen met 5
a @ b  # Dot-product
```

We gaan onze Q-functie implementeren door een lineaire voorspeller te maken voor elke aparte actie.
Gebruik onderstaande opzet om de Q-functie te implementeren:

```python
def q_value(state, action, params):
	# JOUW CODE HIER
	# Bereken de q-waarde voor `action` in `state`.
	pass
```

Hint: in de bovenstaande code is `state` een numpy array, `action` een integer en `params` een matrix.
In numpy is een matrix een array van arrays:

```python
params = np.zeros((2, 6))
```

Oftewel, elke rij in de matrix is een vector met parameters.

### Activatiefunctie (2 punten)

Zoals we eerder gezien hebben, is het in neurale netwerken gebruikelijk om na elke laag een non-lineaire activatie functie te gebruiken, zoals de logistieke sigmoid functie `1 / (1 + exp(-x)`.
Aangezien onze lineaire voorspeller in feite een neuraal netwerk is met 1 laag en een paar output nodes, zouden we hier ook een activatie functie voor kunnen gebruiken.

Waarom willen we voor onze Q-functie _geen_ activatie functie gebruiken?
Wat voor waardes willen we dat onze parametrische functie kan voorspellen en hoe verhouden die zich tot de waardes die een activatie functie als de sigmoid kunnen voorspellen?

### Epsilon Greedy (2 punten)

Door onze nieuwe Q-functie is de implementatie van het beleid ook iets anders geworden.
Wanneer we de optimale actie willen vinden, moeten we nu de Q-waarde voor elke actie in de huidige staat berekenen en de actie met de hoogste waarde kiezen.

```python
def egreedy_policy(params, state, epsilon):
	n_actions = len(params)

	if random.random() < epsilon:
		return random.randint(0, n_actions - 1)
	else:
		# JOUW CODE HIER
		# Bereken de Q-waarde voor elke actie in `state`.
		# Return de actie met de hoogste waarde.
		return 0
```

### Episodic semi-gradient Sarsa (4 punten)

Implementeer episodic semi-gradient Sarsa zoals we die in het college hebben behandelt.

```python
def semigrad_sarsa(env, num_episodes, learning_rate=1e-2, discount_factor=1.0, epsilon=0.1):
		nS = env.observation_space.shape[0]
    nA = env.action_space.nA
		params = np.zeros((nA, nS))
    rewards = []

    for episode in range(num_episodes):
        done = False
        total_reward = 0

        state = env.reset()
        action = egreedy_policy(params, state, epsilon)

        while not done:
            state_new, reward, done, _ = env.step(action)
            total_reward += reward

            # JOUW CODE HIER
            # Kies een nieuwe actie met het egreedy beleid.
            # Update `params` voor actie `action`.
            # Update `state` en `action` voor de volgende iteratie.

        rewards.append(total_reward)

        print(f"Episode {episode}, sum reward: {total_reward}")

    return params, rewards
```

Hint: voor de update stap hebben we de afgeleide van de Q-functie met betrekking tot de parameters nodig.
Voor een lineaire functie is die gelukkig erg makkelijk.
Als we de lineaire functie `q(a, b) = a @ b` hebben, dan is de afgeleide met betrekking tot `a` simpelweg `b`.

### Trainen (2 punten)

We gaan nu het algoritme op de CartPole omgeving trainen.
Dit algoritme is erg gevoelig voor verschillende waardes voor `num_episodes`, `learning_rate` en `epsilon`.
Probeer verschillende waardes uit en kijk of dit helpt om betere beloningen te krijgen.

Om een idee te krijgen van hoe goed de geleerde Q-functie werkt, moeten we een paar episodes uitvoeren met een _greedy_ beleid, oftewel kies altijd de actie met de hoogste Q-waarde.
De onderstaande code voert een episode uit met het greedy beleid.

```python
def evaluate(env, params, render=False):
	state = env.reset()
	action = egreedy_policy(params, state, 0)

	done = False
	total_reward = 0

	while not done:
		state, reward, done, _ = env.step(action)
		action = egreedy_policy(params, state, 0)

		total_reward += reward

		if render:
			env.render()

	return total_reward
```

Aangezien de totale beloning van de episode stochastisch is, moeten we meerdere episodes simuleren om een redelijke schatting te krijgen.

Bereken het gemiddelde van 100 evaluaties na het trainen.
Doe dit ook met een ongetrainde Q-functie (waar de parameters een vector met nullen zijn).
Wat voor scores krijg je hier uit?
Heeft het trainen voor een verbetering gezorgd?
Welke hyperparameters heb je gebruikt?

Hint: trainen met gradients wil soms een beetje instabiel zijn.
Wanneer het de eerste keer niet lukt, probeer je code dan nog een kaar keer uit te voeren en kijk of je verbetering zien.

## Augmented Random Search

### LunarLander omgeving

In de Lunar Lander omgeving moet de speler een maanlander veilig op het landingsplatform zetten.
Zie de [documentatie](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) voor details.
Lunar Lander gebruikt Box2d voor de implementatie.
Box2d is een 2d physics engine en kan worden geinstalleerd met pip:

```
pip install gym[box2d]
```

Waarschijnlijk moet je ook nog [swig.exe](https://www.swig.org/Doc1.3/Windows.html#Windows_installation) installeren.

### Parametrisch beleid (2 punten)

Bij random search methodes kunnen we direct het beleid trainen zonder een Q-functie bij te hoeven houden.
Als we weer uit gaan van een lineair beleid, kunnen we de parameters weer in een matrix stoppen.
Deze matrix heeft zoveel rijen als er elementen in de staat vector zijn en het aantal kolommen komt overeen met het aantal acties.
Ons beleid kiest een actie door de staat vector te transformeren met de parameter vector, dit levert een nieuwe vector op met hetzelfde aantal elementen als dat er acties zijn.
De index van de actie met de hoogste waarde, is de gekozen actie.

```python
def policy(state, params):
	# JOUW CODE HIER
	# Vermenigvuldig `state` en `params`, let op dat de volgorde belangrijk is bij matrix-vermenigvuldiging.
	# Vind de index van de actie met de hoogste score.
	return 0
```

Hint: Als je twee matrices vermenigvuldigt, is het belangrijk dat hun vormen kloppen.
Als we een matrix van formaat (a, b) (a rijen en b kolommen) vermenigvuldigen met een matrix van formaat (c, d), dan moeten b en c even groot zijn.
Deze vermenigvuldiging levert een matrix van (a, d) op.
Vectoren worden vaak beschouwd als matrices van formaat (1, n) of (n, 1).
Je kan het formaat van een numpy matrix/array bekijken met het `.shape` attribuut.

### Exploratie (2 punten)

Zoals je misschien al gezien hebt, is het beleid dat we hier gebruiken deterministisch.
Bij eerdere algoritmes zoals Sarsa en Q-learning hadden we een stochastisch beleid nodig voor het trainen zodat we alle staten blijven bezoeken.
Waarom is dat voor BRS en ARS niet nodig?
Hoe zorgen we ervoor dat we blijven exploreren met deze algoritmes?

### Rollout (1 punt)

Wanneer we een volledige simulatie uitvoeren om een totale beloning van een episode te genereren, noemen we dat ook wel een rollout.
Schrijf een functie genaamd `rollout(env, params, render=False)` die een volledige episode simuleert met het beleid dat we hierboven hebben geschreven en de som van alle geobserveerde beloningen teruggeeft.

Hint: gebruik de functie `evaluate` hierboven als een startpunt.

### Basic Random Search (4 punten)

```python
def brs(env, num_iters, lr=0.2, v=0.2, N=4):
	nS = env.observation_space.shape[0]
	nA = env.action_space.nA
	params = np.zeros((nA, nS))

	for iteration in range(num_iter):
		# JOUW CODE HIER.
		# Genereer de noise vectors.
		# Genereer de twee beloningen (met de `rollout` functie) voor elke noise vector en sla die samen met de bijbehorende noise vector op in een lijst.
		# Update de parameters.

		print(f"It: {iteration}, reward: {rollout(env, params)}")

	return params
```

Hint: je kan de 'noise' genereren met `np.random.randn(nA, nS)`.

### Trainen (3 punten)

Probeer het algoritme uit op de LunarLander omgeving.
Net als met Sarsa, kan het helpen om de hyperparameters wat aan te passen.

Vergelijk de gemiddelde beloningen voor en na het trainen weer over meerdere iteraties.
Welke hyperparameters heb je gebruikt?

Als je `rollout` aanroept op je getrainde model met het argument `render=True`, wordt de gesimuleerde episode weergeven.
Bekijk een stel van deze episodes, wat voor strategie heeft het algoritme geleerd?

Hint: dit algoritme kan wat langer nodig hebben om te trainen, daarom kan het veel tijd schelen om de geleerde parameters op te slaan in een bestand:

```python
np.save("trained_params.npy", params)
```

Lees het bestand weer uit in een ander script waar je naar hartenlust episodes kan produceren zonder eerst te hoeven trainen:

```python
params = np.load("trained_params.npy")
```

### Random Search vs Reinforcement Learning (3 punten)

[Het artikel](https://arxiv.org/pdf/1803.07055.pdf) dat ARS omschreef liet zien dat hun algoritme in een aantal benchmarks even goed werkt als geavanceerde RL algoritmes.
Aangezien ARS conceptueel vrij eenvoudig werkt, is dit erg opvallend.

De algoritmes die we in de vorige colleges hebben behandeld konden we onderverdelen aan de hand van eigenschappen zoals: on-/off-policy, on/offline, bootstrapping/Monte Carlo.
Hoe zou je ARS categoriseren volgens deze eigenschappen?
In wat voor situaties zou een RL algoritme beter werken dan ARS?

### Bonus: Augmentaties (5 punten)

We hebben gekeken naar drie aanpassingen van BRS om er ARS van te maken.
Kopieer de `brs` functie, noem die `ars` en probeer 1 of meer van de genoemde aanpassingen te implementeren.
Evalueer het algoritme weer net zoals we dat deden voor BRS en vergelijk de resultaten.
Zie je een verbetering?
Laat ook weer de gebruikte hyperparameters zien.

Als je BRS niet hebt kunnen implementeren, mag je in plaats van het implementeren van de verbeteringen ook een korte omschrijving in eigen woorden geven over waarom deze aanpassingen het algoritme beter zouden laten werken.
