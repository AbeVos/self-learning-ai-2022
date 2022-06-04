# Huiswerk opdracht 4

## Omschrijving

Kies een machine learning algoritme uit.
Dit mag een algoritme zijn dat we tijdens de colleges behandelt hebben, maar je mag ook zelf iets kiezen.

Probeer een implementatie van dit algoritme werkend te krijgen.
Hier voor mag je zelf een implementatie schrijven, een tutorial volgen of een kant en klare implementatie gebruiken.
Sommige algoritmes zullen niet praktisch zijn om lokaal te draaien (zoals GPT-3), als dit het geval is kan je een online implementatie gebruiken via een API of een implementatie in Google Colab.
Let er op dat je voor deze opdracht dus niet in Python hoeft te werken.

Probeer zelf een toepassing te maken met het algoritme.
Dit kan een webtool zijn om verschillende inputs te proberen op het model, een eigen (simpele) game waarop je een reinforcement learning algoritme laat draaien of een model dat je traint op je eigen dataset.
Natuurlijk word je aangemoedigd om iets creatiefs te bedenken!

De focus van deze opdracht is om creatieve toepassingen te bedenken voor zelf-lerende AI in games.
Verdoe daarom niet te veel tijd aan het schrijven van je eigen implementaties, maar ga op zoek naar kant en klare stukken software.

## Deliverables

De deliverables voor de opdracht zijn:

- Een verslag (1-2 A4) waarin je de volgende vragen beantwoordt:
	- Welk algoritme heb je gebruikt? (Geef ook een korte samenvatting van hoe het werkt)
	- Hoe gebruik je dit algoritme om een nieuwe toepassing te maken?
	- Waarom werkt dit algoritme beter voor deze taak dan andere algoritmes?
	- Hoe zou deze toepassing gebruikt kunnen worden in een game/interactieve applicatie?
	- Welke bronnen (voor code en/of data) heb je gebruikt?
- Een van de volgende:
	- Een korte video (ongeveer 1 minuut) van je toepassing in actie.
	- Of een reeks screenshots (met uitleg) van de resultaten die je uit je methode hebt verkregen.

## Deadlines

De deadline voor het verslag en de video is 24 juni.
Zorg dat je _uiterlijk_ 9 juni in ieder geval het volgende op orde hebt:

- Je gekozen algoritme/model.
- De toepassing die je wilt bouwen.
- Benodigde frameworks/implementaties van modellen/datasets om deze toepassing te bouwen.
- De mensen met wie je samen werkt (je mag ook alleen werken).

Stuur deze informatie in een mail naar abe_vos@msn.com.

## Inspiratie

Om je op weg te helpen met een idee, zijn hier wat voorbeelden van eventuele projecten.
Deze voorbeelden kan je kiezen en/of aanpassen voor je eigen project.

### Custom environment

Schrijf je eigen OpenAI gym environment en train een RL algoritme.

- [Custom environment in Python](https://blog.paperspace.com/creating-custom-environments-openai-gym/)
- [Gym environments in Godot](https://github.com/HugoTini/GymGodot)
- [Baseline implementaties van RL algoritmes](https://github.com/Stable-Baselines-Team/stable-baselines/)

### Face editor

Gebruik een generatief model voor afbeeldingen van gezichten.
Maak een applicatie waarin waarin een gebruiker met behulp van sliders een fotorealistisch gezicht kan maken.

- [Eigenfaces tutorial](https://scipy-lectures.org/packages/scikit-learn/auto_examples/plot_eigenfaces.html)

### Texture upscaler

Maak een tool die de resolutie van textures kan verhogen.

- [Superresolutie in Pytorch](https://github.com/twtygqyy/pytorch-SRResNet)

### Texture Style Transfer

Mod een game met textures die er uit zien alsof ze door een impressionistische schilder zijn gemaakt.

- [Style Transfer in Pytorch](https://towardsdatascience.com/implementing-neural-style-transfer-using-pytorch-fd8d43fb7bfa)

### Speech2Control

Gebruik een spraak-naar-tekst algoritme om met gesproken commando's een game aan te sturen.

- [Voice to Text in Javascript](https://www.assemblyai.com/blog/voice-to-text-javascript/)
