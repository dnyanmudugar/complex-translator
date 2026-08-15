import os
import pandas as pd

def expand_parallel_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "your_dataset.csv")

    # Expanded parallel corpus (English -> French translation samples)
    data = {
        "source_text": [
            # Greetings & Basics
            "hello computer", "hello world", "good morning friend", "good night everyone",
            "thank you very much", "please help me", "yes of course", "no problem",
            
            # Simple Actions & Descriptions
            "machine translation model", "the artificial intelligence system", "this is a neural network",
            "the system processes data", "we are building a translator", "i love writing python code",
            
            # Conversational
            "how are you", "i am doing well", "what is your name", "my name is computer",
            "where do you live", "i live in the cloud", "what time is it", "it is late",
            
            # Short Sentences
            "the dog barked", "the cat slept", "i see the screen", "you read the text",
            "she wrote a book", "he likes machine learning", "the program runs fast", "the model learns quick",
            
            # Expanded Tech Terminology
            "open the application", "save the project file", "run the automated test suite",
            "the processing unit is fast", "gpu acceleration is active", "check the terminal output"
        ],
        "target_text": [
            # Greetings & Basics
            "bonjour ordinateur", "bonjour le monde", "bonjour mon ami", "bonne nuit tout le monde",
            "merci beaucoup", "s il vous plait aidez moi", "oui bien sur", "pas de probleme",
            
            # Simple Actions & Descriptions
            "modele de traduction automatique", "le systeme d intelligence artificielle", "ceci est un reseau de neurones",
            "le systeme traite les donnees", "nous construisons un traducteur", "j adore ecrire du code python",
            
            # Conversational
            "comment allez vous", "je vais bien", "quel est votre nom", "mon nom est ordinateur",
            "ou habitez vous", "je vis dans le nuage", "quelle heure est il", "il est tard",
            
            # Short Sentences
            "le chien a aboye", "le chat a dormi", "je vois l ecran", "vous lisez le texte",
            "elle a ecrit un livre", "il aime l apprentissage automatique", "le programme s execute rapidement", "le modele apprend vite",
            
            # Expanded Tech Terminology
            "ouvrez l application", "enregistrez le fichier du projet", "executez la suite de tests automatises",
            "l unite de traitement est rapide", "l acceleration gpu est active", "verifiez la sortie du terminal"
        ]
    }

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Dataset successfully expanded! Total parallel sample pairs: {len(df)}")
    print(f"File updated at: {dataset_path}")

if __name__ == "__main__":
    expand_parallel_dataset()
