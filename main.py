import spacy
import networkx as nx

# Load English tokenizer, tagger, parser, NER
nlp = spacy.load("en_core_web_sm")

# Sample text
text = """
Barack Obama was born in Hawaii. He was the 44th President of the United States.
Michelle Obama is his wife. They have two daughters, Sasha and Malia.
"""

# Process text with spaCy
doc = nlp(text)

# Extract entities and relationships
entities = [ent.text for ent in doc.ents if ent.label_ in ['PERSON', 'GPE']]
relationships = []
for sent in doc.sents:
    for i in sent:
        if i.dep_ in ('attr', 'dobj'):
            subject = [w for w in i.head.lefts if w.dep_ == 'nsubj']
            if subject:
                subject = subject[0]
                relationships.append((subject, i))

# Construct graph
G = nx.Graph()
G.add_nodes_from(entities)
G.add_edges_from(relationships)

# Visualize the graph
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
nx.draw(G, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold')
plt.show()
