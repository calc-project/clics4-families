import igraph
from pathlib import Path
from pycldf import Dataset
import networkx as nx
from collections import defaultdict
from pyclics.util import write_gml
from tabulate import tabulate
from lingpy import flat_upgma, Pairwise
from tqdm import tqdm as progressbar
from clldutils.misc import slug
import csv
import sys


csv.field_size_limit(10000000)


def dist(w_a, w_b):
    pair = Pairwise(w_a, w_b)
    pair.align(distance=True)
    return pair.alignments[0][-1]


def get_cognates(varieties, languages, words):
    matrix = [[0 for w in words] for w in words]
    for i, word_a in enumerate(words):
        for j, word_b in enumerate(words):
            if i < j:
                matrix[i][j] = matrix[j][i] = dist(word_a, word_b)
    cluster = flat_upgma(0.45, matrix, varieties)
    return len(cluster)


CLICSPATH = "clics4-0.2"


ds1 = Dataset.from_metadata(Path(CLICSPATH) / "cldf/StructureDataset-metadata.json")
ds2 = Dataset.from_metadata(Path(CLICSPATH) / "cldf/Wordlist-metadata.json")

edges = ds1.objects("ParameterTable")
languages = ds1.objects("LanguageTable")
concepts = ds2.objects("ParameterTable")
forms = ds2.objects("FormTable")

print("[i] loaded all tables")

# dtermine language families
families = defaultdict(lambda : defaultdict(list))
for language in languages:
    families[language.data["Family"]][language.cldf.glottocode] += [language.id]

family_table = sorted(
        [[slug(k), len(v)] for k, v in families.items()],
        key=lambda x: x[1], 
        reverse=True)[:30]

networks = {}
for row in family_table:
    networks[row[0]] = nx.Graph()
    for concept in concepts:
        networks[row[0]].add_node(
            concept.data["Concepticon_Gloss"],
            varieties=[],
            languages=[],
            words=[]
            )

for edge in edges:
    source, target = edge.data["Source_Concept"], edge.data["Target_Concept"]
    form_idxs = [edge.data["Forms"][0] + "/" + edge.data["Forms"][2]] + edge.data["Forms"][3:]
    all_forms = []
    for fidx in form_idxs:
        all_forms += [forms[fidx.split("/")[0][7:]].cldf.segments]

    for variety, language, family, form in zip(
            edge.data["Varieties"],
            edge.data["Languages"],
            edge.data["Families"],
            all_forms):
        if family in networks:
            networks[family].nodes[source]["varieties"] += [variety]
            networks[family].nodes[source]["languages"] += [language]
            networks[family].nodes[target]["varieties"] += [variety]
            networks[family].nodes[target]["languages"] += [language]
            networks[family].nodes[target]["words"] += [" ".join(form)]

            try:
                networks[family][source][target]["varieties"] += [variety]
                networks[family][source][target]["languages"] += [language]
                networks[family][source][target]["words"] += [" ".join(form)]
            except:
                networks[family].add_edge(
                        source,
                        target,
                        varieties=[variety],
                        languages=[language],
                        words=[" ".join(form)],
                        )



for family, network in networks.items():
    print("[i] calculating data for {0}".format(family))
    for node, data in network.nodes(data=True):
        data["Variety_Count"] = len(data["varieties"])
        data["Language_Count"] = len(set(data["languages"]))
    for node_a, node_b, data in progressbar(network.edges(data=True), desc="refining edges"):
        data["Variety_Count"] = len(data["varieties"])
        data["Language_Count"] = len(set(data["languages"]))
        data["Cognate_Count"] = get_cognates(
                data["varieties"], 
                data["languages"], 
                data["words"])

    write_gml(network, Path("graphs") / "{0}.gml".format(family))

with open("overview.md", "w") as f:
    f.write(tabulate(family_table, headers=["Family", "Languages"],
            tablefmt="pipe"))
