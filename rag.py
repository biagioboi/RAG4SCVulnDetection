import re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

# === CONFIG ===
EMBEDDING_PATH = "./contract_embeddings.json"
EMBEDDING_MODEL = "hkunlp/instructor-xl"
VULN_INFO = "./algorand_vuln_info.json"

# === LOAD EMBEDDING FILE
df = pd.read_json(EMBEDDING_PATH)
df = df.to_dict()

# === LOAD EMBEDDING MODEL ===
print(f"Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)


def search_similarity(query: str, find_duplicates: bool = False) -> list:
    num_of_emb = len(df['embedding'])
    instruction = "Represent the following functional description of an Algorand smart contract, written in PyTeal:"
    query_emb = model.encode([instruction, query])

    embeddings = df['embedding']

    k_vet = VetOfMax(n_of_max=num_of_emb)

    for index in range(num_of_emb):
        e = embeddings[index]
        e = np.array(e)
        e = e.reshape(1, -1)
        query_emb = query_emb.reshape(1, -1)

        similarity = cosine_similarity(e, query_emb)

        k_vet.insert_sorted_desc({'index': index, 'num': f"{similarity[0][0]:.4f}"})

    entries = []
    for k in k_vet.vet_of_max:
        entry = get_contract_info(k['index'])
        entry['similarity'] = k['num']
        flag = True

        if not find_duplicates:
            for i, e in enumerate(entries):
                if e['vulnerability'] == entry['vulnerability']:
                    flag = False
                    if entry['similarity'] > e['similarity']:
                        entries[i] = entry
                        break
                    else:
                        break

        if flag:
            entries.append(entry)

    return entries


def get_contract_info(index: int):

    return {
        'vulnerability': df['vulnerability'][index],
        'code': df['code'][index],
        'vulnerable_part': df['vulnerable_part'][index],
        'id': df['id'][index],
    }


def map_vulnerability(name):
    mapping = {
        "Arbitrary delete": "arbitrary_delete",
        "Arbitrary update": "arbitrary_update",
        "Unchecked Asset Close To": "asset_close_to",
        "Unchecked Close Remainder To": "close_remainder_to",
        "Unchecked Rekey to": "rekey_to",
        "Unchecked Transaction Fee": "transaction_fee",
        "Unchecked Asset Receiver": "Unchecked_Asset_Receiver",
        "Unchecked Payment Receiver": "Unchecked_Payment_Receiver",
        "arbitrary_delete": "Arbitrary delete",
        "arbitrary_update": "Arbitrary update",
        "asset_close_to": "Unchecked Asset Close To",
        "close_remainder_to": "Unchecked Close Remainder To",
        "rekey_to": "Unchecked Rekey to",
        "transaction_fee": "Unchecked Transaction Fee",
        "Unchecked_Asset_Receiver": "Unchecked Asset Receiver",
        "Unchecked_Payment_Receiver": "Unchecked Payment Receiver",
        "no vuln": "Not Vulnerable"
    }

    return mapping.get(name, "Unknown")

def create_rag_checklist(rag_contracts: list[dict], num_relevant=3, with_few_shot=True):
    checklist = ""
    info_few_shot = """
    {num}. **{vulnerability}** ({score})
        - **Description:** {description}
        - **Preconditions:** {precondition}
        - **Security check to perform**: {check}
        - **Example Vulnerable Pattern: 
          ```pyteal
    {vuln_part}
          ```
      """
    info_zero_shot = """
    {num}. **{vulnerability}** ({score})
        - **Description:** {description}
        - **Preconditions:** {precondition}
        - **Security check to perform**: {check}
      """

    if with_few_shot:
        info = info_few_shot
    else:
        info = info_zero_shot

    info_no_desc = """
    {num}. **{vulnerability}** ({score})
    """
    with open("./vulnerability_info.json", 'r', encoding='utf-8') as f:
        content = json.load(f)
    for i, c in enumerate(rag_contracts):
        vuln = rag_contracts[i]['vulnerability']
        vuln_part = rag_contracts[i]['vulnerable_part']
        score = rag_contracts[i]['similarity']
        data = content[vuln]
        if i < num_relevant:
            checklist += info.format(
                num=i+1,
                score=score,
                vulnerability=data['name'],
                description=data['description'],
                precondition=data['precondition'],
                check=data['security_check'],
                vuln_part=vuln_part.strip(),
            )
        else:
            checklist += info_no_desc.format(
                num=i+1,
                score=score,
                vulnerability=data['name'],
            )
    return checklist

def make_prompt_context(rag_contracts: list[dict]):
    user_prompt = """
    ---
    Remember this critical information about probable vulnerabilities you could find:
    {context}
    DO NOT FORGET TO CHECK THE OTHER POSSIBLE VULNERABILITIES: {others}.
    ---
    Now perform a detailed security analysis of the following PyTeal smart contract.
    
    Contract Code:
    
    {code}
    """

    info = """
    - **Name:** {vulnerability}
    - **Description:** {description}
    - **Preconditions:** {precondition}
    - **Security check to perform: {check}
    - **Vulnerable Part: {vuln_part}
    
    """
    context = ""
    for i in range(3):
        vuln = rag_contracts[i]['vulnerability']
        vuln_part = rag_contracts[i]['vulnerable_part']
        rag_contracts.pop(i)
        with open("./algorand_vuln_info.json", 'r', encoding='utf-8') as f:
            content = json.load(f)
        for v in content['vulnerabilities']:
            if v['name'] == vuln:
                context += info.format(
                    vulnerability=map_vulnerability(vuln),
                    description=v['description'],
                    precondition=v['precondition'],
                    check=v['security_check']['goal'],
                    vuln_part=vuln_part
                )
    others = ""
    for v in rag_contracts:
        vuln = map_vulnerability(v['vulnerability'])
        others = f"{others}, {vuln}"

    return user_prompt.format(code="{code}", context=context, others=others.removeprefix(','))



VULNS = ["Arbitrary delete",
         "Arbitrary Application Deletion",
         "Arbitrary Application update",
         "Arbitrary update",
         "Unchecked Asset Close To",
         "Unchecked Close Remainder To",
         "Unchecked Rekey to",
         "Unchecked Rekey operation",
         "Unchecked Transaction Fee",
         "Unchecked Asset Receiver",
         "Unchecked AssetReceiver",
         "Unchecked Payment Receiver",
         "Not Vulnerable",
         "No security risk",
         "No vulnerabilities"]


def parse_response(response: str) -> list[str]:
    pattern = re.compile(r"###\s*vulnerability\s*:?\s*\n?\s*(.+)", re.IGNORECASE)
    match = pattern.search(response.lower())

    vulnerabilities = []
    if match:
        vulns = match.group(1)
        for v in VULNS:
            if v.lower() in vulns:
                vulnerabilities.append(v)

    return vulnerabilities


def initialize(n: int) -> list:
    vet = [None] * n
    return vet


class VetOfMax:
    def __init__(self, n_of_max):
        self.n_of_max = n_of_max
        self.vet_of_max = initialize(n_of_max)

    def insert_sorted_desc(self, index_similarity: dict, verbose: bool = False):
        if verbose:
            print(f"insert_sorted_desc {index_similarity}")

        for i, m in enumerate(self.vet_of_max):
            num = index_similarity['num']
            if m is None:
                self.vet_of_max[i] = index_similarity
                break
            if m['num'] < num:
                self.vet_of_max[i] = index_similarity
                index_similarity = m

        if verbose:
            print(self.vet_of_max)