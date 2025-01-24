def can_smiles(smi):
    """
    将输入的 SMILES 字符串转换为标准化的 SMILES 字符串。
    参数:
    smi (str): 输入的 SMILES 字符串。
    返回:
    str: 标准化的 SMILES 字符串。如果转换失败，则返回原始的 SMILES 字符串。
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        standardizer = MolStandardize.normalize

        # standardize the molecule
        standardized_mol = standardizer.Normalizer().normalize(mol)
        # get the standardized SMILES string
        standardized_smiles = Chem.MolToSmiles(standardized_mol, isomericSmiles=True)
    except:
        standardized_smiles = smi

    return standardized_smiles

def ToDevice(obj, device):
    if isinstance(obj, dict):
        for k in obj:
            obj[k] = ToDevice(obj[k], device)
        return obj
    elif isinstance(obj, tuple) or isinstance(obj, list):
        for i in range(len(obj)):
            obj[i] = ToDevice(obj[i], device)
        return obj
    else:
        return obj.to(device)
    
def load_hugo2ncbi():
    ncbi2hugo = {}
    hugo2ncbi = {}
    try:
        with open("../assets/drp/enterez_NCBI_to_hugo_gene_symbol_march_2019.txt", "r") as f:
            for line in f.readlines():
                line = line.strip("\n").split("\t")
                if len(line) > 1:
                    ncbi2hugo[line[0]] = line[1]
                    hugo2ncbi[line[1]] = line[0]
    except:
        logger.warn("NCBI2hugo gene not found")
    return ncbi2hugo, hugo2ncbi



def to_clu_sparse(data):
    s = "%d %d %d" % (data.shape[0], data.shape[1], np.sum(data))
    s_row = [""] * data.shape[0]
    non_zero_row, non_zero_col = np.where(data > 0)
    for i in tqdm(range(len(non_zero_row))):
        s_row[non_zero_row[i]] += " %d %f" % (non_zero_col[i] + 1, data[non_zero_row[i], non_zero_col[i]])
    return "\n".join([s] + s_row)