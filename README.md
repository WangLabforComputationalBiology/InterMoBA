InterMoBA
--------------------
InterMoBA is a deep learning framework speciffcally designed for protein-ligand docking and binding afffnity prediction, with a particular emphasis on energy-based evaluation. The architecture draws inspiration from Graph-Transformer models and incorporates the mixture of block attention (MoBA) module, which directly predicts binding energies and utilizes them as the final afffnity score. The lower the quality of the docking pose, the more probably predicting the lower affinity value.


## Contents

- [Installation](#set-up)
- [Example](#example)
- [FAQ](#FAQ)

<a id="set-up"></a>

### Installation

Create InterMoBA environment using conda

```
cd InterMoBA
conda env create --file environment.yml
conda activate interMoBA

# Install PLIP without wrong openbabel dependency
pip install --no-deps plip

# Compile Docking sampling program from source code
cd docking && python setup.py install && cd ../
```

<a id="example"></a>

### Example

Data Preparation

```
# Ligand
# Add hydrogens/protonation states
obabel examples/raw/3g2n_ligand.sdf -p 7.4 -O examples/ligand/3g2n_docked.sdf

# Generate initial conformation
python tools/rdkit_ETKDG_3d_gen.py examples/ligand/ examples/uff  
####
# Protein
# Preprocess protein
mkdir -p examples/raw/pocket && reduce examples/raw/3g2n.pdb > examples/raw/pocket/3g2n_reduce.pdb

# Extract binding pocket (10Å radius)
python tools/extract_pocket_by_ligand.py examples/raw/pocket/ examples/ligand/ 1 && mv examples/raw/pocket/output/3g2n_pocket.pdb examples/pocket
```

Input data strcutures

```
examples/
├── demo_dock.csv  # the query csv for interformer prediction [Target=PDB, Molecule ID=name in SDF file, pose_rank=the nth molecule in sdf file]
├── ligand/$PDB_docked.sdf  # [reference ligand] foler contains the reference ligand conformation from PDB
├── pocket/$PDB_pocket.pdb # [binding pocket site] foler contains the target protein PDB structure 
├── uff/$PDB_uff.sdf # [initial ligand conformation] foler contains a single ligand conformation minimized by field foce
```

Predicting energy functions file. Download checkpoints from [zenodo](https://doi.org/10.5281/zenodo.16147492).

```
DOCK_FOLDER=energy_output

PYTHONPATH=interformer/ python inference.py -test_csv examples/demo_dock.csv \
-work_path examples/ \
-ensemble checkpoints/energy_model \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_FOLDER \
-reload \
-debug
```

Output data strcutures

```
energy_output/
├── complex  # pocket intercepted through reference ligand
├── gaussian_predict  # predicted energy functions file
├── ligand  # copy from $work_path ligand folder [used for locate an autobox (20Ax20Ax20A sampling space) from reference ligand]
└── uff  # copy from $work_path uff folder
```

Giving the energy files produce docking poses via MonteCarlo sampling, uses pose_rank=0 prediction as final affinity result
```
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all find

# Make a docking summary csv 
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all stat

# Merging original csv with the docking summary, gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
python docking/merge_summary_input.py $DOCK_FOLDER/ligand_reconstructing/stat_concated.csv examples/demo_dock.csv
```

<a id="FAQ"></a>
### FAQ

- **Benchmark data and paper results**  
  Download the required benchmark files from the [InterMoBA Zenodo repository](https://doi.org/10.5281/zenodo.16147492).  
  Extract `benchmark.zip` and `dock_results.zip`; place their contents as instructed in the project structure.

- **Training data**  
  Obtain the processed training set(PDBbind v2020) from the [PDBBind+ website](https://www.pdbbind-plus.org.cn/).  
  After downloading, place the processed files under `interformer/poses/`.

