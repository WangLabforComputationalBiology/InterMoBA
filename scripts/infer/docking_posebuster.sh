DOCK_FOLDER=dock_results/energy_posebuster
WORK_PATH=benchmark/docking/posebuster
mkdir -p DOCK_FOLDER
######
PYTHONPATH=interformer/ python inference.py -test_csv $WORK_PATH/posebuster_infer.csv  \
-work_path $WORK_PATH \
-ensemble checkpoints/energy_model \
-gpus 1 \
-batch_size 1 \
-posfix *val_loss* \
-energy_output_folder $DOCK_FOLDER \
-reload

# Using refer conformation
OMP_NUM_THREADS="64,64" python docking/reconstruct_ligands.py -y --cwd $DOCK_FOLDER -y --find_all --output_folder ref find 
python docking/reconstruct_ligands.py --cwd $DOCK_FOLDER --find_all --output_folder ref stat

python docking/merge_summary_input.py dock_results/energy_posebuster/ref/stat_concated.csv $WORK_PATH/posebuster_infer.csv # gather information of rmsd, enery, num_torsions and poserank(cid, id of the conformation in a sdf)
mv dock_results/energy_posebuster/ref $WORK_PATH/infer
