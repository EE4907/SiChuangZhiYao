echo "SMILES,Label
CC1CCC(CC1)C(=O)NC2=NC(=C(S2)CC(=O)N3CCN(CC3)C)C4=CC=CS4,1.0
C1=C(C=C(C=C1O)Cl)C(=O)O,1.0
CCOC1=C(C=C(C(=C1)Cl)C(=O)NC(=O)NC2=NC3=C(S2)C=C(C=C3)S(=O)(=O)C4CCN(CC4)C)N5C=CC=N5,1.0
C1=CC(=CC(=C1)O)C(=O)O,1.0
C1=C(C=C(C=C1O)O)C(=O)O,1.0
CCCCC/C=C\C/C=C\C/C=C\C=C\C(=O)CCCC(=O)OC,1.0" > "profile.csv"

export gpu_id=0
export max_gpu_id=3
export label_col="Label"
export smiles_column="SMILES"
export keep_top=3000
export probe_cluster="Yes" # Yes or No
for library in "merged";do
for decoy_set in "profile.csv" ;do # 
export basename=${decoy_set%%.csv}
cat<<EOF > ${basename}_${library}_PVS.pbs
#PBS -N ${basename}_${library}
#PBS -l nodes=1:ppn=4:gpus=1:p40
#PBS -S /bin/bash
#PBS -j oe
#PBS -l walltime=720:00:00
#PBS -q pub_gpu

cd \$PBS_O_WORKDIR
module remove cuda/7/11.0
source activate GeminiMol
export PATH=/public/home/wanglin3/software/miniconda3/envs/GeminiMol/bin:\${PATH}
module load 7/compiler/gnu_8.3.0 cuda/7/11.6
hostname
nvidia-smi

CUDA_VISIBLE_DEVICES=${gpu_id} python -u ${geminimol_app}/PharmProfiler.py "${geminimol_lib}/GeminiMol" "${basename}_${library}" "${smiles_column}" "${geminimol_data}/compound_library/${library}.csv" "${decoy_set}:${label_col}" ${keep_top} "${probe_cluster}"
EOF
qsub ${basename}_${library}_PVS.pbs
if [ ${gpu_id} == ${max_gpu_id} ];then
    export gpu_id=0
else
    ((gpu_id+=1))
fi
done
done
