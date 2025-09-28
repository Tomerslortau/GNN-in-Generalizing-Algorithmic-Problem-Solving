To precompute eval and training states:

python data/precompute_dfs.py \                                                         
  --N_min 1 --N_max 20 \
  --per_N 2000 \
  --eval_starts_per_N 2000 \
  --cache_dir data/cache \
  --seed 0

where chache_dir is the directory to store the precomputed states.
Note that running by default looks in data/cache unless otherwise provided in the args.
Precomputing states could take 10-20 minutes depending on the hardware.

To run the experiment cd into the the checkers jumping directory
run:

python experiments/statewise_success_eval_gs.py \                                            
  --N_train_low 3 --N_train_high 8 \
  --N_eval_low 9 --N_eval_high 18 \
  --per_N 1500 --epochs 40 --lr 2e-3 \
  --seed 0 --device cpu \
  --out_dir final_results/statewise_delta \
  --starts_per_N 800 \
  --rollout_slack 1.0 \
  --dagger_rollouts 0 \
  --depths 2 3 \
  --hiddens 16 32 64

To generate the plots from the experiment results run:

python generate_plots.py --csv <csv from experiment> --out_dir final_final_results/

where --csv is the generated csv from the experiment