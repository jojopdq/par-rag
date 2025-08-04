round_name="round-1_ablation_no-review-mode"
ablation_mode="no_review_mode"
#ablation_mode="no_plan_mode"
for data in 2wikimultihopqa hotpotqa musique trivia;
do
  python main.py run --dataset $data --ablation_mode $ablation_mode --example_mode similar --round_name $round_name
done
