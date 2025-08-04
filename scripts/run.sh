llm_provider="openai"
llm_model="gpt-4o-mini"
outer=(1)
datasets=(2wikimultihopqa hotpotqa musique trivia)

for i in ${outer[@]}
do
    round_name="round-"$i"_gpt-4o-mini"
    for data in ${datasets[@]}
    do
        python main.py run --dataset $data --round_name $round_name --example_mode similar --llm_provider=$llm_provider --llm_model=$llm_model
    done
done