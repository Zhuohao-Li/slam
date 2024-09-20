# based on quest longbench.sh
cd evaluation/LongBench

model="longchat-v1.5-7b-32k"

for task in "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "triviaqa"
do   
    python -u pred.py \
        --model $model --task $task

    CUDA_VISIBLE_DEVICES=0 python -u pred.py \
        --model $model --task $task
done

python -u eval.py --model $model
