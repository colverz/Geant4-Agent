# Multi-task (structure + token) training
python nlu/bert_lab/data_multitask.py --out nlu/bert_lab/data/bert_lab_multitask_samples.jsonl --n 800 --seed 7
python nlu/bert_lab/train_multitask.py --data nlu/bert_lab/data/bert_lab_multitask_samples.jsonl --outdir nlu/bert_lab/models/multitask --model distilbert-base-uncased --epochs 1 --batch_size 4 --max_length 192 --device cuda --llm_aug_n 50
