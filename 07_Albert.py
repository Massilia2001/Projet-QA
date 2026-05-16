import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import time

MODEL_NAME = "albert-base-v2" 
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 3e-5

print("=" * 60)
print(" FINE-TUNING ALBERT POUR QUESTION-ANSWERING (CPU OPTIMISÉ)")
print("=" * 60)

start_time = time.time()

# Charger le tokenizer et le modèle
print("\n Chargement du modèle et du tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

# Charger les données SQuAD
print(" Chargement du dataset SQuAD 2.0...")
dataset = load_dataset("squad_v2")

# Réduction du dataset
print(" Réduction du dataset pour CPU...")
train_dataset_reduced = dataset["train"].select(range(10000))
eval_dataset_reduced = dataset["validation"].select(range(1000))

# Fonction de prétraitement (IDENTIQUE)
def preprocess_squad_v2(examples):
    questions = [q.strip() for q in examples["question"]]
    
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )
    
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(offset_mapping):
        answer = examples["answers"][i]
        
        if not answer["answer_start"]:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i)
        
        context_start = next((j for j, sid in enumerate(sequence_ids) if sid == 1), 0)
        context_end = len(sequence_ids) - next((j for j, sid in enumerate(reversed(sequence_ids)) if sid == 1), 1)
        
        if offsets[context_start][0] > start_char or offsets[context_end - 1][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            start_pos = next((j for j in range(context_start, context_end) if offsets[j][0] <= start_char < offsets[j][1]), context_start)
            end_pos = next((j for j in range(context_end - 1, context_start - 1, -1) if offsets[j][0] < end_char <= offsets[j][1]), context_end - 1)
            
            start_positions.append(start_pos)
            end_positions.append(end_pos + 1)
    
    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print(" Prétraitement des données...")
train_dataset = train_dataset_reduced.map(
    preprocess_squad_v2, 
    batched=True, 
    remove_columns=train_dataset_reduced.column_names
)

eval_dataset = eval_dataset_reduced.map(
    preprocess_squad_v2, 
    batched=True, 
    remove_columns=eval_dataset_reduced.column_names
)

print(f" Données prétraitées !")
print(f"   - Training samples: {len(train_dataset)}")
print(f"   - Validation samples: {len(eval_dataset)}")

# Configuration de l'entraînement
print("\n  Configuration de l'entraînement...")
training_args = TrainingArguments(
    output_dir="./albert_qa_results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Créer le trainer
print(" Création du trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Entraîner le modèle
print("\n Début de l'entraînement...")
print("=" * 60)
print("  Temps estimé : 8-10 heures (CPU)")
print(" Modèle : albert-base-v2")
print(" Samples : 10000")
print("  Batch size : 4")
print(" Epochs : 1")
print("=" * 60)

trainer.train()

# Sauvegarder le modèle
print("\n Sauvegarde du modèle...")
model.save_pretrained("./albert_qa_model")
tokenizer.save_pretrained("./albert_qa_model")

end_time = time.time()
training_time = (end_time - start_time) / 3600

print("\n Entraînement terminé !")
print("=" * 60)
print(f" Modèle sauvegardé dans : ./albert_qa_model")
print(f"  Temps total : {training_time:.2f} heures")
print("=" * 60)
