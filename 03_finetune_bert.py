import torch
import time
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer
from datasets import load_dataset
import warnings
warnings.filterwarnings('ignore')

#  CONFIGURATION OPTIMISÉE POUR CPU
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 3e-5
MAX_SAMPLES = 10000

print("=" * 70)
print(" FINE-TUNING DISTILBERT POUR QUESTION-ANSWERING (CPU OPTIMISÉ)")
print("=" * 70)

# 1. Charger le tokenizer et le modèle
print("\n Chargement du modèle et du tokenizer...")
start_time = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)
load_time = time.time() - start_time
print(f"    Modèle chargé en {load_time:.2f}s")

device = torch.device("cpu")
print(f"    Device utilisé : {device}")
model.to(device)

# 2. Charger les données SQuAD
print("\n Chargement du dataset SQuAD 2.0...")
start_time = time.time()
dataset = load_dataset("squad_v2")
load_data_time = time.time() - start_time
print(f"    Dataset chargé en {load_data_time:.2f}s")

# 3. Prétraiter les données
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

print("🔄 Prétraitement des données...")
start_time = time.time()
train_dataset = dataset["train"].map(
    preprocess_squad_v2, 
    batched=True, 
    remove_columns=dataset["train"].column_names,
    batch_size=1000
)

eval_dataset = dataset["validation"].map(
    preprocess_squad_v2, 
    batched=True, 
    remove_columns=dataset["validation"].column_names,
    batch_size=1000
)

print(f"\n Réduction du dataset pour CPU...")
train_dataset = train_dataset.select(range(min(MAX_SAMPLES, len(train_dataset))))
eval_dataset = eval_dataset.select(range(min(MAX_SAMPLES // 10, len(eval_dataset))))

preprocess_time = time.time() - start_time
print(f"    Prétraitement terminé en {preprocess_time:.2f}s")
print(f"   - Training samples: {len(train_dataset)}")
print(f"   - Validation samples: {len(eval_dataset)}")

# 4. Configuration de l'entraînement (CORRIGÉE)
print("\n  Configuration de l'entraînement...")
training_args = TrainingArguments(
    output_dir="./distilbert_qa_results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    logging_steps=100,
    save_total_limit=1,
    load_best_model_at_end=True,
    use_cpu=True, 
    dataloader_num_workers=0,
    report_to="none",
)

# 5. Créer le trainer
print(" Création du trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 6. Entraîner le modèle
print("\n Début de l'entraînement...")
print("=" * 70)
print(f"  Temps estimé : 3-4 heures (CPU)")
print(f" Modèle : {MODEL_NAME}")
print(f" Samples : {len(train_dataset)}")
print(f"  Batch size : {BATCH_SIZE}")
print(f" Epochs : {EPOCHS}")
print("=" * 70)

start_training = time.time()
try:
    trainer.train()
    training_time = time.time() - start_training
    print(f"\n Entraînement terminé en {training_time/3600:.2f} heures !")
except RuntimeError as e:
    if "out of memory" in str(e):
        print("\n ERREUR : Pas assez de RAM !")
        print("   Solutions :")
        print("   1. Réduis BATCH_SIZE à 2")
        print("   2. Réduis MAX_SAMPLES à 5000")
    else:
        raise

# 7. Sauvegarder le modèle
print("\n Sauvegarde du modèle...")
model.save_pretrained("./distilbert_qa_model")
tokenizer.save_pretrained("./distilbert_qa_model")

print("\n" + "=" * 70)
print(" ENTRAÎNEMENT TERMINÉ !")
print("=" * 70)
print(" Modèle sauvegardé dans : ./distilbert_qa_model")
