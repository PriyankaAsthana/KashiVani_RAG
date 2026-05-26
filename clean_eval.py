import json

with open('data/eval_questions.json', 'r') as f:
    questions = json.load(f)

bad_indices = [5, 11, 13, 16, 17, 23, 29, 31, 32, 36, 40, 45, 47, 49, 50]

clean_questions = [
    q for i, q in enumerate(questions) 
    if i not in bad_indices
]

with open('data/eval_questions_clean.json', 'w', encoding='utf-8') as f:
    json.dump(clean_questions, f, ensure_ascii=False, indent=2)

print(f"Original questions: {len(questions)}")
print(f"Removed: {len(bad_indices)}")
print(f"Clean questions saved: {len(clean_questions)}")
print("\nClean questions:")
for i, q in enumerate(clean_questions):
    print(f"[{i+1}] {q['question'][:60]}")