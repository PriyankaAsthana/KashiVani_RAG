import json

with open('data/eval_questions.json', 'r') as f:
    questions = json.load(f)

bad = []
for i, q in enumerate(questions):
    ref = q.get('reference', '')
    flags = [
        'no information',
        'not mentioned',
        'not explicitly',
        'not specified',
        'not provided',
        'does not mention',
        'does not provide',
        'context does not'
    ]
    if any(flag in ref.lower() for flag in flags):
        bad.append(i)
        print(f"[{i}] {q['question'][:60]}")

print(f"\nTotal bad references: {len(bad)}")
print(f"Indices to remove: {bad}")