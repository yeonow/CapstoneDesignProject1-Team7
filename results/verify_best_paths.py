"""One-off final validation against all evaluation rows and seeded replays."""
import csv
import json
from pathlib import Path

from src.best_paths_experiment import paper_config, validate_best_path
from src.compare_algorithms import train_q_learning_agent, evaluate_q_learning, evaluate_random_agent

directory = Path('results/best_paths_8dir/random_target/grid_20x20')
with (directory / 'algorithm_comparison_episodes.csv').open(encoding='utf-8', newline='') as file:
    rows = list(csv.DictReader(file))
assert len(rows) == 6000
paired = {}
for row in rows:
    paired.setdefault((row['run'], row['episode']), []).append(row)
assert len(paired) == 3000
assert all(len(pair) == 2 and pair[0]['target_x'] == pair[1]['target_x']
           and pair[0]['target_y'] == pair[1]['target_y'] for pair in paired.values())
for algorithm, filename in [('q_learning', 'q_learning_best'), ('random', 'random_search_best')]:
    payload = json.loads((directory / f'{filename}.json').read_text(encoding='utf-8'))
    record = payload['trajectory']
    validate_best_path(record)
    candidates = [row for row in rows if row['algorithm'] == algorithm and row['success'] == 'True']
    best = min(candidates, key=lambda row: (int(row['steps']) / int(row['shortest_path_steps']),
                                          int(row['steps']), int(row['run']), int(row['episode'])))
    assert all(int(best[key]) == record[key] for key in ('run', 'episode', 'steps', 'train_seed', 'eval_seed'))
    with paper_config():
        if algorithm == 'q_learning':
            agent = train_q_learning_agent(1000, seed=record['train_seed'])
            replay = evaluate_q_learning(agent, record['episode'], seed=record['eval_seed'], record_paths=True)[-1]
        else:
            replay = evaluate_random_agent(record['episode'], seed=record['eval_seed'], record_paths=True)[-1]
    assert json.loads(json.dumps(replay['trajectory']))['path'] == record['path']
    assert replay['steps'] == record['steps'] and replay['success']
    print(f"VERIFIED {algorithm}: run={record['run']} episode={record['episode']} "
          f"target={record['target_position']} steps={record['steps']} "
          f"shortest={record['shortest_path_steps']} ratio={record['path_ratio']} "
          f"blocked={record['blocked_move_count']} successes={len(candidates)}/3000")
print('VERIFIED all 6000 evaluation rows, 3000 paired targets, global ranking and exact seeded coordinate replays')
