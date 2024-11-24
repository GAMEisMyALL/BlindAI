import argparse
import os
import csv

def calculate(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
    rounds = []
    p1_hp = []
    p2_hp = []
    times = []

    for file in files:
        with open(file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rounds.append(int(row[0]))
                p1_hp.append(int(row[1]))
                p2_hp.append(int(row[2]))
                times.append(float(row[3]))

    total_games = len(rounds)
    win_ratio = sum(p1 > p2 for p1, p2 in zip(p1_hp, p2_hp)) / total_games
    hp_diff = sum(p1 - p2 for p1, p2 in zip(p1_hp, p2_hp)) / total_games

    return win_ratio, hp_diff

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    path1 = os.path.join('KirariAI', 'KiraAI', 'with_testAI')
    path2 = os.path.join('KirariAI', 'KiraAI', 'with_randomAI')
    #parser.add_argument('--path', type=str, required=True, help='The directory containing result log')
    #args = parser.parse_args()
    win_ratio, hp_diff = calculate(path1)
    print('The winning ratio is:', win_ratio)
    print('The average HP difference is:', hp_diff)
    print('randomAI')
    win_ratio, hp_diff = calculate(path2)
    print('The winning ratio is:', win_ratio)
    print('The average HP difference is:', hp_diff)
