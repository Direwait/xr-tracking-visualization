import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

def parse_tracking_file(path):
    frames = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            line = re.sub(
                r'("(?:pose|p)"\s*:\s*")([^"]+)(")',
                lambda m: m.group(1) + m.group(2).replace(',', ' ') + m.group(3),
                line
            )

            frames.append(json.loads(line))
    return frames


def parse_pose(pose_str):
    tokens = pose_str.split()
    if len(tokens) != 14:
        raise ValueError(f'Ожидалось 14 токенов, получено {len(tokens)}: {pose_str}')

    numbers = []
    for i in range(0, 14, 2):
        numbers.append(float(tokens[i] + '.' + tokens[i + 1]))

    return tuple(numbers[:3]), tuple(numbers[3:])


HAND_BONES = [
    (0, 1),
    (1, 2), (2, 3), (3, 4), (4, 5),
    (1, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (1, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (1, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]


if __name__ == '__main__':
    frames = parse_tracking_file('trackingData_20260505_165740.txt')
    print(f'Кадров: {len(frames)}')

    parsed = []
    skipped = 0
    for i, f in enumerate(frames):
        if 'Head' not in f or 'pose' not in f.get('Head', {}):
            skipped += 1
            continue

        head_pos, head_quat = parse_pose(f['Head']['pose'])

        left_joints = []
        left = f.get('Hand', {}).get('leftHand', {})
        if left.get('isActive') == 1:
            for j in left.get('HandJointLocations', []):
                if 'p' not in j:
                    continue
                pos, quat = parse_pose(j['p'])
                left_joints.append({'pos': pos, 'quat': quat,
                                    'valid': j.get('s', 0) > 0})

        right_joints = []
        right = f.get('Hand', {}).get('rightHand', {})
        if right.get('isActive') == 1:
            for j in right.get('HandJointLocations', []):
                if 'p' not in j:
                    continue
                pos, quat = parse_pose(j['p'])
                right_joints.append({'pos': pos, 'quat': quat,
                                     'valid': j.get('s', 0) > 0})

        body = []
        for j in f.get('Body', {}).get('joints', []):
            if 'p' not in j:
                continue
            pos, quat = parse_pose(j['p'])
            body.append({'pos': pos, 'quat': quat})

        parsed.append({
            'timestamp_ns': f.get('timeStampNs', 0),
            'head_pos': head_pos,
            'head_quat': head_quat,
            'left': left_joints,
            'right': right_joints,
            'body': body,
        })

    print(f'Распарсено: {len(parsed)}, пропущено: {skipped}')
    if parsed:
        print(f'Пример: голова {parsed[0]["head_pos"]}, '
              f'левая рука {len(parsed[0]["left"])} суставов, '
              f'тело {len(parsed[0]["body"])} суставов')
        # ============================================================
    # ВИЗУАЛИЗАЦИЯ
    # ============================================================

    # --- Время в секундах от начала ---
    ts = np.array([p['timestamp_ns'] for p in parsed], dtype=np.float64)
    t = (ts - ts[0]) / 1e9

    # --- График 1: позиция головы ---
    head_xyz = np.array([p['head_pos'] for p in parsed])

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, head_xyz[:, i])
        ax.set_ylabel(['X, м', 'Y, м', 'Z, м'][i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Время, с')
    axes[0].set_title('Позиция головы')
    fig.tight_layout()
    fig.savefig('head_position.png', dpi=120)
    print('Сохранено: head_position.png')
    plt.close(fig)

    # --- График 2: позиция ладоней ---
    def palm_pos(key):
        out = []
        for p in parsed:
            if p[key]:
                out.append(p[key][0]['pos'])
            else:
                out.append((np.nan, np.nan, np.nan))
        return np.array(out, dtype=np.float64)

    left_palm = palm_pos('left')
    right_palm = palm_pos('right')

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, left_palm[:, i], label='Левая')
        ax.plot(t, right_palm[:, i], label='Правая', alpha=0.6)
        ax.set_ylabel(['X, м', 'Y, м', 'Z, м'][i])
        ax.grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_title('Позиция ладоней')
    axes[-1].set_xlabel('Время, с')
    fig.tight_layout()
    fig.savefig('hands_position.png', dpi=120)
    print('Сохранено: hands_position.png')
    plt.close(fig)

    # === АНИМАЦИЯ: ОБЕ РУКИ НА ОДНОЙ 3D-СЦЕНЕ ===
    from matplotlib.animation import FuncAnimation

    # Собираем кадры, где есть хотя бы одна валидная рука
    valid_frames_both = []
    for idx, p in enumerate(parsed):
        has_left = p['left'] and len(p['left']) == 26
        has_right = p['right'] and len(p['right']) == 26
        if has_left or has_right:
            valid_frames_both.append(idx)

    if len(valid_frames_both) < 2:
        print('Мало валидных кадров для анимации')
    else:
        step = max(1, len(valid_frames_both) // 200)
        anim_frames_both = valid_frames_both[::step]
        print(f'Кадров в объединённой анимации: {len(anim_frames_both)}')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Собираем границы по всем кадрам и обеим рукам
        all_pts_both = []
        for idx in anim_frames_both:
            if parsed[idx]['left']:
                all_pts_both.extend([j['pos'] for j in parsed[idx]['left']])
            if parsed[idx]['right']:
                all_pts_both.extend([j['pos'] for j in parsed[idx]['right']])
        all_pts_both = np.array(all_pts_both)
        margin = 0.05

        xlim = (all_pts_both[:, 0].min() - margin, all_pts_both[:, 0].max() + margin)
        ylim = (all_pts_both[:, 1].min() - margin, all_pts_both[:, 1].max() + margin)
        zlim = (all_pts_both[:, 2].min() - margin, all_pts_both[:, 2].max() + margin)

        def draw_hand(hand_joints, color):
            """Рисует скелет одной руки заданным цветом."""
            pts = np.array([j['pos'] for j in hand_joints])
            for a, b in HAND_BONES:
                ax.plot([pts[a, 0], pts[b, 0]],
                        [pts[a, 1], pts[b, 1]],
                        [pts[a, 2], pts[b, 2]],
                        color=color, linewidth=2)
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='red', s=15)

        def update_both(frame_idx):
            ax.clear()
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_xlabel('X, м')
            ax.set_ylabel('Y, м')
            ax.set_zlabel('Z, м')

            t_sec = (parsed[frame_idx]['timestamp_ns'] - parsed[0]['timestamp_ns']) / 1e9
            ax.set_title(f'Обе руки — кадр {frame_idx}, t = {t_sec:.2f} с')

            # Левая — синяя
            if parsed[frame_idx]['left'] and len(parsed[frame_idx]['left']) == 26:
                draw_hand(parsed[frame_idx]['left'], 'steelblue')
            # Правая — оранжевая
            if parsed[frame_idx]['right'] and len(parsed[frame_idx]['right']) == 26:
                draw_hand(parsed[frame_idx]['right'], 'darkorange')

        anim_both = FuncAnimation(fig, update_both, frames=anim_frames_both,
                                  interval=50, repeat=True)
        anim_both.save('hands_animation_both.gif', writer='pillow', fps=20)
        print('Сохранено: hands_animation_both.gif')
        plt.close(fig)