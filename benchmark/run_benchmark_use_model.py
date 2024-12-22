from pathlib import Path

import pandas as pd
import numpy as np
import tqdm
import time
import os

import bird_view.utils.bz_utils as bzu
import bird_view.utils.carla_utils as cu

from bird_view.models.common import crop_birdview

from raindrop.dropgenerator import generateDrops
from raindrop.config import cfg
from PIL import Image

from tensorflow.keras.models import load_model

        
model_path = 'self_driving_input_assessment_model.h5'  # モデルのパスを指定してください
model = load_model(model_path)

# global image_counter
image_counter = 0

def add_raindrops_to_frame(frame):

    pil_image = Image.fromarray(frame)
    output_image, _ = generateDrops(pil_image, cfg)
    # 黒いフレームを作成（すべてのピクセル値を0に設定
    # height, width, channels = frame.shape
    # black_frame = np.zeros((height, width, channels), dtype=np.uint8)
    # resize_output_image = Image.fromarray(black_frame).resize((384, 160))
    # output_array = np.array(resize_output_image)

    resize_output_image = output_image.resize((384, 160))
    output_array = np.array(resize_output_image)

    return output_array

def preprocess_image(image):
    import cv2

    # 画像をリサイズ（学習時と同じサイズに合わせる）
    img = cv2.resize(image, (224, 224))  # 幅384、高さ160
    # ピクセル値を0〜1に正規化
    img = img / 255.0
    # 次元を追加（バッチサイズの次元）
    img = np.expand_dims(img, axis=0)
    return img

def _paint(observations, control, diagnostic, debug, env, show=False):
    import cv2
    import numpy as np

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    birdview = cu.visualize_birdview(observations['birdview'])
    birdview = crop_birdview(birdview)

    if 'big_cam' in observations:
        canvas = np.uint8(observations['big_cam']).copy()
        rgb = np.uint8(observations['rgb']).copy()
    else:
        canvas = np.uint8(observations['rgb']).copy()

    lane_invasion_count = env.lane_invasion_count
    collision_count = env.collision_count

    def _stick_together(a, b, axis=1):

        if axis == 1:
            h = min(a.shape[0], b.shape[0])

            r1 = h / a.shape[0]
            r2 = h / b.shape[0]
        
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 1)
            
        else:
            h = min(a.shape[1], b.shape[1])
            
            r1 = h / a.shape[1]
            r2 = h / b.shape[1]
        
            a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
            b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))
    
            return np.concatenate([a, b], 0)

    def _write(text, i, j, canvas=canvas, fontsize=0.4):
        rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
        cols = [x * (canvas.shape[1] // 9) for x in range(9+1)]
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, fontsize, WHITE, 1)
                
    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(observations['command'], '???')
            
    if 'big_cam' in observations:
        fontsize = 0.8
    else:
        fontsize = 0.4

    # 信頼度の予測
    preprocessed_img = preprocess_image(observations['rgb'])
    prediction = model.predict(preprocessed_img)
    confidence_score = prediction[0][0]

    _write('Confidence Score: %.4f' % confidence_score, 7, 0, fontsize=fontsize)


    _write('Command: ' + _command, 1, 0, fontsize=fontsize)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0, fontsize=fontsize)

    _write('Steer: %.2f' % control.steer, 4, 0, fontsize=fontsize)
    _write('Throttle: %.2f' % control.throttle, 5, 0, fontsize=fontsize)
    _write('Brake: %.1f' % control.brake, 6, 0, fontsize=fontsize)

    # _write('Collided: %s' % diagnostic['collided'], 1, 6, fontsize=fontsize)
    _write('Collided: %s' % collision_count, 1, 6, fontsize=fontsize)
    # _write('Invaded: %s' % diagnostic['invaded'], 2, 6, fontsize=fontsize)
    _write('Lane Invaded: %s' % lane_invasion_count, 2, 6, fontsize=fontsize)
    _write('Lights Ran: %d/%d' % (env.traffic_tracker.total_lights_ran, env.traffic_tracker.total_lights), 3, 6, fontsize=fontsize)
    _write('Goal: %.1f' % diagnostic['distance_to_goal'], 4, 6, fontsize=fontsize)

    _write('Time: %d' % env._tick, 5, 6, fontsize=fontsize)
    _write('FPS: %.2f' % (env._tick / (diagnostic['wall'])), 6, 6, fontsize=fontsize)

    for x, y in debug.get('locations', []):
        x = int(X - x / 2.0 * CROP_SIZE)
        y = int(Y + y / 2.0 * CROP_SIZE)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED

    for x, y in debug.get('locations_world', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED
    
    for x, y in debug.get('locations_birdview', []):
        S = R // 2
        birdview[x-S:x+S+1,y-S:y+S+1] = RED       
 
    for x, y in debug.get('locations_pixel', []):
        S = R // 2
        if 'big_cam' in observations:
            rgb[y-S:y+S+1,x-S:x+S+1] = RED
        else:
            canvas[y-S:y+S+1,x-S:x+S+1] = RED
        
    for x, y in debug.get('curve', []):
        x = int(X - x * 4)
        y = int(Y + y * 4)

        try:
            birdview[x,y] = [155, 0, 155]
        except:
            pass

    if 'target' in debug:
        x, y = debug['target'][:2]
        x = int(X - x * 4)
        y = int(Y + y * 4)
        birdview[x-R:x+R+1,y-R:y+R+1] = [0, 155, 155]

    ox, oy = observations['orientation']
    rot = np.array([
        [ox, oy],
        [-oy, ox]])
    u = observations['node'] - observations['position'][:2]
    v = observations['next'] - observations['position'][:2]
    u = rot.dot(u)
    x, y = u
    x = int(X - x * 4)
    y = int(Y + y * 4)
    v = rot.dot(v)
    x, y = v
    x = int(X - x * 4)
    y = int(Y + y * 4)

    if 'big_cam' in observations:
        _write('Network input/output', 1, 0, canvas=rgb)
        _write('Projected output', 1, 0, canvas=birdview)
        full = _stick_together(rgb, birdview)
    else:
        full = _stick_together(canvas, birdview)

    if 'image' in debug:
        full = _stick_together(full, cu.visualize_predicted_birdview(debug['image'], 0.01))
        
    if 'big_cam' in observations:
        full = _stick_together(canvas, full, axis=0)
    
    if show:
        bzu.show_image('canvas', full)
    bzu.add_to_video(full)


def run_single(env, weather, start, target, agent_maker, seed, autopilot, show=False):
    # HACK: deterministic vehicle spawns.
    global image_counter
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])

    if not autopilot:
        print('-------autopilot1--------')
        agent = agent_maker()
    else:
        agent = agent_maker(env._player, resolution=1, threshold_before=7.5, threshold_after=7.5)
        agent.set_route(env._start_pose.location, env._target_pose.location)

    diagnostics = list()
    result = {
            'weather': weather,
            'start': start, 'target': target,
            'success': None, 't': None,
            'total_lights_ran': None,
            'total_lights': None,
            'collided': None,
            }
    
    data_list = []

    image_save_dir = '/home/yoshi-22/LearningByCheating/datasets/images2'
    os.makedirs(image_save_dir, exist_ok=True)

    while env.tick():
        observations = env.get_observations()
        observations['rgb'] = add_raindrops_to_frame(observations['rgb'])
        control = agent.run_step(observations)
        diagnostic = env.apply_control(control)

        _paint(observations, control, diagnostic, agent.debug, env, show=show)

        diagnostic.pop('viz_img')
        diagnostics.append(diagnostic)

        # image_filename = f'image_{env._tick:05d}.png'
        image_filename = f'image_{image_counter:08d}.png'
        collision_count = env.collision_count
        lane_invasion_count = env.lane_invasion_count
        current_speed = np.linalg.norm(observations['velocity'])
        vehicle_command = observations['command']

        image_save_path = os.path.join(image_save_dir, image_filename)
        Image.fromarray(observations['rgb']).save(image_save_path)

        image_counter += 1

        data_list.append({
            'image_filename': image_filename,
            'collision_count': collision_count,
            'lane_invasion_count': lane_invasion_count,
            'current_speed': current_speed,
            'vehicle_command': vehicle_command
        })

        if env.is_failure() or env.is_success():
            result['success'] = env.is_success()
            result['total_lights_ran'] = env.traffic_tracker.total_lights_ran
            result['total_lights'] = env.traffic_tracker.total_lights
            result['collided'] = env.collided
            result['t'] = env._tick
            break
    
    df = pd.DataFrame(data_list)
    df.to_csv('/home/yoshi-22/LearningByCheating/datasets/data2.csv', index=False)

    return result, diagnostics


def run_benchmark(agent_maker, env, benchmark_dir, seed, autopilot, resume, max_run=5, show=False):
    """
    benchmark_dir must be an instance of pathlib.Path
    """

    summary_csv = benchmark_dir / 'summary.csv'
    diagnostics_dir = benchmark_dir / 'diagnostics'
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # summaryにベンチマークの結果を追加
    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=total):
        if resume and len(summary) > 0 and ((summary['start'] == start) \
                       & (summary['target'] == target) \
                       & (summary['weather'] == weather)).any():
            print (weather, start, target)
            continue


        diagnostics_csv = str(diagnostics_dir / ('%s.csv' % run_name))

        bzu.init_video(save_dir=str(benchmark_dir / 'videos'), save_path=run_name)

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed, autopilot, show=show)

        summary = pd.concat([summary, pd.DataFrame([result])], ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        num_run += 1

        if num_run >= max_run:
            break
