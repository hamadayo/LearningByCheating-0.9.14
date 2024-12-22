"""
Stores tuples of (birdview, measurements, rgb).

Run from top level directory.
Sample usage -

python3 bird_view/data_collector.py \
        --dataset_path $PWD/data \
        --frame_skip 10 \
        --frames_per_episode 1000 \
        --n_episodes 100 \
        --port 3000 \
        --n_vehicles 0 \
        --n_pedestrians 0
"""
import argparse

from pathlib import Path

import numpy as np
import tqdm
import lmdb

import carla

from benchmark import make_suite
from bird_view.utils import carla_utils as cu
from bird_view.utils import bz_utils as bu

from bird_view.models.common import crop_birdview
from bird_view.models.controller import PIDController
from bird_view.models.roaming import RoamingAgentMine


# デバッグ用
def _debug(observations, agent_debug):
    import cv2

    processed = cu.process(observations)

    control = observations['control']
    control = [control.steer, control.throttle, control.brake]
    # 小数点以下2桁まで表示
    control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in control)
    real_control = observations['real_control']
    real_control = [real_control.steer, real_control.throttle, real_control.brake]
    real_control = ' '.join(str('%.2f' % x).rjust(5, ' ') for x in real_control)

    canvas = np.uint8(observations['rgb']).copy()
    # canvasはRGBが画像。縦横それぞれ１０で割った位置を取得
    rows = [x * (canvas.shape[0] // 10) for x in range(10+1)]
    cols = [x * (canvas.shape[1] // 10) for x in range(10+1)]

    WHITE = (255, 255, 255)
    CROP_SIZE = 192
    X = 176
    Y = 192 // 2
    R = 2

    def _write(text, i, j):
        # canvasが画像、textが表示する文字、iが縦位置、jが横位置, cv2.FONT_HERSHEY_SIMPLEXはフォント、0.4はフォントサイズ、WHITEは色、1は太さ
        cv2.putText(
                canvas, text, (cols[j], rows[i]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)

    # 1~4に対応するコマンドを取得。それ以外は'???'を取得
    _command = {
            1: 'LEFT',
            2: 'RIGHT',
            3: 'STRAIGHT',
            4: 'FOLLOW',
            }.get(int(observations['command']), '???')

    _write('Command: ' + _command, 1, 0)
    _write('Velocity: %.1f' % np.linalg.norm(observations['velocity']), 2, 0)
    # -5だと右から5番目、00だと左から0番目
    _write('Real: %s' % control, -5, 0)
    _write('Control: %s' % control, -4, 0)

    r = 2
    # 鳥瞰図を取得し描画用に加工
    birdview = cu.visualize_birdview(crop_birdview(processed['birdview']))

    # 鳥瞰図にマーカーを描画
    def _dot(x, y, color):
        x = int(x)
        y = int(y)
        # x,yの範囲を指定96-2+0 : 96+2+1+0 = 94 : 99
        birdview[176-r-x:176+r+1-x,96-r+y:96+r+1+y] = color

    _dot(0, 0, [255, 255, 255])
            # デバッグ用出力

    # 回転行列　もし[1,0]（cos, sin）なら右向き、[0.866, 0.5],  # 車両が30度右回転している方向
    ox, oy = observations['orientation']
    R = np.array([
        [ox,  oy],
        [-oy, ox]])

    # agent_debug['vehicle'] = [10, 10, 0], agent_debug['waypoint'] = [15, 12, 0]（両方ともワールド座標）のとき、u = [15 - 10, 12 - 10] → [5, 2]
    u = np.array(agent_debug['waypoint']) - np.array(agent_debug['vehicle'])
    # u = R.dot([5, 2])  # Rは単位行列なので u = [5, 2]
    #最初の２要素のみを取得x,yのみでいいのでzはなし
    u = R.dot(u[:2])
    # u = [5 * 4, 2 * 4] → [20, 8] スケーリング。鳥瞰図画像の単位に
    u = u * 4

    _dot(u[0], u[1], [255, 255, 255])

    # 黒い鳥瞰図画像 (192x192ピクセル),黒いRGB画像 (300x400ピクセル)
    # def _stick_together(a, b):
    #     h = min(a.shape[0], b.shape[0])
    #     # canvas.shape[0] = 300, birdview.shape[0] = 192
    #     # h = 192  # 鳥瞰図の高さに合わせる

    #     r1 = h / a.shape[0]
    #     # 192 / 300 = 0.64
    #     r2 = h / b.shape[0]
    #     # 192 / 192 = 1.0

    #     a = cv2.resize(a, (int(r1 * a.shape[1]), int(r1 * a.shape[0])))
    #     # リサイズ後: (400 * 0.64, 300 * 0.64) = (256, 192)
    #     b = cv2.resize(b, (int(r2 * b.shape[1]), int(r2 * b.shape[0])))

    #     return np.concatenate([a, b], 1)

    def _stick_together(a, b):
        # 画像の高さを取得
        h_a, w_a = a.shape[:2]
        h_b, w_b = b.shape[:2]

        # 最大の高さを基準にする
        max_h = max(h_a, h_b)

        # 画像 a にパディングを追加
        if h_a < max_h:
            pad_top = (max_h - h_a) // 2
            pad_bottom = max_h - h_a - pad_top
            a = cv2.copyMakeBorder(a, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 画像 b にパディングを追加
        if h_b < max_h:
            pad_top = (max_h - h_b) // 2
            pad_bottom = max_h - h_b - pad_top
            b = cv2.copyMakeBorder(b, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        # 横方向に結合
        return np.concatenate([a, b], axis=1)

    full = _stick_together(canvas, birdview)

    scale_factor = 2
    full = cv2.resize(
        full,
        (full.shape[1] * scale_factor, full.shape[0] * scale_factor),
        interpolation=cv2.INTER_LINEAR
    )

    bu.show_image('full', full)

class NoisyAgent(RoamingAgentMine):
    """
    Each parameter is in units of frames.
    State can be "drive" or "noise".
    """
    def __init__(self, env, noise=None):
        super().__init__(env._player, resolution=1, threshold_before=7.5, threshold_after=5.)

        # ’drive’が通常の制御動作、00フレームで’noise’に遷移、’noise’は10フレームで’drive’に遷移
        # self.params = {'drive': (100, 'noise'), 'noise': (10, 'drive')}
        self.params = {'drive': (100, 'drive')}

        self.steps = 0
        self.state = 'drive'
        # 最初は絶対driveなので
        self.noise_steer = 0
        self.last_throttle = 0
        # noise（関数）が渡された場合はnoiseを使用して、それ以外の場合は、ノイズを0.25~-0.25の間で生成
        self.noise_func = noise if noise else lambda: np.random.uniform(-0.25, 0.25)

        # 車両のスピードと方向を制御するPIDコントローラ
        self.speed_control = PIDController(K_P=0.5, K_I=0.5/20, K_D=0.1)
        self.turn_control = PIDController(K_P=0.75, K_I=1.0/20, K_D=0.0)

    def run_step(self, observations):
        # 1フレーム進める
        self.steps += 1

        # 現在の状態を保存
        last_status = self.state
        # 現在の状態に対応する継続フレーム数と次の状態を取得
        # params = {drive:(), noise:()}なので
        num_steps, next_state = self.params[self.state]
        # 親クラスのrun_stepを実行し制御コマンド取得
        real_control = super().run_step(observations)
        # ステアに応じてアクセルを減少
        real_control.throttle *= max((1.0 - abs(real_control.steer)), 0.25)

        # 車両の操作のためのクラス
        control = carla.VehicleControl()
        control.manual_gear_shift = False

        # noiseの場合はノイズがあるときのステアと前回のアクセルを適用
        if self.state == 'noise':
            control.steer = self.noise_steer
            control.throttle = self.last_throttle
        else:
            control.steer = real_control.steer
            control.throttle = real_control.throttle
            control.brake = real_control.brake

        # num_steps分のフレームが経過したら次の状態に遷移
        if self.steps == num_steps:
            self.steps = 0
            self.state = next_state
            self.noise_steer = self.noise_func()
            self.last_throttle = control.throttle

        # waypointと車両の位置を保存
        self.debug = {
                'waypoint': (self.waypoint.x, self.waypoint.y, self.waypoint.z),
                'vehicle': (self.vehicle.x, self.vehicle.y, self.vehicle.z)
                }

        return control, self.road_option, last_status, real_control


def get_episode(env, params):
    data = list()
    # エピソードごとの進捗表示
    progress = tqdm.tqdm(range(params.frames_per_episode), desc='Frame')
    # スタート位置とゴール位置をランダムに選択 pose_tasksはbenchmark.pyで定義
    # pose_tasksの長さ文のインデックスをランダムに選択
    start, target = env.pose_tasks[np.random.randint(len(env.pose_tasks))]
    # cuはcarla_utilsのキーを取得し、ランダムに選択
    env_params = {
            'weather': np.random.choice(list(cu.TRAIN_WEATHERS.keys())),
            'start': start,
            'target': target,
            'n_pedestrians': params.n_pedestrians,
            'n_vehicles': params.n_vehicles,
            }

    env.init(**env_params)
    # ゴール位置に到達したとみなす距離5m
    env.success_dist = 5.0

    # 車両制御用のエージェント作成
    agent = NoisyAgent(env)
    # エージェントのスタート位置とゴール位置を設定
    agent.set_route(env._start_pose.location, env._target_pose.location)

    # Real loop.
    # データが指定したフレームに到達する、ゴールに到達する、衝突するまで繰り返す
    while len(data) < params.frames_per_episode and not env.is_success() and not env.collided:
        for _ in range(params.frame_skip):
            # 時間を進める
            env.tick()

            # 観測データ取得（車両周辺の情報）
            observations = env.get_observations()
            # エージェントが制御コマンドを計算
            # controlエージェントの計算したステア、アクセル、ブレーキ。commandは方向命令（左、右、直進、追従）、real_controlは実際に環境に適用された制御
            control, command, last_status, real_control = agent.run_step(observations)
            agent_debug = agent.debug
            # 制御を適用
            env.apply_control(control)

            # 最新のcommandに上書き
            observations['command'] = command
            observations['control'] = control
            observations['real_control'] = real_control

            if not params.nodisplay:
                #デバッグを画面表示
                _debug(observations, agent_debug)

        observations['control'] = real_control
        # データを加工
        processed = cu.process(observations)

        # データを追加
        data.append(processed)
        # 進捗表示を更新
        progress.update(1)

    progress.close()

    # ゴールに到達していないかつ衝突していない、もしくはフレーム数が500未満の場合はNoneを返す
    if (not env.is_success() and not env.collided) or len(data) < 500:
        return None

    return data


def main(params):

    save_dir = Path(params.dataset_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    
    # 指定されたエピソード数だけデータを収集 tqdmで進捗表示
    for i in tqdm.tqdm(range(params.n_episodes), desc='Episode'):

        # make_suiteかカーラのシナリオセットアップ関数
        with make_suite('FullTown01-v1', port=params.port, planner=params.planner) as env:
            filepath = save_dir.joinpath('%03d' % i)

            if filepath.exists():
                print(f"[INFO] Episode {i} already exists. Skipping...")
                continue

            data = None

            # データ取得が成功するまでループ
            while data is None:
                print(f"[INFO] Collecting data for episode {i}")
                data = get_episode(env, params)

            # データが取得できた場合の確認
            if data:
                print(f"[INFO] Data collected for episode {i}. Frames: {len(data)}")

            # lmdbデータベースを作成
            lmdb_env = lmdb.open(str(filepath), map_size=int(1e10))
            n = len(data)

            # データをlmdbに保存
            with lmdb_env.begin(write=True) as txn:
                print(f"[INFO] Saving data for episode {i} to LMDB...")

                # エピソードのフレーム数を保存
                txn.put('len'.encode(), str(n).encode())

                # 各フレームのデータを保存
                for frame_index, x in enumerate(data):
                    if frame_index % 50 == 0:
                        print(f"[INFO] Saving frame {frame_index}/{n} for episode {i}")

                    txn.put(
                        ('rgb_%04d' % frame_index).encode(),
                        np.ascontiguousarray(x['rgb']).astype(np.uint8))
                    txn.put(
                        ('birdview_%04d' % frame_index).encode(),
                        np.ascontiguousarray(x['birdview']).astype(np.uint8))
                    txn.put(
                        ('measurements_%04d' % frame_index).encode(),
                        np.ascontiguousarray(x['measurements']).astype(np.float32))
                    txn.put(
                        ('control_%04d' % frame_index).encode(),
                        np.ascontiguousarray(x['control']).astype(np.float32))

                print(f"[INFO] Finished saving episode {i}")

            # 全エピソードのフレームの合計
            total += len(data)
            print(f"[INFO] Total frames so far: {total}")

    print(f"[INFO] Total frames collected across all episodes: {total}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--planner', type=str, choices=['old', 'new'], default='new')
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--n_vehicles', type=int, default=0)
    parser.add_argument('--n_pedestrians', type=int, default=0)
    parser.add_argument('--n_episodes', type=int, default=50)
    parser.add_argument('--frames_per_episode', type=int, default=4000)
    parser.add_argument('--frame_skip', type=int, default=1)
    parser.add_argument('--nodisplay', action='store_true', default=False)
    parser.add_argument('--port', type=int, default=2000)

    params = parser.parse_args()

    main(params)
