import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# 1. データの読み込みと前処理
# 画像データと数値データのパスを指定
image_folder = '/home/yoshi-22/LearningByCheating/datasets/images'
csv_file = '/home/yoshi-22/LearningByCheating/datasets/data.csv'
model_path = 'self_driving_input_assessment_model.h5'  # 学習済みモデルのパスを指定

# CSVファイルの読み込み
data = pd.read_csv(csv_file)

# 画像のぼやけ具合を計算する関数
def calculate_blur_score(image):
    # 画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # ラプラシアンフィルタを適用
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    # ブレ度合いを計算（値が小さいほどぼやけている）
    return laplacian_var

# ラベルの生成（0〜1のスコア）
# ここでは、衝突回数やレーン逸脱回数などを用いてスコアを計算します
def calculate_score(row, blur_score):
    # スコア計算の例（必要に応じて調整）
    max_collision = data['collision_count'].max()
    if max_collision == 0:
        max_collision = 1

    max_lane_deviation = data['lane_invasion_count'].max()
    if max_lane_deviation == 0:
        max_lane_deviation = 1

    collision_score = 1 - (row['collision_count'] / max_collision)
    lane_invasion_score = 1 - (row['lane_invasion_count'] / max_lane_deviation)

    # blur_scoreを正規化（0〜1にスケーリング）
    blur_score_normalized = min(blur_score / 1000.0, 1.0)  # 1000は適切なスケーリング値
    blur_score_normalized = 1 - blur_score_normalized  # 値が大きいほどぼやけていない

    # 速度スコア
    max_speed = data['current_speed'].max()
    if max_speed == 0:
        max_speed = 1
    speed_score = min(row['current_speed'] / max_speed, 1.0)

    # コマンドスコア
    now_command = row['vehicle_command']
    if now_command == 4:
        command_score = 1.0
    else:
        command_score = 0.5

    # 総合スコアの計算
    total_score = (
        collision_score * 0.2 +
        lane_invasion_score * 0.2 +
        blur_score_normalized * 0.2 +
        speed_score * 0.2 +
        command_score * 0.2
    )
    
    if np.isnan(total_score):
        total_score = 0.0
    
    return total_score

# 画像データの読み込みとぼやけ具合の計算
images = []
scores = []
for index, row in data.iterrows():
    img_name = row['image_filename']
    img_path = os.path.join(image_folder, img_name)
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))  # 画像サイズを統一
    images.append(img)
    # 画像のぼやけ具合を計算
    blur_score = calculate_blur_score(img)
    # スコアを計算
    score = calculate_score(row, blur_score)
    scores.append(score)

images = np.array(images)
labels = np.array(scores)

# ピクセル値を0〜1に正規化
images = images / 255.0

# 2. データの分割
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

if os.path.exists(model_path):
    print('loading model...')
    model = load_model(model_path)
else:
    # 3. モデルの構築
    input_shape = (224, 224, 3)
    image_input = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu')(image_input)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=image_input, outputs=output)

    # 4. モデルのコンパイル
    model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['mae'])

# 5. モデルの学習
model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_val, y_val)
)

# 6. モデルの評価
loss, mae = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation MAE: {mae}')

# 7. モデルの保存
model.save('self_driving_input_assessment_model.h5')
