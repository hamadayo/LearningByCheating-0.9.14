import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# 1. 学習済みモデルのロード
model_path = 'self_driving_input_assessment_model.h5'  # 学習済みモデルのパスを指定
model = load_model(model_path)

# 2. 画像の前処理関数を定義
def preprocess_image(image_path):
    # 画像を読み込む
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"not found: {image_path}")
    
    # 画像をリサイズ（学習時と同じサイズに合わせる）
    img = cv2.resize(img, (224, 224))  # 幅384、高さ160
    # ピクセル値を0〜1に正規化
    img = img / 255.0
    # 次元を追加（バッチサイズの次元）
    img = np.expand_dims(img, axis=0)
    return img

# 3. 信頼度を予測する関数を定義
def predict_confidence(image_path):
    # 画像を前処理
    preprocessed_img = preprocess_image(image_path)
    # モデルを使用して予測
    prediction = model.predict(preprocessed_img)
    # 信頼度スコアを取得（0〜1の範囲）
    confidence_score = prediction[0][0]
    return confidence_score

# 4. テスト用の画像パスを指定
test_image_path = 'path_to_test_image.png'  # テスト画像のパスを指定

# 5. 信頼度を予測して表示
try:
    score = predict_confidence(test_image_path)
    print(f" reliable score: {score:.4f}")
except FileNotFoundError as e:
    print(e)
