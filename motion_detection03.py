import cv2
import numpy as np
import pandas as pd
from datetime import timedelta, datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import traceback

def show_settings_dialog():
    """設定ダイアログを表示し、設定値を返す"""
    dialog = tk.Tk()
    dialog.title("設定")
    dialog.geometry("400x450")
    
    # 設定値の初期化
    settings = {
        'screenshot_interval': tk.StringVar(value="1.0"),
        'data_collection_interval': tk.StringVar(value="0.5"),
        'detection_threshold': tk.StringVar(value="1"),  # デフォルト値を調整
        'tracking_threshold': tk.StringVar(value="1"),  # 追跡中の検出閾値
        'min_area': tk.StringVar(value="20"),  # 最小面積の設定
        'max_area': tk.StringVar(value="100"),  # 最大面積の設定
        'max_missing_frames': tk.StringVar(value="30")  # 最大追跡フレーム数
    }
    result = None
    
    def on_ok():
        nonlocal result
        try:
            # 値の検証
            screenshot = float(settings['screenshot_interval'].get())
            data_collection = float(settings['data_collection_interval'].get())
            threshold = float(settings['detection_threshold'].get())
            tracking_threshold = float(settings['tracking_threshold'].get())
            min_area = int(settings['min_area'].get())
            max_area = int(settings['max_area'].get())
            max_missing_frames = int(settings['max_missing_frames'].get())
            if (screenshot <= 0 or data_collection <= 0 or threshold <= 0 or
                tracking_threshold <= 0 or min_area <= 0 or max_area <= 0 or max_missing_frames <= 0):
                raise ValueError("値は正の数である必要があります")
                
            result = {
                'screenshot_interval': screenshot,
                'data_collection_interval': data_collection,
                'detection_threshold': threshold,
                'tracking_threshold': tracking_threshold,
                'min_area': min_area,
                'max_area': max_area,
                'max_missing_frames': max_missing_frames
            }
            dialog.quit()
        except ValueError:
            messagebox.showerror("エラー", "無効な入力値です。正の数を入力してください。")
    
    # GUI要素の配置
    tk.Label(dialog, text="スクリーンショット間隔（秒）:").grid(row=0, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['screenshot_interval']).grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="データ収集間隔（秒）:").grid(row=1, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['data_collection_interval']).grid(row=1, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="通常検出感度（大きいほど厳しく）:").grid(row=2, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['detection_threshold']).grid(row=2, column=1, padx=5, pady=5)

    tk.Label(dialog, text="追跡時検出感度:").grid(row=3, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['tracking_threshold']).grid(row=3, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="最小検出面積:").grid(row=4, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['min_area']).grid(row=4, column=1, padx=5, pady=5)

    tk.Label(dialog, text="最大検出面積:").grid(row=5, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['max_area']).grid(row=5, column=1, padx=5, pady=5)

    tk.Label(dialog, text="最大追跡フレーム数:").grid(row=6, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['max_missing_frames']).grid(row=6, column=1, padx=5, pady=5)
    
    # ボタン
    tk.Button(dialog, text="OK", command=on_ok).grid(row=7, column=0, padx=5, pady=40)
    tk.Button(dialog, text="キャンセル", command=dialog.quit).grid(row=7, column=1, padx=5, pady=40)
    
    # ダイアログを表示
    dialog.mainloop()
    
    # 結果を取得
    final_result = result
    dialog.destroy()
    return final_result

def draw_japanese_text(image, text, position, font_size=32):
    """日本語テキストを画像に描画する関数"""
    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image_pil)
    
    try:
        font = ImageFont.truetype('C:\\Windows\\Fonts\\meiryo.ttc', font_size)
    except:
        font = ImageFont.load_default()
    
    # 黒い縁取り（8方向）
    for offset_x, offset_y in [(x, y) for x in [-3,3] for y in [-3,3]]:
        draw.text((position[0] + offset_x, position[1] + offset_y), text, font=font, fill=(0, 0, 0))
    
    # 白い文字
    draw.text(position, text, font=font, fill=(255, 255, 255))
    
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

def resize_frame(frame, target_width=960):
    """フレームをリサイズする関数"""
    height, width = frame.shape[:2]
    if width > target_width:
        ratio = target_width / width
        new_height = int(height * ratio)
        return cv2.resize(frame, (target_width, new_height))
    return frame

def is_human_like(contour):
    """人らしい輪郭かどうかを判定する関数"""
    # 輪郭の境界矩形を取得
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h
    area = cv2.contourArea(contour)
    
    # アスペクト比と面積で判定（人らしい比率と大きさ）
    return 0.3 < aspect_ratio < 1.0

def process_video(filepath, settings):
    try:
        print("動画処理を開始します...")
        print(f"設定値: {settings}")
        
        cap = cv2.VideoCapture(str(filepath))
        
        if not cap.isOpened():
            print(f"エラー: 動画ファイルを開けませんでした: {filepath}")
            return
        
        # 動画の情報を取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = total_frames / fps  # 動画の長さ（秒）
        
        # 予想処理時間を計算（1フレームあたり約0.1秒と仮定）
        estimated_time = total_frames * 0.1  # 秒単位
        estimated_duration = timedelta(seconds=int(estimated_time))
        
        # 分析情報を表示
        print("\n===== 分析情報 =====")
        print(f"動画の長さ: {timedelta(seconds=int(video_duration))}")
        print(f"総フレーム数: {total_frames}")
        print(f"フレームレート: {fps}fps")
        print(f"予想処理時間: {estimated_duration}")
        
        # ユーザーに確認
        while True:
            response = input("\n分析を実行しますか？ (y/n): ").lower()
            if response in ['y', 'n']:
                break
            print("yまたはnで入力してください。")
        
        if response == 'n':
            print("分析を中止しました。")
            cap.release()
            return
        
        # 出力ディレクトリの設定
        current_datetime = datetime.now().strftime("%Y%m%d%H%M")
        output_dir = Path(filepath).parent / current_datetime
        output_dir.mkdir(exist_ok=True, parents=True)
        print(f"出力ディレクトリを作成しました: {output_dir}")
        
        print(f"出力ディレクトリを作成しました: {output_dir}")
        
        # 結果を保存するリスト
        results_data = []
        
        # フレーム間隔の設定
        frames_per_screenshot = int(fps * settings['screenshot_interval'])
        frames_per_data = int(fps * settings['data_collection_interval'])
        detection_threshold = settings['detection_threshold']
        tracking_threshold = settings['tracking_threshold']
        min_area = settings['min_area']
        max_missing_frames = settings['max_missing_frames']
        
        print(f"スクリーンショット間隔: {frames_per_screenshot}フレーム (1/{settings['screenshot_interval']}秒)")
        print(f"データ収集間隔: {frames_per_data}フレーム (1/{settings['data_collection_interval']}秒)")
        print(f"通常検出閾値: {detection_threshold}")
        print(f"追跡時検出閾値: {tracking_threshold}")
        print(f"最小検出面積: {min_area}")
        print(f"最大追跡フレーム数: {max_missing_frames}")
        
        avg = None
        frame_idx = 0
        start_time = datetime.now()
        should_stop = False
        
        # トラッキング用の変数とクラス
        class TrackedObject:
            def __init__(self, object_id, position):
                self.id = object_id
                self.positions = [position]  # 位置の履歴
                self.frames_tracked = 1      # 追跡フレーム数
                self.kalman = cv2.KalmanFilter(4, 2)  # 4次元状態ベクトル(x,y,dx,dy)、2次元測定(x,y)
                self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
                self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03
                self.kalman.statePost = np.array([[position[0]], [position[1]], [0], [0]], np.float32)
                self.missing_frames = 0
                
            def predict(self):
                prediction = self.kalman.predict()
                return (int(prediction[0]), int(prediction[1]))
                
            def update(self, position):
                measurement = np.array([[position[0]], [position[1]]], np.float32)
                self.kalman.correct(measurement)
                self.positions.append(position)
                self.frames_tracked += 1
                self.missing_frames = 0
                
            def is_valid(self):
                return self.missing_frames < max_missing_frames  # 設定値以上見失ったら無効
        
        tracked_objects = []  # [TrackedObject]
        next_obj_id = 0
        
        def calculate_distance(p1, p2):
            """2点間の距離を計算"""
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def get_contour_center(cnt):
            """輪郭の中心座標を取得"""
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
            return None
            
        def update_tracked_objects(current_contours):
            """トラッキング対象を更新"""
            nonlocal next_obj_id, tracked_objects
            
            # 現在の輪郭の中心点を計算
            current_centers = []
            for cnt in current_contours:
                center = get_contour_center(cnt)
                if center:
                    current_centers.append(center)
            
            # 各オブジェクトの次の位置を予測
            predictions = {obj.id: obj.predict() for obj in tracked_objects}
            
            # 既存のオブジェクトと新しい検出との対応付け
            used_centers = set()
            matched_objects = set()
            
            # 予測位置と実際の検出位置のマッチング
            for obj in tracked_objects:
                if not obj.is_valid():
                    continue
                    
                predicted_pos = predictions[obj.id]
                min_dist = float('inf')
                best_center = None
                
                for center in current_centers:
                    if center not in used_centers:
                        dist = calculate_distance(predicted_pos, center)
                        if dist < min_dist and dist < 100:  # 最大移動距離を制限
                            min_dist = dist
                            best_center = center
                
                if best_center:
                    used_centers.add(best_center)
                    matched_objects.add(obj.id)
                    obj.update(best_center)
                else:
                    obj.missing_frames += 1
            
            # 未使用の検出点を新規オブジェクトとして追加
            for center in current_centers:
                if center not in used_centers:
                    new_obj = TrackedObject(next_obj_id, center)
                    tracked_objects.append(new_obj)
                    next_obj_id += 1
            
            # 無効なオブジェクトを除去
            tracked_objects = [obj for obj in tracked_objects if obj.is_valid()]
        
        def on_close(event):
            nonlocal should_stop
            should_stop = True
        
        # ウィンドウを作成し、クローズイベントを設定
        cv2.namedWindow("Human Motion Detection")
        cv2.setWindowProperty("Human Motion Detection", cv2.WND_PROP_TOPMOST, 1)
        
        while not should_stop:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 180度回転
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            # フレームをリサイズ
            frame = resize_frame(frame, target_width=960)
                
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # ガウシアンブラーでノイズを軽減
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # 比較用のフレームを取得する
            if avg is None:
                avg = gray.copy().astype("float")
                continue
                
            # 現在のフレームと移動平均との差を計算
            cv2.accumulateWeighted(gray, avg, 0.6)
            frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
            
            # デルタ画像を閾値処理を行う
            # 通常の検出用の閾値処理
            thresh_normal = cv2.threshold(frameDelta, detection_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # 追跡中のオブジェクト用の閾値処理（より低い閾値）
            thresh_tracking = cv2.threshold(frameDelta, tracking_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # 最終的な二値化画像を初期化（通常の閾値で検出）
            thresh = thresh_normal.copy()
            
            # 追跡中のオブジェクトの周辺領域には低い閾値を適用
            for obj in tracked_objects:
                if obj.is_valid():
                    # 追跡中のオブジェクトの最後の位置を中心に矩形領域を定義
                    x, y = obj.positions[-1]
                    search_area = 100  # 探索範囲（ピクセル）
                    x1, y1 = max(0, x - search_area), max(0, y - search_area)
                    x2, y2 = min(thresh.shape[1], x + search_area), min(thresh.shape[0], y + search_area)
                    
                    # その領域内では低い閾値の結果を使用
                    thresh[y1:y2, x1:x2] = thresh_tracking[y1:y2, x1:x2]
            
            # ノイズ除去のためのモルフォロジー演算を強化
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.dilate(thresh, kernel, iterations=3)
            thresh = cv2.erode(thresh, kernel, iterations=2)
            
            # 画像の閾値に輪郭線を入れる
            contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 人らしい動きの輪郭のみを抽出
            significant_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > min_area and is_human_like(cnt):
                    significant_contours.append(cnt)
            
            movement_count = len(significant_contours)
            frame_with_contours = frame.copy()
            
            # トラッキング対象を更新
            update_tracked_objects(significant_contours)

            # 輪郭と追跡情報を描画
            frame_with_contours = frame.copy()
            for cnt in significant_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # トラッキング情報の表示
                center = get_contour_center(cnt)
                if center:
                    for obj in tracked_objects:
                        if calculate_distance(center, obj.positions[-1]) < 50:
                            # ID、フレーム数、予測位置の表示
                            predicted_pos = obj.predict()
                            cv2.circle(frame_with_contours, predicted_pos, 5, (0, 0, 255), -1)  # 予測位置を赤で表示
                            cv2.putText(frame_with_contours, f"ID:{obj.id}({obj.frames_tracked}f)",
                                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break
            
            # 現在の時間を計算
            current_time = timedelta(seconds=frame_idx/fps)
            
            # 設定された間隔でデータを記録
            if frame_idx % frames_per_data == 0:
                results_data.append({
                    '経過時間': str(current_time),
                    '検出人数': movement_count
                })
            
            # 設定された間隔で画像を保存
            if frame_idx % frames_per_screenshot == 0:
                # 動体数を画像に追記
                # 動体数を画像に追記
                text = f"検出人数：{movement_count}人"
                annotated_frame = draw_japanese_text(
                    frame_with_contours,
                    text,
                    (10, 30),
                    font_size=50
                )
                
                # 画像を保存
                video_time = str(current_time).replace(':', '').replace('.', '')
                image_path = output_dir / f"{video_time}_{movement_count}人.jpg"
                cv2.imwrite(str(image_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                print(f"\nスクリーンショットを保存しました: {image_path}")
                
                # 進捗表示
                progress = (frame_idx / total_frames) * 100
                elapsed_time = datetime.now() - start_time
                estimated_total_time = elapsed_time * (total_frames / (frame_idx + 1))
                remaining_time = estimated_total_time - elapsed_time
                print(f"\r処理進捗: {progress:.0f}% (残り時間: {str(remaining_time).split('.')[0]})", end="")
            
            frame_idx += 1  # フレームカウンタを先頭に移動
            
            # 画面表示
            cv2.imshow("Human Motion Detection", frame_with_contours)
            key = cv2.waitKey(30)
            if key == 27 or cv2.getWindowProperty("Human Motion Detection", cv2.WND_PROP_VISIBLE) < 1:
                should_stop = True
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # 実際の処理時間を計算
        total_time = datetime.now() - start_time
        print("\n処理完了!")
        print(f"実際の処理時間: {str(total_time).split('.')[0]}")
        
        if should_stop:
            print("\n処理を中止しました。")
            return
        
        if not results_data:
            print("警告: データが収集されませんでした。")
            return
        
        # 結果をDataFrameに変換
        df = pd.DataFrame(results_data)
        
        # 基本統計の計算
        avg_count = df['検出人数'].mean()
        max_count = df['検出人数'].max()
        min_count = df['検出人数'].min()
        
        # 統計情報の表示と保存
        print(f"\n===== 分析結果 =====")
        print(f"平均検出人数: {avg_count:.1f}")
        print(f"最大検出人数: {max_count}")
        print(f"最小検出人数: {min_count}")
        
        # 統計情報のDataFrame
        stats_df = pd.DataFrame({
            '統計項目': ['平均検出人数', '最大検出人数', '最小検出人数'],
            '値': [f"{avg_count:.1f}", f"{max_count}", f"{min_count}"]
        })
        
        # Excelファイルに出力
        excel_path = output_dir / f"human_detection_{current_datetime}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='詳細データ', index=False)
            stats_df.to_excel(writer, sheet_name='統計情報', index=False)
        
        print(f"\nExcelファイルを保存しました: {excel_path}")
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        print("詳細なエラー情報:")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        # ファイル選択ダイアログを表示
        video_file = filedialog.askopenfilename(
            title='動画ファイルを選択してください',
            filetypes=[
                ('動画ファイル', '*.mp4 *.avi *.mov *.wmv'),
                ('すべてのファイル', '*.*')
            ]
        )

        if video_file:  # ファイルが選択された場合
            print(f"選択された動画ファイル: {video_file}")
            video_path = Path(video_file)
            if video_path.exists():
                # 設定を取得
                settings = show_settings_dialog()
                if settings:
                    print(f"設定が完了しました: {settings}")
                    process_video(video_path, settings)
                else:
                    print("設定がキャンセルされました。")
            else:
                print(f"エラー: 動画ファイルが見つかりません: {video_path}")
        else:
            print("ファイルが選択されませんでした。")
            
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {str(e)}")
        print("詳細なエラー情報:")
        traceback.print_exc()