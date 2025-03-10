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
    dialog.geometry("400x250")
    
    # 設定値の初期化
    settings = {
        'screenshot_interval': tk.StringVar(value="1.0"),
        'data_collection_interval': tk.StringVar(value="0.5"),
        'detection_threshold': tk.StringVar(value="25"),  # デフォルト値を調整
        'min_area': tk.StringVar(value="2000")  # 最小面積の設定を追加
    }
    result = None
    
    def on_ok():
        nonlocal result
        try:
            # 値の検証
            screenshot = float(settings['screenshot_interval'].get())
            data_collection = float(settings['data_collection_interval'].get())
            threshold = int(settings['detection_threshold'].get())
            min_area = int(settings['min_area'].get())
            
            if screenshot <= 0 or data_collection <= 0 or threshold <= 0 or min_area <= 0:
                raise ValueError("値は正の数である必要があります")
                
            result = {
                'screenshot_interval': screenshot,
                'data_collection_interval': data_collection,
                'detection_threshold': threshold,
                'min_area': min_area
            }
            dialog.quit()
        except ValueError:
            messagebox.showerror("エラー", "無効な入力値です。正の数を入力してください。")
    
    # GUI要素の配置
    tk.Label(dialog, text="スクリーンショット間隔（秒）:").grid(row=0, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['screenshot_interval']).grid(row=0, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="データ収集間隔（秒）:").grid(row=1, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['data_collection_interval']).grid(row=1, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="検出感度（大きいほど厳しく）:").grid(row=2, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['detection_threshold']).grid(row=2, column=1, padx=5, pady=5)
    
    tk.Label(dialog, text="最小検出面積:").grid(row=3, column=0, padx=5, pady=5)
    tk.Entry(dialog, textvariable=settings['min_area']).grid(row=3, column=1, padx=5, pady=5)
    
    # ボタン
    tk.Button(dialog, text="OK", command=on_ok).grid(row=4, column=0, padx=5, pady=20)
    tk.Button(dialog, text="キャンセル", command=dialog.quit).grid(row=4, column=1, padx=5, pady=20)
    
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
    return 0.3 < aspect_ratio < 1.0 and area < 15000

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
        frames_per_screenshot = int(fps)  # 1秒間隔に固定
        frames_per_data = int(fps * 0.5)  # 0.5秒間隔に固定
        detection_threshold = settings['detection_threshold']
        min_area = settings['min_area']
        
        print(f"スクリーンショット間隔: {frames_per_screenshot}フレーム (1/{settings['screenshot_interval']}秒)")
        print(f"データ収集間隔: {frames_per_data}フレーム (1/{settings['data_collection_interval']}秒)")
        print(f"検出閾値: {detection_threshold}")
        print(f"最小検出面積: {min_area}")
        
        avg = None
        frame_idx = 0
        start_time = datetime.now()
        should_stop = False
        
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
            thresh = cv2.threshold(frameDelta, detection_threshold, 255, cv2.THRESH_BINARY)[1]
            
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
            
            # 検出された動体を矩形で囲む
            for cnt in significant_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame_with_contours, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 現在の時間を計算
            current_time = timedelta(seconds=frame_idx/fps)
            
            # 設定された間隔でデータを記録
            if frame_idx % frames_per_data == 0:
                results_data.append({
                    '経過時間': str(current_time),
                    '検出人数': movement_count
                })
            
            # 1秒ごとに画像を保存
            if frame_idx % frames_per_screenshot == 0:
                try:
                    # スクリーンショット用のフレームを準備
                    save_frame = frame.copy()
                    save_frame = cv2.rotate(save_frame, cv2.ROTATE_180)
                    
                    # 検出結果を描画
                    for cnt in significant_contours:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(save_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # 検出人数を画像に追記
                    text = f"検出人数：{movement_count}人"
                    annotated_frame = draw_japanese_text(
                        save_frame,
                        text,
                        (10, 30),
                        font_size=50
                    )
                    
                    # 画像を保存
                    video_time = str(current_time).replace(':', '').replace('.', '')
                    image_path = output_dir / f"{video_time}_{movement_count}人.jpg"
                    success = cv2.imwrite(str(image_path), annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    
                    if success:
                        print(f"\nスクリーンショットを保存しました: {image_path}")
                    else:
                        print(f"\nスクリーンショットの保存に失敗しました: {image_path}")
                except Exception as e:
                    print(f"\nスクリーンショット保存エラー: {str(e)}")
                
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