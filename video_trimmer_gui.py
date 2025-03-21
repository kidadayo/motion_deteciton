import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import threading
import os

class VideoTrimmerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("動画トリマー")
        self.root.geometry("600x300")
        
        # 変数の初期化
        self.input_path = tk.StringVar()
        self.start_time = tk.StringVar(value="0.0")
        self.end_time = tk.StringVar()
        self.cap = None
        
        self.setup_ui()
    
    def setup_ui(self):
        # 入力ファイル選択
        input_frame = ttk.LabelFrame(self.root, text="入力動画", padding=10)
        input_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).pack(side="left", padx=5)
        ttk.Button(input_frame, text="参照...", command=self.select_input_file).pack(side="left")
        
        # 時間設定
        time_frame = ttk.LabelFrame(self.root, text="時間設定", padding=10)
        time_frame.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(time_frame, text="開始時間（秒）:").pack(side="left")
        ttk.Entry(time_frame, textvariable=self.start_time, width=10).pack(side="left", padx=5)
        
        ttk.Label(time_frame, text="終了時間（秒）:").pack(side="left")
        ttk.Entry(time_frame, textvariable=self.end_time, width=10).pack(side="left", padx=5)
        
        # 動画情報表示
        self.info_label = ttk.Label(self.root, text="")
        self.info_label.pack(pady=5)
        
        # 出力ファイル情報
        self.output_label = ttk.Label(self.root, text="")
        self.output_label.pack(pady=5)
        
        # 実行ボタン
        ttk.Button(self.root, text="トリミング実行", command=self.trim_video).pack(pady=10)
        
        # プログレスバー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            self.root, 
            variable=self.progress_var, 
            maximum=100
        )
        self.progress_bar.pack(fill="x", padx=10, pady=5)
    
    def select_input_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("動画ファイル", "*.mp4 *.avi *.mov"), ("すべてのファイル", "*.*")]
        )
        if file_path:
            if os.path.exists(file_path):
                self.input_path.set(file_path)
                self.load_video_info()
                self.update_output_path()
            else:
                messagebox.showerror("エラー", f"指定したファイルが見つかりません：\n{file_path}")
    
    def generate_output_path(self, input_path):
        directory = os.path.dirname(input_path)
        filename = os.path.basename(input_path)
        name, ext = os.path.splitext(filename)
        
        # 新しいファイル名のベース
        output_base = os.path.join(directory, f"{name}_trimming")
        
        # 常に.mp4拡張子を使用
        counter = 1
        while True:
            output_path = f"{output_base}{counter:02d}.mp4"
            if not os.path.exists(output_path):
                return output_path
            counter += 1
    
    def update_output_path(self):
        if self.input_path.get():
            output_path = self.generate_output_path(self.input_path.get())
            self.output_label.config(text=f"出力ファイル: {os.path.basename(output_path)}")
    
    def load_video_info(self):
        if self.cap is not None:
            self.cap.release()
        
        input_path = self.input_path.get()
        if not os.path.exists(input_path):
            messagebox.showerror("エラー", f"ファイルが見つかりません：\n{input_path}")
            return
        
        self.cap = cv2.VideoCapture(input_path)
        if self.cap.isOpened():
            # 動画の情報を取得して表示
            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # ファイルサイズを取得（MB単位）
            file_size = os.path.getsize(input_path) / (1024 * 1024)
            
            info_text = f"動画情報: {width}x{height}, {fps:.1f}fps, 長さ: {duration:.1f}秒, サイズ: {file_size:.1f}MB"
            self.info_label.config(text=info_text)
            
            # 終了時間を動画の長さに設定
            self.end_time.set(f"{duration:.1f}")
        else:
            messagebox.showerror("エラー", f"動画ファイルを開けません：\n{input_path}")
    
    def trim_video(self):
        input_path = self.input_path.get()
        if not input_path:
            messagebox.showerror("エラー", "入力動画を選択してください。")
            return
        
        if not os.path.exists(input_path):
            messagebox.showerror("エラー", f"入力ファイルが見つかりません：\n{input_path}")
            return
        
        try:
            start_time = float(self.start_time.get())
            end_time = float(self.end_time.get())
        except ValueError:
            messagebox.showerror("エラー", "開始時間と終了時間は数値で入力してください。")
            return
        
        if start_time >= end_time:
            messagebox.showerror("エラー", "開始時間は終了時間より前である必要があります。")
            return
        
        # トリミング処理を別スレッドで実行
        threading.Thread(target=self.trim_video_thread).start()
    
    def trim_video_thread(self):
        try:
            start_time = float(self.start_time.get())
            end_time = float(self.end_time.get())
            input_path = self.input_path.get()
            output_path = self.generate_output_path(input_path)
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            total_frames = end_frame - start_frame
            
            # mp4vコーデックを使用（最も互換性が高い）
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )
            
            if not out.isOpened():
                # コーデック情報をデバッグ出力
                error_info = (
                    f"出力ファイルを作成できません。\n"
                    f"コーデック: mp4v\n"
                    f"出力パス: {output_path}\n"
                    f"FPS: {fps}\n"
                    f"サイズ: {width}x{height}"
                )
                raise Exception(error_info)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            current_frame = 0
            while current_frame <= total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                out.write(frame)
                current_frame += 1
                
                # 進捗更新
                progress = (current_frame / total_frames) * 100
                self.root.after(0, self.progress_var.set, progress)
            
            cap.release()
            out.release()
            
            if os.path.exists(output_path):
                self.root.after(0, self.trim_completed, output_path)
            else:
                raise Exception("出力ファイルが生成されませんでした。")
            
        except Exception as e:
            self.root.after(0, self.show_error, f"エラーが発生しました：\n{str(e)}")
    
    def show_error(self, message):
        self.progress_var.set(0)
        messagebox.showerror("エラー", message)
    
    def trim_completed(self, output_path):
        self.progress_var.set(0)
        # 出力ファイルのサイズを取得（MB単位）
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        messagebox.showinfo("完了", f"動画のトリミングが完了しました。\n保存先: {output_path}\nファイルサイズ: {file_size:.1f}MB")
        # 出力パスの表示を更新
        self.update_output_path()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoTrimmerGUI(root)
    root.mainloop()