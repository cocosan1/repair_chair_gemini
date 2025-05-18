# -*- coding: utf-8 -*-

# --- 1. Import Libraries ---
import os
import io
import pandas as pd
import json
import google.generativeai as genai
from PIL import Image
import streamlit as st
import matplotlib.pyplot as plt
import japanize_matplotlib
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity

# --- Streamlit UI ---
st.set_page_config(layout="centered")

st.title("チェア画像検索app")
st.markdown("アップロードされたチェアの画像と類似のチェアを検索・表示します。")

# # --- 2. Configuration ---
# ファイルパスはStreamlitアプリのルートからの相対パスを想定
EXCEL_CLASSIFICATION_PATH = "chair_classification_results.xlsx"
EXCEL_ALL_PATH = "chair_all.xlsx"
IMAGE_FOLDER_PATH = "images" # Streamlitアプリ内の 'images' フォルダ
PRICE_LIST_EXCEL_PATH = "修理価格一覧app用.xlsx"
COMMENT_EXCEL_PATH = "comment.xlsx"
NOTES_EXCEL_PATH = "修理注意点.xlsx"

# Classification items #
classification_items = {
    "背のデザイン": [
        "籐張り（籐が張ってあれば縦桟があっても籐張りで）",
        "布又は革張り（布又は革が張ってあれば縦桟があっても布又は革張りで）",
        "横板1枚",
        "横桟（横板2枚以上。横桟と縦桟が両方ある場合は横桟扱い。）",
        "縦桟",
        "その他"
    ],
    "座面": ["板", "張り"],
    "肘": ["有", "無"]
}

# --- 3. Model Loading (Cached) ---
MODULE_HANDLE = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/5"
IMAGE_SIZE = (224, 224)

@st.cache_resource # Cache the model loading
def load_embedding_model():
    """Loads the TensorFlow Hub image embedding model."""
    success_message = None
    try:
        model = hub.load(MODULE_HANDLE)
        success_message = "✅ 画像埋め込みモデルのロード完了。"
        return model, success_message
    except Exception as e:
        st.error(f"🚨 画像埋め込みモデルのロードエラー: {e}")
        return None, None

# --- 4. API Key and Gemini Model Setup (Cached) ---
@st.cache_resource # Cache the Gemini model initialization
def configure_gemini():
    """Configures Gemini API and initializes the model."""
    api_key = None
    success_messages = []
    if 'GEMINI_API_KEY' in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
        try:
            genai.configure(api_key=api_key)
            success_messages.append("✅ Gemini API Keyの設定完了。")
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            success_messages.append(f"✅ Geminiモデル ({model.model_name}) の初期化完了。")
            return model, success_messages
        except Exception as e:
            st.error(f"🚨 Geminiの設定またはモデル初期化エラー: {e}")
            return None, None
    else:
        return None, None

# --- 5. Helper Functions ---
def build_gemini_prompt(items_dict):
    prompt = "Analyze the provided image of a chair and classify it according to the following criteria.\n"
    prompt += "For each criterion, choose EXACTLY ONE option from the provided list.\n"
    prompt += "Output the results ONLY as a valid JSON object with the keys \"背のデザイン\", \"座面\", and \"肘\".\n\n"
    prompt += "Classification Criteria:\n"
    for key, options in items_dict.items():
        prompt += f'- {key}: {options}\n'

    prompt += '\nExample Output Format:\n{\n'
    example_items = []
    for key, options in items_dict.items():
        if options:
            example_value = json.dumps(options[0], ensure_ascii=False)[1:-1]
            example_items.append(f'  "{key}": "{example_value}"')
        else:
            example_items.append(f'  "{key}": "（選択肢例なし）"')

    prompt += ",\n".join(example_items)
    prompt += '\n}\n'

    prompt += "Provide only the JSON object in your response."
    return prompt

def validate_gemini_response(response_text, items_dict):
    try:
        if response_text.strip().startswith("```json"): response_text = response_text.strip()[7:-3].strip()
        elif response_text.strip().startswith("```"): response_text = response_text.strip()[3:-3].strip()
        parsed_json = json.loads(response_text)
        if not isinstance(parsed_json, dict): return None, "Error: Gemini response is not a JSON object."
        res = {}
        missing, invalid = [], {}
        for k, opts in items_dict.items():
            if k not in parsed_json: missing.append(k)
            elif parsed_json[k] not in opts: invalid[k] = parsed_json[k]
            else: res[k] = parsed_json[k]
        if missing: return None, f"Error: Missing keys: {', '.join(missing)}"
        if invalid: return None, f"Error: Invalid values: {invalid}"
        return res, None
    except json.JSONDecodeError: return None, f"Error: Could not decode JSON. Response:\n{response_text}"
    except Exception as e: return None, f"Error during validation: {e}"

def classify_image_with_gemini(img_data, items_dict):
    if not gemini_model: return None, "Error: Gemini model not initialized."
    img_pil = Image.open(io.BytesIO(img_data))
    prompt = build_gemini_prompt(items_dict)
    try:
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(
            [prompt, img_pil],
            generation_config=genai.types.GenerationConfig(temperature=0.1),
            safety_settings=safety_settings,
            stream=False
        )
        if not response.parts:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            error_msg = f"Error: Content blocked (Reason: {block_reason})" if block_reason != 0 else "Error: Empty response."
            st.warning(f"Gemini分類エラー: {error_msg}")
            return None, error_msg
        return validate_gemini_response(response.text, items_dict)
    except Exception as e:
        st.error(f"Gemini API呼び出しエラー（分類）: {e}")
        return None, f"Error calling Gemini API for classification: {e}"

def load_and_preprocess_image(image_bytes_or_path, target_size):
    try:
        if isinstance(image_bytes_or_path, bytes):
            img = Image.open(io.BytesIO(image_bytes_or_path)).convert('RGB')
        else:
            if not os.path.exists(image_bytes_or_path):
                st.error(f"画像ファイルが見つかりません: {image_bytes_or_path}")
                return None
            img = Image.open(image_bytes_or_path).convert('RGB')
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        return tf.expand_dims(img_array, axis=0)
    except Exception as e:
        st.error(f"画像読み込み/前処理エラー ({'bytes input' if isinstance(image_bytes_or_path, bytes) else image_bytes_or_path}): {e}")
        return None

def get_image_embedding(image_tensor):
    if embedding_model is None:
        st.error("🚨 画像埋め込みモデルがロードされていません。")
        return None
    if image_tensor is None:
        st.warning("⚠️ 埋め込み計算のための画像テンソルがありません。")
        return None
    try:
        features = embedding_model(image_tensor)
        return features.numpy()
    except Exception as e:
        st.error(f"🚨 画像埋め込みの生成中にエラーが発生しました: {e}")
        return None

def calculate_similarity(embedding1, embedding2):
    if embedding1 is None or embedding2 is None: return 0.0
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def get_gemini_similarity_score(target_img_data, candidate_img_data):
    if not gemini_model: return None
    if not target_img_data or not candidate_img_data: return None
    try:
        target_img_pil = Image.open(io.BytesIO(target_img_data))
        candidate_img_pil = Image.open(io.BytesIO(candidate_img_data))
        prompt = """提供された2つの椅子の画像を分析してください。
        外観（形状、スタイル、ディテール）のみに基づいて、視覚的な類似度を評価してください。
        0.00（全く似ていない）から1.00（同一またはほぼ同一）までの単一の浮動小数点数のみを出力してください。
        他のテキスト、説明、書式設定は一切含めないでください。数値だけをお願いします。

        出力例:
        0.75
        """
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = gemini_model.generate_content(
            [prompt, target_img_pil, candidate_img_pil],
            generation_config=genai.types.GenerationConfig(temperature=0.0),
            safety_settings=safety_settings,
            stream=False
        )
        if not response.parts:
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else "Unknown"
            error_msg = f"コンテンツブロック (Reason: {block_reason if block_reason != 0 else 'Empty response'})"
            print(f"Gemini Sim Error: {error_msg}")
            return None
        response_text = response.text.strip()
        try:
            import re
            match = re.search(r"[-+]?\d*\.\d+|\d+", response_text)
            if match:
                score = float(match.group(0))
                if 0.0 <= score <= 1.0: return score
                else:
                    print(f"Gemini Sim Error: スコアが範囲外 ({score}). Response: '{response_text}'")
                    return None
            else:
                print(f"Gemini Sim Error: レスポンスから数値を抽出できませんでした。 Response: '{response_text}'")
                return None
        except ValueError:
            print(f"Gemini Sim Error: スコアを数値に変換できませんでした。 Response: '{response_text}'")
            return None
    except FileNotFoundError:
        print(f"Gemini Sim Error: 画像ファイルの読み込みに失敗しました (内部エラー)。")
        return None
    except Exception as e:
        print(f"Gemini Sim Error: 予期せぬエラー: {type(e).__name__}: {e}")
        return None

# --- 6. Search and Rank Function (Modified) ---
def perform_search_and_rank(search_classifications, target_image_data):
    """Filters, ranks by embedding, re-ranks top 8 by Gemini, shows top 3. Returns detailed lists for debugging."""
    log_messages = []
    fig = None
    filtered_filenames = []
    embedding_ranked_results = []
    gemini_ranked_results = [] # 最終結果リストも初期化しておく

    log_messages.append("---")
    log_messages.append("ステップ3: 類似画像検索 & ランキング処理 開始")
    log_messages.append("\n🔄 分類結果に基づいて候補画像をフィルタリング中...")

    if not os.path.exists(EXCEL_CLASSIFICATION_PATH):
        log_messages.append(f"❌ エラー: 分類用Excelファイルが見つかりません: {EXCEL_CLASSIFICATION_PATH}")
        return None, None, log_messages, None, None
    try:
        df = pd.read_excel(EXCEL_CLASSIFICATION_PATH)
        required_cols = list(classification_items.keys()) + ['ファイル名']
        if not all(col in df.columns for col in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            log_messages.append(f"❌ エラー: Excelに必要な列がありません: {missing}")
            return None, None, log_messages, None, None
        log_messages.append(f"📊 分類用Excel読み込み完了: {EXCEL_CLASSIFICATION_PATH} ({len(df)}件)")
    except Exception as e:
        log_messages.append(f"❌ エラー: 分類用Excel読み込みエラー: {e}")
        return None, None, log_messages, None, None

    mask = pd.Series([True] * len(df))
    try:
        for key, value in search_classifications.items():
            if key in df.columns:
                mask &= (df[key].astype(str).str.strip() == str(value).strip())
            else:
                log_messages.append(f"⚠️ Excelに '{key}' 列がないためフィルタリングをスキップします。")
        matching_df = df[mask].copy()
    except Exception as e:
        log_messages.append(f"❌ エラー: Excelフィルタリングエラー: {e}")
        return None, None, log_messages, None, None

    if matching_df.empty:
        log_messages.append("ℹ️ 分類条件に一致する候補画像は見つかりませんでした。")
        return None, None, log_messages, [], []

    if 'ファイル名' not in matching_df.columns:
         log_messages.append("❌ エラー: フィルタリング後のデータに 'ファイル名' 列がありません。")
         return None, None, log_messages, None, None

    filtered_filenames = matching_df['ファイル名'].dropna().astype(str).str.strip().tolist()
    log_messages.append(f"\n✅ {len(filtered_filenames)}件の候補画像が見つかりました（分類フィルター後）。")

    if not embedding_model:
         log_messages.append("\n⚠️ 画像埋め込みモデルがロードされていないため、類似度ランキングをスキップします。")
         return None, None, log_messages, filtered_filenames, None

    log_messages.append("\n⏳ 1. 埋め込みベクトルによる類似度を計算中...")
    start_sim_time = time.time()
    target_image_tensor = load_and_preprocess_image(target_image_data, IMAGE_SIZE)
    if target_image_tensor is None:
         log_messages.append("❌ エラー: ターゲット画像を処理できませんでした。")
         return None, None, log_messages, filtered_filenames, None
    target_embedding = get_image_embedding(target_image_tensor)

    if target_embedding is None:
        log_messages.append("❌ エラー: ターゲット画像の埋め込みを生成できませんでした。")
        return None, None, log_messages, filtered_filenames, None

    similarity_results_temp = []
    processed_count = 0
    if not os.path.isdir(IMAGE_FOLDER_PATH):
        log_messages.append(f"❌ エラー: 画像フォルダが見つかりません: {IMAGE_FOLDER_PATH}")
        return None, None, log_messages, filtered_filenames, None

    for filename in filtered_filenames:
        img_path = os.path.join(IMAGE_FOLDER_PATH, filename)
        if not os.path.exists(img_path): continue

        candidate_tensor = load_and_preprocess_image(img_path, IMAGE_SIZE)
        if candidate_tensor is not None:
            candidate_embedding = get_image_embedding(candidate_tensor)
            if candidate_embedding is not None:
                try:
                    similarity_score = calculate_similarity(target_embedding, candidate_embedding)
                    similarity_results_temp.append({"filename": filename, "path": img_path, "score": float(similarity_score)})
                    processed_count += 1
                except Exception as sim_e:
                    log_messages.append(f"   - ⚠️ 埋め込み類似度計算エラー ({filename}): {sim_e}")

    end_sim_time = time.time()
    log_messages.append(f"✅ 埋め込み類似度計算完了 ({processed_count}/{len(filtered_filenames)}件処理)。時間: {end_sim_time - start_sim_time:.2f} 秒")

    if not similarity_results_temp:
         log_messages.append("ℹ️ 埋め込み類似度を計算できた候補画像がありませんでした。")
         return None, None, log_messages, filtered_filenames, []

    similarity_results_temp.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    embedding_ranked_results = similarity_results_temp[:8]
    top_8_candidates = embedding_ranked_results

    if not top_8_candidates:
        log_messages.append("ℹ️ 埋め込み類似度でランク付けできる候補がありませんでした。")
        return None, None, log_messages, filtered_filenames, []

    log_messages.append(f"\n✅ 埋め込み類似度 上位{len(top_8_candidates)}件を取得。")

    log_messages.append(f"\n⏳ 2. 上位{len(top_8_candidates)}件について、Geminiによる視覚的類似度を評価中 (時間がかかる場合があります)...")
    gemini_ranked_results_temp = []
    start_gemini_time = time.time()
    gemini_api_call_count = 0

    for rank, candidate in enumerate(top_8_candidates):
        log_messages.append(f"   - Gemini評価中 ({rank+1}/{len(top_8_candidates)}): {candidate['filename']}...")
        try:
            with open(candidate['path'], 'rb') as f: candidate_image_data = f.read()
            gemini_score = get_gemini_similarity_score(target_image_data, candidate_image_data)
            gemini_api_call_count += 1
            candidate['gemini_score'] = gemini_score
            gemini_ranked_results_temp.append(candidate)
            log_messages.append(f"   - Gemini評価完了 ({rank+1}/{len(top_8_candidates)}): {candidate['filename']} -> スコア: {gemini_score if gemini_score is not None else '評価失敗/エラー'}")
        except FileNotFoundError:
            log_messages.append(f"   - ❌ エラー: Gemini評価用 候補画像ファイルが見つかりません: {candidate['path']}")
            candidate['gemini_score'] = None
            gemini_ranked_results_temp.append(candidate)
        except Exception as e:
            log_messages.append(f"   - ❌ Gemini評価中にエラー ({candidate['filename']}): {type(e).__name__}: {e}")
            candidate['gemini_score'] = None
            gemini_ranked_results_temp.append(candidate)

    end_gemini_time = time.time()
    total_gemini_time = end_gemini_time - start_gemini_time
    log_messages.append(f"✅ Gemini類似度評価完了 ({gemini_api_call_count}回 APIコール)。時間: {total_gemini_time:.2f} 秒")

    gemini_ranked_results_temp.sort(key=lambda x: x.get('gemini_score', -1), reverse=True)
    gemini_ranked_results = gemini_ranked_results_temp

    log_messages.append("\n✅ 上位3件の画像表示を準備中...")
    fig = None
    final_top_3_for_figure = gemini_ranked_results[:3]

    if not final_top_3_for_figure:
         log_messages.append("ℹ️ 表示できる上位画像がありません。")
    else:
        n_images_to_display = len(final_top_3_for_figure)
        try:
            fig, axes = plt.subplots(1, n_images_to_display, figsize=(n_images_to_display * 4, 5))
            if n_images_to_display == 1: axes = [axes]
            for i, result in enumerate(final_top_3_for_figure):
                    img = Image.open(result["path"])
                    axes[i].imshow(img)
                    embed_score_str = f"{result.get('score', 'N/A'):.3f}" if isinstance(result.get('score'), (int, float)) else "N/A"
                    gemini_score_val = result.get('gemini_score')
                    gemini_score_str = f"{gemini_score_val:.3f}" if isinstance(gemini_score_val, (int, float)) else "評価失敗" if gemini_score_val is None else "N/A"
                    title = f"Rank {i+1}: {result['filename']}\nGemini Sim: {gemini_score_str}\nEmbed Sim: {embed_score_str}"
                    axes[i].set_title(title, fontsize=14)
                    axes[i].axis('off')
            plt.tight_layout(pad=2.0)
            log_messages.append("✅ 上位3画像の準備完了。")
        except Exception as e:
            log_messages.append(f"❌ エラー: 上位3画像の準備中にエラー: {type(e).__name__}: {e}")
            fig = None
    return gemini_ranked_results, fig, log_messages, filtered_filenames, embedding_ranked_results

# --- 7. Display Product Information ---
def display_product_info(product_number):
    st.markdown("---")
    st.subheader(f"品番 '{product_number}' の詳細情報")
    if not os.path.exists(EXCEL_ALL_PATH):
        st.error(f"❌ エラー: 詳細情報Excelファイルが見つかりません: {EXCEL_ALL_PATH}")
        return
    try:
        df_all = pd.read_excel(EXCEL_ALL_PATH)
    except Exception as e:
        st.error(f"❌ エラー: 詳細情報Excel読み込みエラー: {e}")
        return
    if '品番' not in df_all.columns:
        st.error(f"❌ エラー: 詳細情報Excelに '品番' 列が見つかりません。")
        return
    try:
        df_all['品番_str'] = df_all['品番'].astype(str).str.strip()
        search_term = str(product_number).strip()
        matching_row = df_all[df_all['品番_str'] == search_term]
        if matching_row.empty:
            st.warning(f"ℹ️ 品番 '{product_number}' に一致する情報は {EXCEL_ALL_PATH} に見つかりませんでした。")
        else:
            product_data = matching_row.iloc[0]
            if '品番_str' in product_data.index: product_data = product_data.drop('品番_str')
            product_data = product_data.fillna("").astype(str)
            display_df = product_data.reset_index()
            display_df.columns = ['項目', '内容']
            st.table(display_df.set_index('項目'))
    except Exception as e:
         st.error(f"❌ エラー: 詳細情報の検索または表示中にエラーが発生しました: {e}")

    st.markdown("---")
    st.subheader("関連コメント")
    if not os.path.exists(COMMENT_EXCEL_PATH):
        st.info(f"ℹ️ コメントファイル ({COMMENT_EXCEL_PATH}) が見つかりません。")
    else:
        try:
            df_comment = pd.read_excel(COMMENT_EXCEL_PATH)
            if '品番' not in df_comment.columns:
                st.error(f"❌ エラー: コメントExcel ({COMMENT_EXCEL_PATH}) に '品番' 列がありません。")
            else:
                try:
                    df_comment['品番_str'] = df_comment['品番'].astype(str).str.strip()
                    search_term_comment = str(product_number).strip()
                    matching_rows_comment = df_comment[df_comment['品番_str'] == search_term_comment]
                    if matching_rows_comment.empty:
                        st.info(f"ℹ️ 品番 '{product_number}' のコメントは {COMMENT_EXCEL_PATH} に見つかりませんでした。")
                    else:
                        for index, comment_row in matching_rows_comment.iterrows():
                            if '品番_str' in comment_row.index: comment_row = comment_row.drop('品番_str')
                            comment_data = comment_row.fillna("").astype(str)
                            comment_display_df = comment_data.reset_index()
                            comment_display_df.columns = ['項目', '内容']
                            st.table(comment_display_df.set_index('項目'))
                            st.markdown("---") # コメントごとに区切り線
                except Exception as e:
                     st.error(f"❌ エラー: コメント情報の検索または表示中にエラーが発生しました: {e}")
        except Exception as e:
            st.error(f"❌ エラー: コメントExcel ({COMMENT_EXCEL_PATH}) の読み込みエラー: {e}")

# --- 8. Initialize Session State ---
if 'uploaded_file_info' not in st.session_state: st.session_state['uploaded_file_info'] = None
if 'classification_result' not in st.session_state: st.session_state['classification_result'] = None
if 'error_msg' not in st.session_state: st.session_state['error_msg'] = None
if 'top_8_results' not in st.session_state: st.session_state['top_8_results'] = None
if 'search_figure' not in st.session_state: st.session_state['search_figure'] = None
if 'selected_product_number' not in st.session_state: st.session_state['selected_product_number'] = None
if 'search_logs' not in st.session_state: st.session_state['search_logs'] = None
if 'filtered_filenames_list' not in st.session_state: st.session_state['filtered_filenames_list'] = None
if 'embedding_ranked_list' not in st.session_state: st.session_state['embedding_ranked_list'] = None

# --- モデルとAPIの初期化 ---
loading_message_embed = f"🔄 画像埋め込みモデルをロード中: {MODULE_HANDLE}"
embedding_model, embed_load_msg = load_embedding_model()
gemini_model, gemini_init_msgs = configure_gemini()

# --- エラーチェック ---
api_key_present = 'GEMINI_API_KEY' in st.secrets
files_present = (
    os.path.exists(EXCEL_CLASSIFICATION_PATH) and
    os.path.exists(EXCEL_ALL_PATH) and
    os.path.isdir(IMAGE_FOLDER_PATH) and
    os.path.exists(COMMENT_EXCEL_PATH) and
    os.path.exists(PRICE_LIST_EXCEL_PATH) and
    os.path.exists(NOTES_EXCEL_PATH)
)
models_ready = gemini_model is not None and embedding_model is not None

# --- エラーメッセージ表示 ---
if not api_key_present: st.error("🚨 **設定エラー:** Streamlit Secretsに 'GEMINI_API_KEY' が設定されていません。")
if not files_present:
    missing_files_msg = "🚨 **ファイルエラー:** 必要なファイルまたはフォルダが見つかりません。\n"
    missing_files_msg += f"- 分類Excel: {EXCEL_CLASSIFICATION_PATH} {'✅' if os.path.exists(EXCEL_CLASSIFICATION_PATH) else '❌'}\n"
    missing_files_msg += f"- 詳細Excel: {EXCEL_ALL_PATH} {'✅' if os.path.exists(EXCEL_ALL_PATH) else '❌'}\n"
    missing_files_msg += f"- 画像フォルダ: {IMAGE_FOLDER_PATH} {'✅' if os.path.isdir(IMAGE_FOLDER_PATH) else '❌'}\n"
    missing_files_msg += f"- コメントExcel: {COMMENT_EXCEL_PATH} {'✅' if os.path.exists(COMMENT_EXCEL_PATH) else '❌'}\n"
    missing_files_msg += f"- 修理価格Excel: {PRICE_LIST_EXCEL_PATH} {'✅' if os.path.exists(PRICE_LIST_EXCEL_PATH) else '❌'}\n"
    missing_files_msg += f"- 注意点Excel: {NOTES_EXCEL_PATH} {'✅' if os.path.exists(NOTES_EXCEL_PATH) else '❌'}"
    st.error(missing_files_msg)

# --- 初期化ログ表示 ---
initialization_messages = [loading_message_embed]
if embed_load_msg: initialization_messages.append(embed_load_msg)
if gemini_init_msgs: initialization_messages.extend(gemini_init_msgs)
if not api_key_present: initialization_messages.append("⚠️ Gemini API Keyが見つかりません。Gemini関連機能は使用できません。")
elif gemini_model is None and not gemini_init_msgs: initialization_messages.append("⚠️ Geminiモデルの初期化に失敗しました。Gemini関連機能は使用できません。")

if initialization_messages:
    with st.expander("アプリケーション初期化ログ", expanded=False):
        for msg in initialization_messages:
            if msg.startswith("🔄"): st.write(msg)
            elif msg.startswith("✅"): st.success(msg)
            elif msg.startswith("⚠️"): st.warning(msg)
            else: st.write(msg)

# --- アプリケーションのメインロジック ---
if api_key_present and models_ready and files_present:
    st.subheader("ステップ1: チェア画像のアップロード")

    # 修正点 1: st.file_uploader に type 引数を指定
    allowed_upload_extensions = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"] # ドットなし
    uploaded_file = st.file_uploader(
        "チェア画像をアップロードしてください (JPG, PNG)",
        type=allowed_upload_extensions
    )
    upload_status_placeholder = st.empty()

    if uploaded_file is not None:

         # --- デバッグ情報表示 (Cloud環境での確認用) ---
        st.write(f"--- DEBUG INFO ---")
        st.write(f"Uploaded File Name: {uploaded_file.name}")
        st.write(f"Uploaded File Type (MIME): {uploaded_file.type}") # MIMEタイプ
        
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension_lower = file_extension.lower()

        st.write(f"os.path.splitext - file_name: {file_name}")
        st.write(f"os.path.splitext - file_extension: {file_extension}")
        st.write(f"file_extension_lower: {file_extension_lower}")
        st.write(f"--- END DEBUG INFO ---")
        # --- デバッグ情報表示ここまで ---



        # ファイル名から拡張子を取得し、小文字に変換
        file_name, file_extension = os.path.splitext(uploaded_file.name)
        file_extension_lower = file_extension.lower() # 例: ".jpg"

        # Python側でのチェック用拡張子リスト（小文字、ドット付き）
        allowed_extensions_check_python = ['.jpg', '.jpeg', '.png']

        if file_extension_lower not in allowed_extensions_check_python:
            st.error(f"エラー:許可されていないファイル形式です ({file_extension})。JPG, JPEG, PNGファイルをアップロードしてください。")
            # エラーの場合はセッション情報をクリア
            if st.session_state.uploaded_file_info is not None: # 既に何かアップロード情報があればクリア
                st.session_state.uploaded_file_info = None
                # 関連するセッション状態をリセット
                st.session_state.classification_result = None
                st.session_state.error_msg = None
                st.session_state.top_8_results = None
                st.session_state.search_figure = None
                st.session_state.selected_product_number = None
                st.session_state.search_logs = None
                st.session_state.filtered_filenames_list = None
                st.session_state.embedding_ranked_list = None
                # 必要であれば st.rerun() を呼び出してUIをクリーンにする
                # st.rerun()
        else:
             # 許可された拡張子の場合
             # 新しいファイルがアップロードされたか、または以前と異なるファイルがアップロードされた場合
            new_file_condition = (
                st.session_state.uploaded_file_info is None or
                st.session_state.uploaded_file_info['name'] != uploaded_file.name or
                st.session_state.uploaded_file_info['size'] != uploaded_file.size # サイズも比較に加える
            )
            if new_file_condition:
                st.session_state.uploaded_file_info = {
                    'name': uploaded_file.name,
                    'type': uploaded_file.type, # MIMEタイプ
                    'size': uploaded_file.size,
                    'data': uploaded_file.getvalue()
                }
                # 関連するセッション状態をリセット
                st.session_state.classification_result = None
                st.session_state.error_msg = None
                st.session_state.top_8_results = None
                st.session_state.search_figure = None
                st.session_state.selected_product_number = None
                st.session_state.search_logs = None
                st.session_state.filtered_filenames_list = None
                st.session_state.embedding_ranked_list = None
                # upload_status_placeholder.empty() # 必要に応じて以前の表示をクリア

    # 修正点 2: 冗長なセッション更新ブロックを削除
    # 以下のブロックは、上記の if/else 内で既に処理されているため不要です。
    # # 新しいファイルがアップロードされたら状態をリセット
    # if st.session_state.uploaded_file_info is None or st.session_state.uploaded_file_info['name'] != uploaded_file.name:
    #     st.session_state.uploaded_file_info = {
    #         'name': uploaded_file.name, 'type': uploaded_file.type,
    #         'size': uploaded_file.size, 'data': uploaded_file.getvalue()
    #     }
    #     # 関連するセッション状態をリセット
    #     st.session_state.classification_result = None
    #     # ... (以下略)

    if st.session_state.uploaded_file_info:
        with upload_status_placeholder.container():
            st.markdown("##### target image")
            st.image(st.session_state.uploaded_file_info['data'], caption=f"アップロード画像: {st.session_state.uploaded_file_info['name']}", width=200)

        # --- Step 2: Classification ---
        if st.session_state.classification_result is None and st.session_state.error_msg is None:
             with st.spinner("Geminiによる画像の自動分類を実行中..."):
                target_image_data = st.session_state.uploaded_file_info['data']
                classification_result, error_msg = classify_image_with_gemini(target_image_data, classification_items)
                st.session_state.classification_result = classification_result
                st.session_state.error_msg = error_msg

        if st.session_state.error_msg:
            st.error(f"❌ 分類エラー:\n{st.session_state.error_msg}")
        elif st.session_state.classification_result:
            with st.expander("ステップ2: 自動分類結果 詳細", expanded=False):
                st.success("✅ 分類結果:")
                class_df = pd.DataFrame(list(st.session_state.classification_result.items()), columns=['項目', '分類'])
                st.table(class_df)

            # --- Step 3: Search and Rank ---
            search_logs = []
            if st.session_state.classification_result and st.session_state.top_8_results is None and st.session_state.filtered_filenames_list is None:
                with st.spinner("🔍 類似画像を検索・ランキング中... (時間がかかる場合があります)"):
                    target_image_data = st.session_state.uploaded_file_info['data']
                    top_8, fig_top_3, search_logs, filtered_list, embed_list = perform_search_and_rank(
                        st.session_state.classification_result,
                        target_image_data
                    )
                    st.session_state.top_8_results = top_8
                    st.session_state.search_figure = fig_top_3
                    st.session_state.search_logs = search_logs
                    st.session_state.filtered_filenames_list = filtered_list
                    st.session_state.embedding_ranked_list = embed_list

            st.markdown("---")
            st.markdown("###### ⚙️ 検索プロセス詳細 (デバッグ用)")

            with st.expander(f"1. 分類フィルター後の候補リスト ({len(st.session_state.get('filtered_filenames_list', []))}件)", expanded=False):
                filtered_list = st.session_state.get('filtered_filenames_list')
                if filtered_list is not None:
                    if not filtered_list:
                        st.info("分類条件に一致するファイルは見つかりませんでした。")
                    else:
                        st.dataframe(pd.DataFrame(filtered_list, columns=["ファイル名"]), height=300, use_container_width=True)
                else:
                    st.info("フィルタリングはまだ実行されていません。")

            with st.expander(f"2. 埋め込み類似度 上位8件 ({len(st.session_state.get('embedding_ranked_list', []))}件)", expanded=False):
                embed_list = st.session_state.get('embedding_ranked_list')
                if embed_list is not None:
                    if not embed_list:
                        st.info("埋め込み類似度でランク付けされた候補はありません。")
                    else:
                        embed_df = pd.DataFrame(embed_list)
                        if 'score' in embed_df.columns:
                           embed_df['score'] = embed_df['score'].map('{:.3f}'.format)
                        st.dataframe(embed_df[['filename', 'score']], height=300, use_container_width=True)
                else:
                    st.info("埋め込み類似度計算はまだ実行されていないか、スキップされました。")

            gemini_list_for_count = st.session_state.get('top_8_results')
            gemini_count = len(gemini_list_for_count) if gemini_list_for_count is not None else 0
            with st.expander(f"3. Gemini類似度評価後 上位8件 ({gemini_count}件)", expanded=False):
                gemini_list = st.session_state.get('top_8_results')
                if gemini_list is not None:
                    if not gemini_list:
                        st.info("最終的な類似候補はありません。")
                    else:
                        gemini_df = pd.DataFrame(gemini_list)
                        if 'score' in gemini_df.columns:
                           gemini_df['score'] = gemini_df['score'].map('{:.3f}'.format)
                        if 'gemini_score' in gemini_df.columns:
                           gemini_df['gemini_score'] = gemini_df['gemini_score'].apply(lambda x: f'{x:.3f}' if isinstance(x, (int, float)) else ('評価失敗' if x is None else 'N/A'))
                        st.dataframe(gemini_df[['filename', 'gemini_score', 'score']].rename(columns={'score': 'embed_score'}), height=300, use_container_width=True)
                else:
                    st.info("Gemini類似度評価はまだ実行されていないか、スキップされました。")

            if 'search_logs' in st.session_state and st.session_state.search_logs:
                 with st.expander("ステップ3: 検索・ランキング処理ログ", expanded=False):
                     for log_line in st.session_state.search_logs:
                         if log_line.startswith("❌"): st.error(log_line)
                         elif log_line.startswith("⚠️"): st.warning(log_line)
                         elif log_line.startswith("ℹ️"): st.info(log_line)
                         elif log_line.startswith("✅"): st.success(log_line)
                         elif log_line.startswith("📊") or log_line.startswith("⏳") or log_line.startswith("🔄"): st.write(log_line)
                         else: st.text(log_line)

            if st.session_state.top_8_results:
                st.markdown("---")
                st.subheader("類似度 上位3件:")
                if st.session_state.search_figure:
                    st.pyplot(st.session_state.search_figure)
                else:
                     st.warning("上位3画像の表示中にエラーが発生したか、画像が見つかりませんでした。")

            if st.session_state.top_8_results and len(st.session_state.top_8_results) > 3:
                with st.expander("類似度 4位～8位を表示"):
                    results_4_to_8 = st.session_state.top_8_results[3:8]
                    if not results_4_to_8:
                        st.info("4位から8位の画像はありません。")
                    else:
                        cols = st.columns(2)
                        col_idx = 0
                        for rank_idx, result in enumerate(results_4_to_8):
                            rank = rank_idx + 4
                            with cols[col_idx % 2]:
                                try:
                                    st.image(result["path"], width=200)
                                except FileNotFoundError:
                                    st.error(f"画像ファイルが見つかりません: {result['filename']}")
                                except Exception as img_e:
                                    st.error(f"画像表示エラー ({result['filename']}): {img_e}")

                                embed_score_str = f"{result.get('score', 'N/A'):.3f}" if isinstance(result.get('score'), (int, float)) else "N/A"
                                gemini_score_val = result.get('gemini_score')
                                gemini_score_str = f"{gemini_score_val:.3f}" if isinstance(gemini_score_val, (int, float)) else "評価失敗" if gemini_score_val is None else "N/A"
                                st.markdown(f"""**Rank {rank}: {result['filename']}**\n- Gemini Sim: {gemini_score_str}\n- Embed Sim: {embed_score_str}""")
                                st.markdown("---")
                            col_idx += 1

            if st.session_state.top_8_results:
                product_numbers = []
                seen_numbers = set()
                for res in st.session_state.top_8_results:
                    if 'filename' in res and '.' in res['filename']:
                        p_num = res['filename'].split('.')[0]
                        if p_num not in seen_numbers:
                            product_numbers.append(p_num)
                            seen_numbers.add(p_num)

                if product_numbers:
                    st.markdown("---")
                    st.subheader("ステップ4: 詳細表示する品番を選択")

                    current_selection = st.session_state.get('selected_product_number')
                    current_index = 0
                    if current_selection in product_numbers:
                        try:
                            current_index = product_numbers.index(current_selection)
                        except ValueError:
                            current_index = 0
                    elif st.session_state.get('selected_product_number') is not None:
                        current_index = 0
                        st.session_state.selected_product_number = None

                    selected_product = st.selectbox(
                        "上位8件から品番を選択してください:",
                        options=product_numbers,
                        index=current_index,
                        key="product_select"
                    )

                    if selected_product:
                        if st.session_state.selected_product_number != selected_product:
                            st.session_state.selected_product_number = selected_product
                        display_product_info(st.session_state.selected_product_number)
else:
    st.warning("⚠️ アプリケーションを実行するための前提条件が満たされていません。上記のエラーメッセージを確認してください。")

st.markdown("---")
st.subheader("修理価格一覧")
if os.path.exists(PRICE_LIST_EXCEL_PATH):
    try:
        df_prices = pd.read_excel(PRICE_LIST_EXCEL_PATH)
        df_prices_display = df_prices.fillna("").astype(str)
        st.table(df_prices_display)
    except FileNotFoundError:
        st.error(f"❌ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")
    except Exception as e:
        st.error(f"❌ 修理価格一覧Excelファイルの読み込みエラー: {e}")
else:
    st.warning(f"⚠️ 修理価格一覧Excelファイルが見つかりません: {PRICE_LIST_EXCEL_PATH}")

st.markdown("---")
st.subheader("注意点")
if os.path.exists(NOTES_EXCEL_PATH):
    try:
        df_notes = pd.read_excel(NOTES_EXCEL_PATH)
        df_notes_display = df_notes.fillna("").astype(str)
        st.table(df_notes_display)
    except Exception as e:
        st.error(f"❌ 注意点Excelファイル ({NOTES_EXCEL_PATH}) の読み込みエラー: {e}")
else:
    st.warning(f"⚠️ 注意点Excelファイルが見つかりません: {NOTES_EXCEL_PATH}")