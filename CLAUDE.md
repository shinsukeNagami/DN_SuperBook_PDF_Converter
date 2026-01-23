# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

DN_SuperBook_PDF_Converterは、スキャンされた書籍PDFを高品質なデジタル書籍に変換するツール。AI画像鮮明化、傾き補正、ページオフセット調整、余白最適化、日本語OCR機能を提供する。

**オリジナル著者:** 登 大遊 (Daiyuu Nobori)
**フォーク:** clearclown/DN_SuperBook_PDF_Converter_Linux
**ライセンス:** AGPL v3.0

---

## 開発ロードマップ

### 現在: C#/.NET 6.0 版 (レガシー)
- 動作するが、メモリ使用量が大きい (10-30GB)
- OOM Killer問題あり
- Containerfileによるコンテナ化済み

### 次期: Rust版 (開発中)
- 完全リライトによる根本的改善
- メモリ使用量: 1-3GB目標
- シングルバイナリ配布
- Python AI連携 (RealESRGAN, YomiToku)

**詳細:** [GitHub Issue #19](https://github.com/clearclown/DN_SuperBook_PDF_Converter_Linux/issues/19)

---

## クイックスタート（C#版 - Linux + Podman + CUDA）

### 前提条件

- NVIDIA GPU（CUDA対応）
- Podman
- Task (https://taskfile.dev/)

### セットアップ

```bash
# 1. Taskのインストール
sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# 2. NVIDIA Container Toolkitのセットアップ（初回のみ）
task setup

# 3. 環境確認
task check

# 4. コンテナビルド
task build
```

### PDF変換

```bash
# 基本的な変換
task convert INPUT_DIR=./input OUTPUT_DIR=./output

# 日本語OCR付き変換
task convert-ocr INPUT_DIR=./input OUTPUT_DIR=./output

# インタラクティブシェル
task shell
```

---

## Rust版開発ガイド

### ディレクトリ構造 (計画)

```
.
├── superbook-pdf/              # Rust新規プロジェクト
│   ├── Cargo.toml
│   ├── src/
│   │   ├── main.rs
│   │   ├── cli/               # CLIインターフェース
│   │   ├── pdf/               # PDF処理
│   │   ├── image/             # 画像処理
│   │   ├── ai/                # Python AI連携
│   │   └── config/            # 設定管理
│   ├── ai_bridge/             # Python AIモジュール
│   │   ├── realesrgan_bridge.py
│   │   └── yomitoku_bridge.py
│   ├── tests/                 # 統合テスト
│   ├── benches/               # ベンチマーク
│   └── specs/                 # 仕様書 (TDD用)
│
├── SuperBookToolsApp/         # C#版 (レガシー)
├── SuperBookTools/
├── internal_libs/
├── Containerfile              # C#版コンテナ
└── Taskfile.yml
```

### TDD開発フロー

```
1. specs/*.md に仕様を記述
2. tests/ にテストを作成 (Red)
3. src/ に実装 (Green)
4. リファクタリング (Refactor)
5. ベンチマーク確認
```

### 仕様ファイル命名規則

```
specs/
├── 01-cli.spec.md           # CLI仕様
├── 02-pdf-reader.spec.md    # PDF読み込み仕様
├── 03-pdf-writer.spec.md    # PDF書き込み仕様
├── 04-image-extract.spec.md # 画像抽出仕様
├── 05-deskew.spec.md        # 傾き補正仕様
├── 06-margin.spec.md        # マージン処理仕様
├── 07-page-number.spec.md   # ページ番号認識仕様
├── 08-ai-bridge.spec.md     # AI連携仕様
└── 09-realesrgan.spec.md    # RealESRGAN仕様
```

---

## C#版アーキテクチャ (参考)

### エントリポイントと処理フロー

```
Startup.cs (Main)
  └→ SuperBookTools() - コンソールコマンドディスパッチャ
      └→ ConvertPdf - プライマリユーザーコマンド
          └→ SuperPdfUtil.PerformPdfAsync() - メイン処理パイプライン
```

### コアモジュール

| ディレクトリ | 役割 |
|-------------|------|
| `SuperBookToolsApp/` | コンソールアプリケーション、エントリポイント |
| `SuperBookTools/Basic/SuperPdfUtil.cs` | PDF処理パイプライン（約5,000行） |
| `SuperBookTools/Basic/PlatformToolResolver.cs` | Linux/Windows クロスプラットフォーム対応 |
| `internal_libs/IPA-DN-Cores/` | 基盤ライブラリ（Apache 2.0） |

### PDF処理パイプライン

1. PDF画像抽出（ImageMagick）
2. マージントリミング（0.5%）
3. AI画像鮮明化（RealESRGAN 2x）
4. 傾き検出・補正（OpenCV）
5. ページ番号OCR・認識（Tesseract + ヒューリスティック）
6. オフセット整列計算
7. 余白統一トリミング
8. PDFメタデータ同期
9. オプション: YomiToku日本語OCR

---

## 重要なアルゴリズム

### ページ番号検出 (Rust版で再実装予定)

- **マルチパスページ番号検出:** 全奇数/偶数ページを集合的に分析し、装飾的要素があってもページ番号を識別
- **ヒューリスティックオフセット補正:** ページ番号位置をアンカーポイントとしてスキャンオフセットを計算・補正
- **統一マージン検出:** 全ページで一貫した最適マージンを算出
- **縦書き/横書き検出:** 縦書きテキストを自動検出し、PDFビューア用の適切なフラグを設定

---

## 既知の問題とRust版での解決策

| 問題 | C#版 | Rust版解決策 |
|------|------|--------------|
| メモリ使用量 | 10-30GB | 1-3GB (ストリーミング処理) |
| OOM Killer | 頻発 | 所有権システムで防止 |
| RealESRGAN失敗 | 無限リトライ | 部分失敗許容設計 |
| 起動時間 | 2-3秒 | <100ms |
| 配布 | コンテナ必須 | シングルバイナリ |

**関連Issue:**
- #14 RealESRGAN無限リトライループ
- #15 OOM Killer問題
- #19 Rust完全リライト計画

---

## 開発コマンド

### C#版 (レガシー)

```bash
# Taskfile経由
task build              # コンテナビルド
task convert            # 変換実行
task shell              # インタラクティブシェル

# ローカルビルド
dotnet build SuperBookToolsApp/SuperBookToolsApp.csproj -c Release
```

### Rust版 (新規)

```bash
cd superbook-pdf

# ビルド
cargo build --release

# テスト (TDD)
cargo test

# 特定のテストのみ
cargo test test_pdf_reader

# ベンチマーク
cargo bench

# ドキュメント生成
cargo doc --open

# フォーマット
cargo fmt

# Lint
cargo clippy
```

---

## ハードウェア要件

- **RAM:** 8GB以上推奨（Rust版は4GBでも動作予定）
- **GPU:** NVIDIA CUDA対応GPU（4GB+ VRAM）
- RealESRGANとYomiTokuはGPU処理を前提

---

## 参考リンク

- [オリジナルリポジトリ](https://github.com/dnobori/DN_SuperBook_PDF_Converter)
- [Rust版ロードマップ (Issue #19)](https://github.com/clearclown/DN_SuperBook_PDF_Converter_Linux/issues/19)
- [RealESRGAN](https://github.com/xinntao/Real-ESRGAN)
- [YomiToku](https://github.com/kotaro-kinoshita/yomitoku)
