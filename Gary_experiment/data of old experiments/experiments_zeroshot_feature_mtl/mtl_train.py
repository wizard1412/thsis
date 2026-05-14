"""
FTDualNet 三階段訓練流程
基於 PDualNet (Scientific Reports, 2025) 的三階段訓練策略

第一階段：自編碼器預訓練（學習特徵的緊湊表示）
第二階段：雙 Head 訓練（凍結編碼器，訓練評分 + 診斷 Head）
第三階段：聯合微調（解凍全部，低學習率端到端優化）
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime

import torch.nn.functional as F

from mtl_config import (
    OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR,
    BATCH_SIZE, NUM_FOLDS, RANDOM_SEED, NUM_FEATURES,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_WEIGHT_DECAY, PHASE1_PATIENCE,
    PHASE2_EPOCHS, PHASE2_LR, PHASE2_WEIGHT_DECAY, PHASE2_PATIENCE,
    PHASE2_DIAG_LOSS_WEIGHT,
    PHASE3_EPOCHS, PHASE3_ENCODER_LR, PHASE3_HEAD_LR,
    PHASE3_WEIGHT_DECAY, PHASE3_PATIENCE,
)
from mtl_model import FTDualNet, FeatureAutoencoder, UncertaintyWeightedLoss
from mtl_data import (
    prepare_all_data, FingerTappingDataset, AutoencoderDataset,
)


# ===========================
# 日誌設定
# ===========================

def setup_logging():
    """設定日誌系統"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"mtl_train_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"日誌檔案: {log_file}")
    return log_file


# ===========================
# 設定隨機種子
# ===========================

def set_seed(seed=None):
    if seed is None:
        seed = RANDOM_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# 第一階段：自編碼器預訓練
# ===========================

def train_phase1_autoencoder(ae_features, device):
    """
    第一階段：訓練自編碼器學習特徵的緊湊表示

    參數：
        ae_features: np.ndarray, (N, num_features) 所有唯一樣本的特徵
        device: torch device

    回傳：
        encoder_state_dict: 訓練好的編碼器權重
        train_losses: list, 每個 epoch 的訓練損失
    """
    logging.info("=" * 60)
    logging.info("第一階段：自編碼器預訓練")
    logging.info("=" * 60)

    # 資料準備
    dataset = AutoencoderDataset(ae_features)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    autoencoder = FeatureAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=PHASE1_LR, weight_decay=PHASE1_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    # 訓練
    best_loss = float("inf")
    patience_counter = 0
    train_losses = []

    for epoch in range(PHASE1_EPOCHS):
        autoencoder.train()
        epoch_loss = 0.0

        for batch in dataloader:
            batch = batch.to(device)
            reconstructed, _ = autoencoder(batch)
            loss = criterion(reconstructed, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch)

        epoch_loss /= len(dataset)
        train_losses.append(epoch_loss)

        # 早停
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in autoencoder.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0 or patience_counter == 0:
            logging.info(
                f"  Epoch {epoch+1}/{PHASE1_EPOCHS} - "
                f"Loss: {epoch_loss:.6f} - Best: {best_loss:.6f} - "
                f"Patience: {patience_counter}/{PHASE1_PATIENCE}"
            )

        if patience_counter >= PHASE1_PATIENCE:
            logging.info(f"  早停於 Epoch {epoch+1}")
            break

    # 載入最佳權重，提取編碼器
    autoencoder.load_state_dict(best_state)
    encoder_state = {
        k: v for k, v in best_state.items() if k.startswith("encoder.")
    }

    logging.info(f"  最佳重建損失: {best_loss:.6f}")
    return encoder_state, train_losses


# ===========================
# 第二階段：雙 Head 訓練
# ===========================

def train_phase2_heads(model, train_loader, val_loader, device):
    """
    第二階段：凍結編碼器，訓練 Attention + 雙 Head

    參數：
        model: FTDualNet（已載入 Phase 1 的編碼器權重）
        train_loader: 訓練資料 DataLoader
        val_loader: 驗證資料 DataLoader
        device: torch device

    回傳：
        best_state: 最佳模型權重
        history: dict, 訓練/驗證損失歷史
    """
    logging.info("=" * 60)
    logging.info("第二階段：雙 Head 訓練（編碼器凍結）")
    logging.info("=" * 60)

    model.freeze_encoder()

    # 只優化未凍結的參數
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=PHASE2_LR, weight_decay=PHASE2_WEIGHT_DECAY
    )

    score_criterion = nn.CrossEntropyLoss()
    diag_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(PHASE2_EPOCHS):
        # --- 訓練 ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            scores = batch["score"].to(device)
            diagnosed = batch["diagnosed"].to(device)

            outputs = model(features)

            # 評分損失（分類）
            loss_score = score_criterion(outputs["score_logits"], scores)

            # 診斷損失（僅計算有標籤的樣本，diagnosed != -1）
            valid_diag_mask = diagnosed >= 0
            if valid_diag_mask.any():
                loss_diag = diag_criterion(
                    outputs["diag_prob"][valid_diag_mask].squeeze(),
                    diagnosed[valid_diag_mask],
                )
            else:
                loss_diag = torch.tensor(0.0, device=device)

            loss = loss_score + PHASE2_DIAG_LOSS_WEIGHT * loss_diag

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(features)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # --- 驗證 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                scores = batch["score"].to(device)
                diagnosed = batch["diagnosed"].to(device)

                outputs = model(features)
                loss_score = score_criterion(outputs["score_logits"], scores)

                valid_diag_mask = diagnosed >= 0
                if valid_diag_mask.any():
                    loss_diag = diag_criterion(
                        outputs["diag_prob"][valid_diag_mask].squeeze(),
                        diagnosed[valid_diag_mask],
                    )
                else:
                    loss_diag = torch.tensor(0.0, device=device)

                val_loss += (loss_score + PHASE2_DIAG_LOSS_WEIGHT * loss_diag).item() * len(features)

                # 計算準確率
                preds = outputs["score_logits"].argmax(dim=-1)
                val_correct += (preds == scores).sum().item()
                val_total += len(scores)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logging.info(
                f"  Epoch {epoch+1}/{PHASE2_EPOCHS} - "
                f"Train: {train_loss:.4f} - Val: {val_loss:.4f} - "
                f"Acc: {val_acc:.3f} - Patience: {patience_counter}/{PHASE2_PATIENCE}"
            )

        if patience_counter >= PHASE2_PATIENCE:
            logging.info(f"  早停於 Epoch {epoch+1}")
            break

    logging.info(f"  最佳驗證損失: {best_val_loss:.4f}")
    return best_state, history


# ===========================
# 第三階段：聯合微調
# ===========================

def train_phase3_finetune(model, train_loader, val_loader, device):
    """
    第三階段：解凍全部，使用差異化學習率 + 不確定性損失權重做端到端微調

    參數：
        model: FTDualNet（已載入 Phase 2 的最佳權重）
        train_loader: 訓練資料 DataLoader
        val_loader: 驗證資料 DataLoader
        device: torch device

    回傳：
        best_state: 最佳模型權重
        history: dict, 訓練/驗證損失歷史
    """
    logging.info("=" * 60)
    logging.info("第三階段：聯合微調（全部解凍）")
    logging.info("=" * 60)

    model.unfreeze_encoder()

    # 差異化學習率
    param_groups = model.get_parameter_groups(PHASE3_ENCODER_LR, PHASE3_HEAD_LR)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=PHASE3_WEIGHT_DECAY)

    # 不確定性加權損失
    uncertainty_loss = UncertaintyWeightedLoss(num_tasks=2).to(device)
    # 將 uncertainty_loss 的參數也加入 optimizer
    optimizer.add_param_group({"params": uncertainty_loss.parameters(), "lr": PHASE3_HEAD_LR})

    score_criterion = nn.CrossEntropyLoss()
    diag_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(PHASE3_EPOCHS):
        # --- 訓練 ---
        model.train()
        uncertainty_loss.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            scores = batch["score"].to(device)
            diagnosed = batch["diagnosed"].to(device)

            outputs = model(features)
            loss_score = score_criterion(outputs["score_logits"], scores)

            valid_diag_mask = diagnosed >= 0
            if valid_diag_mask.any():
                loss_diag = diag_criterion(
                    outputs["diag_prob"][valid_diag_mask].squeeze(),
                    diagnosed[valid_diag_mask],
                )
            else:
                loss_diag = torch.tensor(0.0, device=device)

            # 使用不確定性加權
            loss = uncertainty_loss([loss_score, loss_diag])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(features)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # --- 驗證 ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                scores = batch["score"].to(device)

                outputs = model(features)
                loss_score = score_criterion(outputs["score_logits"], scores)
                val_loss += loss_score.item() * len(features)

                # 計算準確率
                preds = outputs["score_logits"].argmax(dim=-1)
                val_correct += (preds == scores).sum().item()
                val_total += len(scores)

        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / val_total if val_total > 0 else 0
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            log_vars = uncertainty_loss.log_vars.data.tolist()
            logging.info(
                f"  Epoch {epoch+1}/{PHASE3_EPOCHS} - "
                f"Train: {train_loss:.4f} - Val: {val_loss:.4f} - "
                f"Acc: {val_acc:.3f} - "
                f"LogVar: [{log_vars[0]:.3f}, {log_vars[1]:.3f}] - "
                f"Patience: {patience_counter}/{PHASE3_PATIENCE}"
            )

        if patience_counter >= PHASE3_PATIENCE:
            logging.info(f"  早停於 Epoch {epoch+1}")
            break

    logging.info(f"  最佳驗證損失: {best_val_loss:.4f}")
    return best_state, history


# ===========================
# 完整交叉驗證訓練
# ===========================

def run_cross_validation():
    """
    執行完整的 5-Fold 交叉驗證訓練

    每個 Fold 依序執行三階段訓練，記錄每個 Fold 的結果。
    """
    set_seed()
    log_file = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用裝置: {device}")

    # 建立輸出目錄
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 準備資料
    data = prepare_all_data()

    # 第一階段：自編碼器預訓練（所有資料，只需做一次）
    encoder_state, ae_losses = train_phase1_autoencoder(data["ae_features"], device)

    # 儲存自編碼器的訓練損失
    torch.save(encoder_state, CHECKPOINT_DIR / "phase1_encoder.pt")
    logging.info(f"編碼器權重已儲存至 {CHECKPOINT_DIR / 'phase1_encoder.pt'}")

    # 交叉驗證
    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(data["cv_splits"]):
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Fold {fold_idx + 1}/{NUM_FOLDS}")
        logging.info(f"{'#' * 60}")

        # 準備 Fold 資料
        train_dataset = FingerTappingDataset(
            features=data["train_features"][train_idx],
            scores=data["train_scores"][train_idx],
            diagnosed=data["train_diagnosed"][train_idx],
        )
        val_dataset = FingerTappingDataset(
            features=data["train_features"][val_idx],
            scores=data["train_scores"][val_idx],
            diagnosed=data["train_diagnosed"][val_idx],
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        logging.info(f"  訓練集: {len(train_dataset)} 筆")
        logging.info(f"  驗證集: {len(val_dataset)} 筆")

        # 建立模型，載入 Phase 1 的編碼器權重
        model = FTDualNet().to(device)
        ae_state = model.autoencoder.state_dict()
        for k, v in encoder_state.items():
            if k in ae_state:
                ae_state[k] = v
        model.autoencoder.load_state_dict(ae_state)

        # 第二階段
        phase2_state, phase2_history = train_phase2_heads(
            model, train_loader, val_loader, device
        )
        model.load_state_dict(phase2_state)

        # 第三階段
        phase3_state, phase3_history = train_phase3_finetune(
            model, train_loader, val_loader, device
        )
        model.load_state_dict(phase3_state)

        # 最終驗證評估
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                scores = batch["score"]

                outputs = model(features)
                probs = F.softmax(outputs["score_logits"], dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(scores.tolist())
                all_probs.extend(probs.cpu().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 計算指標
        accuracy = np.mean(all_preds == all_labels)
        mae = np.mean(np.abs(all_preds - all_labels))
        off_by_one = np.mean(np.abs(all_preds - all_labels) <= 1)

        fold_result = {
            "fold": fold_idx + 1,
            "accuracy": accuracy,
            "mae": mae,
            "off_by_one_accuracy": off_by_one,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
            "probabilities": all_probs,
            "phase2_history": phase2_history,
            "phase3_history": phase3_history,
        }
        all_fold_results.append(fold_result)

        logging.info(f"\n  Fold {fold_idx + 1} 結果:")
        logging.info(f"    準確率: {accuracy:.3f}")
        logging.info(f"    MAE: {mae:.3f}")
        logging.info(f"    ±1 準確率: {off_by_one:.3f}")

        # 儲存此 Fold 的模型
        torch.save(
            phase3_state,
            CHECKPOINT_DIR / f"ftdualnet_fold{fold_idx + 1}.pt",
        )

    # 彙總結果
    logging.info(f"\n{'=' * 60}")
    logging.info("交叉驗證結果彙總")
    logging.info(f"{'=' * 60}")

    accs = [r["accuracy"] for r in all_fold_results]
    maes = [r["mae"] for r in all_fold_results]
    obo_accs = [r["off_by_one_accuracy"] for r in all_fold_results]

    logging.info(f"  準確率:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    logging.info(f"  MAE:       {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    logging.info(f"  ±1 準確率: {np.mean(obo_accs):.3f} ± {np.std(obo_accs):.3f}")

    for i, r in enumerate(all_fold_results):
        logging.info(
            f"  Fold {i+1}: Acc={r['accuracy']:.3f}, MAE={r['mae']:.3f}, ±1={r['off_by_one_accuracy']:.3f}"
        )

    # 儲存結果
    results_summary = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_off_by_one": float(np.mean(obo_accs)),
        "std_off_by_one": float(np.std(obo_accs)),
        "fold_results": [
            {
                "fold": r["fold"],
                "accuracy": r["accuracy"],
                "mae": r["mae"],
                "off_by_one_accuracy": r["off_by_one_accuracy"],
                "predictions": r["predictions"],
                "labels": r["labels"],
            }
            for r in all_fold_results
        ],
    }

    results_file = OUTPUT_DIR / "cv_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\n結果已儲存至 {results_file}")

    return all_fold_results


# ===========================
# 訓練最終模型（使用全部資料）
# ===========================

def train_final_model():
    """
    使用全部資料訓練最終模型（不做交叉驗證，用於部署）
    """
    set_seed()
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用裝置: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # 準備資料
    data = prepare_all_data()

    # 第一階段
    encoder_state, _ = train_phase1_autoencoder(data["ae_features"], device)

    # 使用全部資料作為訓練集（不分割）
    full_dataset = FingerTappingDataset(
        features=data["train_features"],
        scores=data["train_scores"],
        diagnosed=data["train_diagnosed"],
    )
    full_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 建立模型
    model = FTDualNet().to(device)
    ae_state = model.autoencoder.state_dict()
    for k, v in encoder_state.items():
        if k in ae_state:
            ae_state[k] = v
    model.autoencoder.load_state_dict(ae_state)

    # 第二階段（用 full_loader 同時當訓練和驗證）
    phase2_state, _ = train_phase2_heads(model, full_loader, full_loader, device)
    model.load_state_dict(phase2_state)

    # 第三階段
    phase3_state, _ = train_phase3_finetune(model, full_loader, full_loader, device)
    model.load_state_dict(phase3_state)

    # 儲存最終模型
    final_path = CHECKPOINT_DIR / "ftdualnet_final.pt"
    torch.save(phase3_state, final_path)
    logging.info(f"最終模型已儲存至 {final_path}")

    return model


# ===========================
# 主程式
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTDualNet 訓練")
    parser.add_argument(
        "--mode",
        choices=["cv", "final"],
        default="cv",
        help="訓練模式: cv=交叉驗證, final=訓練最終模型",
    )
    args = parser.parse_args()

    if args.mode == "cv":
        print("執行 5-Fold 交叉驗證訓練...")
        run_cross_validation()
    elif args.mode == "final":
        print("訓練最終模型（使用全部資料）...")
        train_final_model()
