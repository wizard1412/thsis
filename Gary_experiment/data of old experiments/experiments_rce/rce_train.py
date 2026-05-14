"""
FTDualNet + RCE 三階段訓練流程

與原版 mtl_train.py 的關鍵差異：
- 第二、三階段使用 RCE loss 取代 CrossEntropyLoss
- 每個 sample 不展開 rater，而是對每位 rater 分別計算 loss
- 新增 RCE confusion matrix 的訓練和視覺化
- 驗證時同時報告多種 ground truth 策略的結果
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

from rce_config import (
    OUTPUT_DIR, CHECKPOINT_DIR, PLOT_DIR, LOG_DIR,
    BATCH_SIZE, NUM_FOLDS, RANDOM_SEED, NUM_FEATURES, SCORE_NUM_CLASSES,
    MAX_NUM_RATERS,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_WEIGHT_DECAY, PHASE1_PATIENCE,
    PHASE2_EPOCHS, PHASE2_LR, PHASE2_RCE_LR, PHASE2_WEIGHT_DECAY,
    PHASE2_PATIENCE, PHASE2_DIAG_LOSS_WEIGHT,
    PHASE3_EPOCHS, PHASE3_ENCODER_LR, PHASE3_HEAD_LR, PHASE3_RCE_LR,
    PHASE3_WEIGHT_DECAY, PHASE3_PATIENCE,
    RCE_TRACE_LAMBDA,
)
from rce_model import (
    FTDualNet, FeatureAutoencoder, RaterConfusionEstimation,
    OrdinalFocalLoss, UncertaintyWeightedLoss,
)
from rce_data import prepare_all_data, RCEDataset, AutoencoderDataset


# ===========================
# 日誌設定
# ===========================

def setup_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"rce_train_{timestamp}.log"

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


def set_seed(seed=None):
    if seed is None:
        seed = RANDOM_SEED
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===========================
# 第一階段：自編碼器預訓練（與原版相同）
# ===========================

def train_phase1_autoencoder(ae_features, device):
    """第一階段：訓練自編碼器（與原版完全相同）"""
    logging.info("=" * 60)
    logging.info("第一階段：自編碼器預訓練")
    logging.info("=" * 60)

    dataset = AutoencoderDataset(ae_features)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    autoencoder = FeatureAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        autoencoder.parameters(), lr=PHASE1_LR, weight_decay=PHASE1_WEIGHT_DECAY
    )
    criterion = nn.MSELoss()

    best_loss = float("inf")
    patience_counter = 0
    best_state = None

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

    autoencoder.load_state_dict(best_state)
    encoder_state = {
        k: v for k, v in best_state.items() if k.startswith("encoder.")
    }

    logging.info(f"  最佳重建損失: {best_loss:.6f}")
    return encoder_state


# ===========================
# RCE Loss 計算
# ===========================

def compute_rce_loss(pred_prob, ratings, rce_module, of_criterion, device):
    """
    計算 RCE loss：對每位 rater 分別計算 ordinal focal loss

    pred_prob: (batch, C) — 模型 softmax 輸出
    ratings:   (batch, R) — 所有 rater 的分數，-1 = 缺失
    rce_module: RaterConfusionEstimation
    of_criterion: OrdinalFocalLoss

    回傳：
        rce_loss: scalar — 平均 RCE loss
        num_valid: int — 有效 rater 計數
    """
    total_loss = torch.tensor(0.0, device=device)
    num_valid = 0

    for r in range(ratings.shape[1]):
        rater_scores = ratings[:, r]
        valid_mask = rater_scores >= 0

        if valid_mask.sum() == 0:
            continue

        # 取有效樣本
        valid_pred = pred_prob[valid_mask]
        valid_scores = rater_scores[valid_mask]

        # 透過 rater confusion matrix 轉換預測
        transformed = rce_module.transform(valid_pred, rater_idx=r)

        # Ordinal focal loss
        loss_r = of_criterion(transformed, valid_scores)
        total_loss = total_loss + loss_r
        num_valid += 1

    if num_valid > 0:
        total_loss = total_loss / num_valid

    return total_loss, num_valid


# ===========================
# 第二階段：RCE + Dual Head 訓練
# ===========================

def train_phase2_rce(model, rce_module, train_loader, val_loader, device):
    """
    第二階段：凍結編碼器，訓練 Attention + 雙 Head + RCE
    """
    logging.info("=" * 60)
    logging.info("第二階段：RCE + Dual Head 訓練（編碼器凍結）")
    logging.info("=" * 60)

    model.freeze_encoder()

    # 分組學習率：模型參數 vs RCE 參數
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": model_params, "lr": PHASE2_LR},
        {"params": rce_module.parameters(), "lr": PHASE2_RCE_LR},
    ], weight_decay=PHASE2_WEIGHT_DECAY)

    of_criterion = OrdinalFocalLoss()
    diag_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_kappa": []}
    best_state = None
    best_rce_state = None

    for epoch in range(PHASE2_EPOCHS):
        # --- 訓練 ---
        model.train()
        rce_module.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            ratings = batch["ratings"].to(device)
            diagnosed = batch.get("diagnosed")

            outputs = model(features)
            pred_prob = F.softmax(outputs["score_logits"], dim=-1)

            # RCE loss
            rce_loss, _ = compute_rce_loss(
                pred_prob, ratings, rce_module, of_criterion, device
            )

            # Trace regularization
            trace_loss = rce_module.trace_penalty()

            # 診斷 loss
            if diagnosed is not None:
                diagnosed = diagnosed.to(device)
                valid_diag_mask = diagnosed >= 0
                if valid_diag_mask.any():
                    loss_diag = diag_criterion(
                        outputs["diag_prob"][valid_diag_mask].squeeze(),
                        diagnosed[valid_diag_mask],
                    )
                else:
                    loss_diag = torch.tensor(0.0, device=device)
            else:
                loss_diag = torch.tensor(0.0, device=device)

            # 總 loss
            loss = rce_loss + RCE_TRACE_LAMBDA * trace_loss + PHASE2_DIAG_LOSS_WEIGHT * loss_diag

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(features)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # --- 驗證 ---
        val_loss, val_acc, val_kappa = evaluate_model(
            model, rce_module, val_loader, of_criterion, diag_criterion, device
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_rce_state = {k: v.clone() for k, v in rce_module.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or patience_counter == 0:
            logging.info(
                f"  Epoch {epoch+1}/{PHASE2_EPOCHS} - "
                f"Train: {train_loss:.4f} - Val: {val_loss:.4f} - "
                f"Acc: {val_acc:.3f} - κ: {val_kappa:.3f} - "
                f"Patience: {patience_counter}/{PHASE2_PATIENCE}"
            )

        if patience_counter >= PHASE2_PATIENCE:
            logging.info(f"  早停於 Epoch {epoch+1}")
            break

    logging.info(f"  最佳驗證損失: {best_val_loss:.4f}")
    return best_state, best_rce_state, history


# ===========================
# 第三階段：聯合微調
# ===========================

def train_phase3_finetune(model, rce_module, train_loader, val_loader, device):
    """第三階段：解凍全部，差異化學習率 + 不確定性加權"""
    logging.info("=" * 60)
    logging.info("第三階段：聯合微調（全部解凍）")
    logging.info("=" * 60)

    model.unfreeze_encoder()

    param_groups = model.get_parameter_groups(PHASE3_ENCODER_LR, PHASE3_HEAD_LR)
    param_groups.append({"params": rce_module.parameters(), "lr": PHASE3_RCE_LR})

    uncertainty_loss = UncertaintyWeightedLoss(num_tasks=2).to(device)
    param_groups.append({"params": uncertainty_loss.parameters(), "lr": PHASE3_HEAD_LR})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=PHASE3_WEIGHT_DECAY)

    of_criterion = OrdinalFocalLoss()
    diag_criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_kappa": []}
    best_state = None
    best_rce_state = None

    for epoch in range(PHASE3_EPOCHS):
        # --- 訓練 ---
        model.train()
        rce_module.train()
        uncertainty_loss.train()
        train_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(device)
            ratings = batch["ratings"].to(device)
            diagnosed = batch.get("diagnosed")

            outputs = model(features)
            pred_prob = F.softmax(outputs["score_logits"], dim=-1)

            # RCE loss
            rce_loss, _ = compute_rce_loss(
                pred_prob, ratings, rce_module, of_criterion, device
            )
            rce_loss = rce_loss + RCE_TRACE_LAMBDA * rce_module.trace_penalty()

            # 診斷 loss
            if diagnosed is not None:
                diagnosed = diagnosed.to(device)
                valid_diag_mask = diagnosed >= 0
                if valid_diag_mask.any():
                    loss_diag = diag_criterion(
                        outputs["diag_prob"][valid_diag_mask].squeeze(),
                        diagnosed[valid_diag_mask],
                    )
                else:
                    loss_diag = torch.tensor(0.0, device=device)
            else:
                loss_diag = torch.tensor(0.0, device=device)

            # 不確定性加權
            loss = uncertainty_loss([rce_loss, loss_diag])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(features)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        # --- 驗證 ---
        val_loss, val_acc, val_kappa = evaluate_model(
            model, rce_module, val_loader, of_criterion, diag_criterion, device
        )
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_kappa"].append(val_kappa)

        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_rce_state = {k: v.clone() for k, v in rce_module.state_dict().items()}
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter == 0:
            log_vars = uncertainty_loss.log_vars.data.tolist()
            logging.info(
                f"  Epoch {epoch+1}/{PHASE3_EPOCHS} - "
                f"Train: {train_loss:.4f} - Val: {val_loss:.4f} - "
                f"Acc: {val_acc:.3f} - κ: {val_kappa:.3f} - "
                f"LogVar: [{log_vars[0]:.3f}, {log_vars[1]:.3f}] - "
                f"Patience: {patience_counter}/{PHASE3_PATIENCE}"
            )

        if patience_counter >= PHASE3_PATIENCE:
            logging.info(f"  早停於 Epoch {epoch+1}")
            break

    logging.info(f"  最佳驗證損失: {best_val_loss:.4f}")
    return best_state, best_rce_state, history


# ===========================
# 驗證評估
# ===========================

def evaluate_model(model, rce_module, val_loader, of_criterion, diag_criterion, device):
    """驗證集評估（使用多數決分數作為 ground truth）"""
    model.eval()
    rce_module.eval()

    val_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            ratings = batch["ratings"].to(device)
            majority_score = batch["majority_score"]

            outputs = model(features)
            pred_prob = F.softmax(outputs["score_logits"], dim=-1)

            # RCE loss for validation
            rce_loss, _ = compute_rce_loss(
                pred_prob, ratings, rce_module, of_criterion, device
            )
            val_loss += rce_loss.item() * len(features)

            # 預測（直接用模型 softmax 輸出，不經 RCE 轉換）
            preds = pred_prob.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(majority_score.tolist())

    val_loss /= len(val_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels)
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")

    return val_loss, accuracy, kappa


# ===========================
# Consensus-Level 評估
# ===========================

def evaluate_consensus_levels(model, val_features, val_ratings, val_majority, device):
    """
    分不同共識等級評估模型表現

    回傳 dict 包含：
    - full_consensus: 所有 rater 完全一致的樣本
    - majority_consensus: 多數 rater 同意的樣本
    - low_consensus: rater 高度分歧的樣本
    - overall: 所有樣本
    """
    model.eval()
    results = {}

    with torch.no_grad():
        features_tensor = torch.FloatTensor(val_features).to(device)
        outputs = model(features_tensor)
        pred_prob = F.softmax(outputs["score_logits"], dim=-1)
        preds = pred_prob.argmax(dim=-1).cpu().numpy()

    # 分類共識等級
    consensus_full = []   # 所有 rater 完全一致
    consensus_major = []  # 多數一致（但非全部）
    consensus_low = []    # 分散

    for i in range(len(val_ratings)):
        valid = val_ratings[i][val_ratings[i] >= 0]
        if len(valid) < 2:
            consensus_major.append(i)
            continue

        unique = len(set(valid))
        max_count = max(np.bincount(valid.astype(int), minlength=SCORE_NUM_CLASSES))
        agreement_ratio = max_count / len(valid)

        if unique == 1:
            consensus_full.append(i)
        elif agreement_ratio >= 0.6:
            consensus_major.append(i)
        else:
            consensus_low.append(i)

    # 計算各等級指標
    for name, indices in [
        ("full_consensus", consensus_full),
        ("majority_consensus", consensus_major),
        ("low_consensus", consensus_low),
        ("overall", list(range(len(preds)))),
    ]:
        if len(indices) == 0:
            results[name] = {"n": 0, "accuracy": 0, "mae": 0, "kappa": 0}
            continue

        idx = np.array(indices)
        p = preds[idx]
        m = val_majority[idx]

        acc = np.mean(p == m)
        mae = np.mean(np.abs(p - m))
        obo = np.mean(np.abs(p - m) <= 1)
        kappa = cohen_kappa_score(m, p, weights="quadratic") if len(set(m)) > 1 else 0.0

        results[name] = {
            "n": len(indices),
            "accuracy": float(acc),
            "mae": float(mae),
            "off_by_one": float(obo),
            "kappa": float(kappa),
        }

    return results


# ===========================
# Confusion Matrix 視覺化
# ===========================

def log_rce_matrices(rce_module):
    """記錄學習到的 rater confusion matrices"""
    logging.info("\n學習到的 Rater Confusion Matrices:")
    for r in range(rce_module.num_raters):
        A = rce_module.get_matrix(r).detach().cpu().numpy()
        diag = np.diag(A)
        off_diag_max = np.max(A - np.diag(diag))
        logging.info(
            f"  Rater {r+1}: diag_mean={diag.mean():.3f}, "
            f"off_diag_max={off_diag_max:.3f}, trace={diag.sum():.3f}"
        )
        for row in range(A.shape[0]):
            logging.info(f"    [{', '.join(f'{v:.3f}' for v in A[row])}]")


# ===========================
# 完整交叉驗證訓練
# ===========================

def run_cross_validation():
    """執行 5-Fold 交叉驗證"""
    set_seed()
    log_file = setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"使用裝置: {device}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 準備資料
    data = prepare_all_data()

    # 第一階段：自編碼器預訓練
    encoder_state = train_phase1_autoencoder(data["ae_features"], device)
    torch.save(encoder_state, CHECKPOINT_DIR / "phase1_encoder.pt")

    # 交叉驗證
    all_fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(data["cv_splits"]):
        logging.info(f"\n{'#' * 60}")
        logging.info(f"Fold {fold_idx + 1}/{NUM_FOLDS}")
        logging.info(f"{'#' * 60}")

        # 準備 Fold 資料
        train_dataset = RCEDataset(
            features=data["all_features"][train_idx],
            ratings=data["all_ratings"][train_idx],
            diagnosed=data["all_diagnosed"][train_idx],
        )
        val_dataset = RCEDataset(
            features=data["all_features"][val_idx],
            ratings=data["all_ratings"][val_idx],
            diagnosed=data["all_diagnosed"][val_idx],
        )

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        logging.info(f"  訓練集: {len(train_dataset)} 筆 (不展開)")
        logging.info(f"  驗證集: {len(val_dataset)} 筆 (不展開)")

        # 建立模型 + RCE
        model = FTDualNet().to(device)
        rce_module = RaterConfusionEstimation().to(device)

        # 載入 Phase 1 編碼器
        ae_state = model.autoencoder.state_dict()
        for k, v in encoder_state.items():
            if k in ae_state:
                ae_state[k] = v
        model.autoencoder.load_state_dict(ae_state)

        # 第二階段
        phase2_state, phase2_rce_state, phase2_history = train_phase2_rce(
            model, rce_module, train_loader, val_loader, device
        )
        model.load_state_dict(phase2_state)
        rce_module.load_state_dict(phase2_rce_state)

        # 第三階段
        phase3_state, phase3_rce_state, phase3_history = train_phase3_finetune(
            model, rce_module, train_loader, val_loader, device
        )
        model.load_state_dict(phase3_state)
        rce_module.load_state_dict(phase3_rce_state)

        # 記錄 RCE matrices
        log_rce_matrices(rce_module)

        # 最終驗證評估
        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                majority_score = batch["majority_score"]

                outputs = model(features)
                probs = F.softmax(outputs["score_logits"], dim=-1)
                preds = probs.argmax(dim=-1)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(majority_score.tolist())
                all_probs.extend(probs.cpu().tolist())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 基本指標
        accuracy = np.mean(all_preds == all_labels)
        mae = np.mean(np.abs(all_preds - all_labels))
        off_by_one = np.mean(np.abs(all_preds - all_labels) <= 1)
        kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
        f1 = f1_score(all_labels, all_preds, average="weighted")

        # Consensus-level 評估
        consensus_results = evaluate_consensus_levels(
            model,
            data["all_features"][val_idx],
            data["all_ratings"][val_idx],
            data["majority_scores"][val_idx],
            device,
        )

        fold_result = {
            "fold": fold_idx + 1,
            "accuracy": float(accuracy),
            "mae": float(mae),
            "off_by_one_accuracy": float(off_by_one),
            "qwk": float(kappa),
            "f1_weighted": float(f1),
            "consensus_results": consensus_results,
            "predictions": all_preds.tolist(),
            "labels": all_labels.tolist(),
        }
        all_fold_results.append(fold_result)

        logging.info(f"\n  Fold {fold_idx + 1} 結果:")
        logging.info(f"    準確率: {accuracy:.3f}")
        logging.info(f"    MAE: {mae:.3f}")
        logging.info(f"    ±1 準確率: {off_by_one:.3f}")
        logging.info(f"    QWK: {kappa:.3f}")
        logging.info(f"    F1 (weighted): {f1:.3f}")
        logging.info(f"    Consensus-Level:")
        for level, res in consensus_results.items():
            logging.info(
                f"      {level}: n={res['n']}, acc={res.get('accuracy', 0):.3f}, "
                f"κ={res.get('kappa', 0):.3f}"
            )

        # 儲存模型
        torch.save({
            "model": phase3_state,
            "rce": phase3_rce_state,
        }, CHECKPOINT_DIR / f"rce_fold{fold_idx + 1}.pt")

    # 彙總結果
    logging.info(f"\n{'=' * 60}")
    logging.info("交叉驗證結果彙總")
    logging.info(f"{'=' * 60}")

    accs = [r["accuracy"] for r in all_fold_results]
    maes = [r["mae"] for r in all_fold_results]
    obo_accs = [r["off_by_one_accuracy"] for r in all_fold_results]
    kappas = [r["qwk"] for r in all_fold_results]
    f1s = [r["f1_weighted"] for r in all_fold_results]

    logging.info(f"  準確率:    {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    logging.info(f"  MAE:       {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    logging.info(f"  ±1 準確率: {np.mean(obo_accs):.3f} ± {np.std(obo_accs):.3f}")
    logging.info(f"  QWK:       {np.mean(kappas):.3f} ± {np.std(kappas):.3f}")
    logging.info(f"  F1:        {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # Consensus-level 彙總
    logging.info("\n  Consensus-Level 彙總:")
    for level in ["full_consensus", "majority_consensus", "low_consensus", "overall"]:
        level_accs = [r["consensus_results"][level].get("accuracy", 0)
                      for r in all_fold_results if r["consensus_results"][level]["n"] > 0]
        level_kappas = [r["consensus_results"][level].get("kappa", 0)
                        for r in all_fold_results if r["consensus_results"][level]["n"] > 0]
        if level_accs:
            logging.info(
                f"    {level}: acc={np.mean(level_accs):.3f}±{np.std(level_accs):.3f}, "
                f"κ={np.mean(level_kappas):.3f}±{np.std(level_kappas):.3f}"
            )

    for i, r in enumerate(all_fold_results):
        logging.info(
            f"  Fold {i+1}: Acc={r['accuracy']:.3f}, MAE={r['mae']:.3f}, "
            f"±1={r['off_by_one_accuracy']:.3f}, κ={r['qwk']:.3f}, F1={r['f1_weighted']:.3f}"
        )

    # 儲存結果
    results_summary = {
        "method": "FTDualNet + RCE (Rater Confusion Estimation)",
        "reference": "PARQ (2025) + Li et al. (2021)",
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "mean_mae": float(np.mean(maes)),
        "std_mae": float(np.std(maes)),
        "mean_off_by_one": float(np.mean(obo_accs)),
        "std_off_by_one": float(np.std(obo_accs)),
        "mean_qwk": float(np.mean(kappas)),
        "std_qwk": float(np.std(kappas)),
        "mean_f1": float(np.mean(f1s)),
        "std_f1": float(np.std(f1s)),
        "fold_results": all_fold_results,
    }

    results_file = OUTPUT_DIR / "rce_cv_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, ensure_ascii=False, indent=2)
    logging.info(f"\n結果已儲存至 {results_file}")

    return all_fold_results


# ===========================
# 主程式
# ===========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FTDualNet + RCE 訓練")
    parser.add_argument(
        "--mode",
        choices=["cv"],
        default="cv",
        help="訓練模式: cv=交叉驗證",
    )
    args = parser.parse_args()

    if args.mode == "cv":
        print("執行 5-Fold 交叉驗證訓練（RCE 模式）...")
        run_cross_validation()
