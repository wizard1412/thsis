"""
FTDualNet + RCE 模型架構

基於 PARQ (2025) 和 Li et al. (2021)：
1. FeatureAutoencoder (SiVE-inspired)：與原版相同
2. FeatureAttention (DiSE-inspired)：與原版相同
3. FTDualNet：與原版相同的骨幹
4. RaterConfusionEstimation：為每位 rater 學習 confusion matrix
5. OrdinalFocalLoss：結合 focal loss 和 ordinal distance penalty
6. UncertaintyWeightedLoss：自動平衡多任務損失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from rce_config import (
    NUM_FEATURES,
    LATENT_DIM,
    ENCODER_HIDDEN_DIMS,
    DECODER_HIDDEN_DIMS,
    ATTENTION_HEADS,
    ATTENTION_DROPOUT,
    SCORE_HEAD_HIDDEN,
    SCORE_NUM_CLASSES,
    DIAG_HEAD_HIDDEN,
    DROPOUT,
    MAX_NUM_RATERS,
    OF_ALPHA,
    OF_GAMMA,
    OF_BETA,
    RCE_INIT_NOISE,
)


# ===========================
# 1. 特徵自編碼器 (SiVE-inspired)
# ===========================

class FeatureAutoencoder(nn.Module):
    """與原版完全相同"""

    def __init__(self, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = NUM_FEATURES

        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in ENCODER_HIDDEN_DIMS:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(DROPOUT),
            ])
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, LATENT_DIM))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        prev_dim = LATENT_DIM
        for hidden_dim in DECODER_HIDDEN_DIMS:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# ===========================
# 2. 特徵注意力層 (DiSE-inspired)
# ===========================

class FeatureAttention(nn.Module):
    """與原版完全相同"""

    def __init__(self, embed_dim=None, num_heads=None):
        super().__init__()
        if embed_dim is None:
            embed_dim = LATENT_DIM
        if num_heads is None:
            num_heads = ATTENTION_HEADS

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=ATTENTION_DROPOUT,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(ATTENTION_DROPOUT),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        self.layer_norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_seq = x.unsqueeze(1)
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.layer_norm(x_seq + attn_out)
        ffn_out = self.ffn(x_seq)
        x_seq = self.layer_norm2(x_seq + ffn_out)
        return x_seq.squeeze(1)


# ===========================
# 3. FTDualNet 骨幹模型
# ===========================

class FTDualNet(nn.Module):
    """與原版相同的骨幹網路"""

    def __init__(self, input_dim=None):
        super().__init__()
        if input_dim is None:
            input_dim = NUM_FEATURES

        self.autoencoder = FeatureAutoencoder(input_dim)
        self.attention = FeatureAttention()

        self.score_head = nn.Sequential(
            nn.Linear(LATENT_DIM, SCORE_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SCORE_HEAD_HIDDEN, SCORE_NUM_CLASSES),
        )

        self.diagnosis_head = nn.Sequential(
            nn.Linear(LATENT_DIM, DIAG_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIAG_HEAD_HIDDEN, 1),
        )

    def get_encoder(self):
        return self.autoencoder.encoder

    def freeze_encoder(self):
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, encoder_lr, head_lr):
        return [
            {"params": self.autoencoder.encoder.parameters(), "lr": encoder_lr},
            {"params": self.attention.parameters(), "lr": head_lr},
            {"params": self.score_head.parameters(), "lr": head_lr},
            {"params": self.diagnosis_head.parameters(), "lr": head_lr},
        ]

    def forward(self, x):
        latent = self.autoencoder.encode(x)
        refined = self.attention(latent)
        score_logits = self.score_head(refined)
        diag_prob = torch.sigmoid(self.diagnosis_head(refined))

        return {
            "score_logits": score_logits,
            "diag_prob": diag_prob,
            "latent": latent,
        }

    def predict(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            score_probs = F.softmax(outputs["score_logits"], dim=-1)
            predicted_score = score_probs.argmax(dim=-1).item()
            confidence = score_probs[0, predicted_score].item()
            diag_prob = outputs["diag_prob"].item()
        return {
            "predicted_score": predicted_score,
            "score_probs": score_probs[0].cpu().tolist(),
            "confidence": confidence,
            "diag_prob": diag_prob,
        }


# ===========================
# 4. Rater Confusion Estimation (RCE)
# ===========================

class RaterConfusionEstimation(nn.Module):
    """
    為每位 rater 學習一個 C×C confusion matrix A^(r)

    A^(r)[c', c] = P(rater r 給分 c' | 真實分數為 c)

    基於 Li et al. (2021) 和 Tanno et al. (2019)：
    - 初始化為 identity matrix（假設 rater 完美）
    - 訓練中學習每位 rater 的偏差模式
    - 使用 softmax 確保每列加總為 1（simplex projection）
    - Trace regularization 鼓勵學到有意義的偏差（避免退化為 identity）
    """

    def __init__(self, num_raters=None, num_classes=None):
        super().__init__()
        if num_raters is None:
            num_raters = MAX_NUM_RATERS
        if num_classes is None:
            num_classes = SCORE_NUM_CLASSES

        self.num_raters = num_raters
        self.num_classes = num_classes

        # 初始化：identity + 小噪音
        # 使用 logit 空間（pre-softmax），identity 對應大對角線值
        init_logits = torch.zeros(num_raters, num_classes, num_classes)
        for r in range(num_raters):
            # 對角線設大值，使 softmax 後接近 identity
            init_logits[r] = torch.eye(num_classes) * 5.0
            # 加小噪音打破對稱
            init_logits[r] += torch.randn(num_classes, num_classes) * RCE_INIT_NOISE

        self.confusion_logits = nn.Parameter(init_logits)  # (R, C, C)

    def get_matrix(self, rater_idx):
        """取得 rater r 的 confusion matrix（經 softmax 正規化）"""
        logits = self.confusion_logits[rater_idx]  # (C, C)
        # 沿 dim=0（row）做 softmax → 每列加總為 1
        # A[c', c] = P(predict c' | true c)
        return F.softmax(logits, dim=0)

    def get_all_matrices(self):
        """取得所有 rater 的 confusion matrices"""
        # (R, C, C)
        return F.softmax(self.confusion_logits, dim=1)

    def transform(self, pred_prob, rater_idx):
        """
        將模型預測透過 rater confusion matrix 轉換

        pred_prob: (batch, C) — 模型的 softmax 輸出
        rater_idx: int — rater 索引

        回傳：(batch, C) — 轉換後的預測
        """
        A = self.get_matrix(rater_idx)  # (C, C)
        # A @ p：A 的每列對應一個 true class，每行對應一個 predicted class
        # 結果：rater r 會給出的分數分布
        return (A @ pred_prob.unsqueeze(-1)).squeeze(-1)  # (batch, C)

    def trace_penalty(self):
        """
        Trace regularization

        鼓勵 confusion matrix 偏離 identity（學到有意義的 rater 偏差）
        Li et al.: "encourages the estimated raters' noise to be maximally unreliable"
        損失 = mean(trace(A^r))，最小化 trace → 鼓勵 off-diagonal 元素增大
        """
        total = 0.0
        for r in range(self.num_raters):
            A = self.get_matrix(r)
            total += torch.trace(A)
        return total / self.num_raters


# ===========================
# 5. Ordinal Focal Loss
# ===========================

class OrdinalFocalLoss(nn.Module):
    """
    結合 ordinal distance penalty 的 focal loss

    基於 Li et al. (2021) 的 OF Loss：
    OF(y, p) = -Σ_c α(1-p_c)^γ · (1 + β·|y-ŷ|/C) · y_c · log(p_c)

    - Focal 部分 (1-p_c)^γ：專注於難分類的樣本
    - Ordinal 部分 |y-ŷ|/C：離正確分數越遠，penalty 越大
    """

    def __init__(self, num_classes=None, alpha=None, gamma=None, beta=None):
        super().__init__()
        if num_classes is None:
            num_classes = SCORE_NUM_CLASSES
        if alpha is None:
            alpha = OF_ALPHA
        if gamma is None:
            gamma = OF_GAMMA
        if beta is None:
            beta = OF_BETA

        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta

    def forward(self, pred_prob, target_class):
        """
        pred_prob:    (batch, C) — 預測機率分布（已經 softmax）
        target_class: (batch,)   — 真實分數 0~C-1
        """
        # 數值穩定
        pred_prob = pred_prob.clamp(min=1e-7, max=1.0 - 1e-7)

        # One-hot encode target
        y_onehot = F.one_hot(target_class, self.num_classes).float()  # (batch, C)

        # Ordinal distance weight
        pred_class = pred_prob.detach().argmax(dim=1).float()
        ordinal_dist = torch.abs(pred_class - target_class.float())
        ordinal_weight = 1.0 + self.beta * ordinal_dist / self.num_classes  # (batch,)

        # Focal weight
        focal_weight = self.alpha * (1.0 - pred_prob).pow(self.gamma)  # (batch, C)

        # Cross entropy with focal weighting
        log_prob = torch.log(pred_prob)
        loss = -(focal_weight * y_onehot * log_prob).sum(dim=1)  # (batch,)

        # Apply ordinal penalty
        loss = loss * ordinal_weight

        return loss.mean()


# ===========================
# 6. Uncertainty Weighted Loss
# ===========================

class UncertaintyWeightedLoss(nn.Module):
    """與原版相同"""

    def __init__(self, num_tasks=2):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


# ===========================
# 測試
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("FTDualNet + RCE 模型測試")
    print("=" * 60)

    batch_size = 8

    # 1. 測試 FTDualNet
    print("\n1. FTDualNet 前向傳播")
    model = FTDualNet()
    x = torch.randn(batch_size, NUM_FEATURES)
    outputs = model(x)
    print(f"   Score logits: {outputs['score_logits'].shape}")
    print(f"   Diag prob:    {outputs['diag_prob'].shape}")

    pred_prob = F.softmax(outputs["score_logits"], dim=-1)
    print(f"   Pred probs:   {pred_prob[0].detach().tolist()}")

    # 2. 測試 RCE
    print("\n2. RaterConfusionEstimation")
    rce = RaterConfusionEstimation()
    print(f"   Confusion matrices: {rce.confusion_logits.shape}")

    A0 = rce.get_matrix(0)
    print(f"   Rater 0 matrix (should be ~identity):")
    print(f"   {A0.detach().numpy().round(3)}")

    transformed = rce.transform(pred_prob, rater_idx=0)
    print(f"   Transformed probs: {transformed[0].detach().tolist()}")

    trace_pen = rce.trace_penalty()
    print(f"   Trace penalty: {trace_pen.item():.4f}")

    # 3. 測試 OrdinalFocalLoss
    print("\n3. OrdinalFocalLoss")
    of_loss = OrdinalFocalLoss()
    target = torch.randint(0, SCORE_NUM_CLASSES, (batch_size,))
    loss = of_loss(transformed, target)
    print(f"   Target: {target.tolist()}")
    print(f"   Loss: {loss.item():.4f}")

    # 4. 完整 RCE 訓練 loss 模擬
    print("\n4. 完整 RCE training loss")
    ratings = torch.randint(-1, SCORE_NUM_CLASSES, (batch_size, MAX_NUM_RATERS))
    ratings[:, 0] = torch.randint(0, SCORE_NUM_CLASSES, (batch_size,))  # 確保至少 rater 0 有值

    total_loss = torch.tensor(0.0)
    num_valid_raters = 0

    for r in range(MAX_NUM_RATERS):
        rater_scores = ratings[:, r]
        valid_mask = rater_scores >= 0

        if valid_mask.sum() == 0:
            continue

        transformed_r = rce.transform(pred_prob[valid_mask], rater_idx=r)
        loss_r = of_loss(transformed_r, rater_scores[valid_mask])
        total_loss = total_loss + loss_r
        num_valid_raters += 1

    total_loss = total_loss / max(num_valid_raters, 1)
    total_loss = total_loss + 0.1 * rce.trace_penalty()
    print(f"   Total RCE loss: {total_loss.item():.4f}")
    print(f"   Valid raters per batch: {num_valid_raters}")

    # 5. 參數統計
    total_params = sum(p.numel() for p in model.parameters())
    rce_params = sum(p.numel() for p in rce.parameters())
    print(f"\n模型參數: {total_params:,}")
    print(f"RCE 參數: {rce_params:,} ({MAX_NUM_RATERS} × {SCORE_NUM_CLASSES} × {SCORE_NUM_CLASSES})")

    print("\n" + "=" * 60)
    print("所有測試通過！")
    print("=" * 60)
