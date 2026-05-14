"""
FTDualNet 模型架構
基於 PDualNet (Scientific Reports, 2025) 改造

架構：
1. FeatureAutoencoder (SiVE-inspired)：壓縮特徵至潛在空間
2. FeatureAttention (DiSE-inspired)：學習特徵間的交互關係
3. FTDualNet：完整模型，包含評分 Head + 診斷 Head
4. UncertaintyWeightedLoss (Kendall 2018)：自動平衡多任務損失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mtl_config import (
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
)


# ===========================
# 1. 特徵自編碼器 (SiVE-inspired)
# ===========================

class FeatureAutoencoder(nn.Module):
    """
    特徵自編碼器，受 PDualNet 的 SiVE（Single-Visit Embedding）啟發

    將高維特徵壓縮至低維潛在空間，學習特徵的緊湊表示。
    預訓練後，僅保留 Encoder 部分作為後續模型的骨幹。

    架構：
        Encoder: input_dim → 64 → BN → ReLU → Dropout → 32 → BN → ReLU → 16 (latent)
        Decoder: 16 → 32 → ReLU → 64 → ReLU → input_dim (reconstruct)
    """

    def __init__(self, input_dim=None):
        super().__init__()

        if input_dim is None:
            input_dim = NUM_FEATURES

        # 編碼器
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

        # 最後一層壓縮到潛在空間
        encoder_layers.append(nn.Linear(prev_dim, LATENT_DIM))

        self.encoder = nn.Sequential(*encoder_layers)

        # 解碼器（對稱結構）
        decoder_layers = []
        prev_dim = LATENT_DIM
        for hidden_dim in DECODER_HIDDEN_DIMS:
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim

        # 最後一層重建原始維度
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        """編碼：特徵 → 潛在表示"""
        return self.encoder(x)

    def decode(self, z):
        """解碼：潛在表示 → 重建特徵"""
        return self.decoder(z)

    def forward(self, x):
        """前向傳播：編碼 → 解碼"""
        z = self.encode(x)
        reconstructed = self.decode(z)
        return reconstructed, z


# ===========================
# 2. 特徵注意力層 (DiSE-inspired)
# ===========================

class FeatureAttention(nn.Module):
    """
    特徵交互注意力層，受 PDualNet 的 DiSE（Disease State Embedding）啟發

    原始 PDualNet 使用 Transformer 處理多次就診的時序關係。
    由於我們的資料是橫斷面（單次就診），改為使用 Self-Attention
    學習潛在特徵維度之間的交互關係。

    架構：
        Input: (batch, latent_dim) → reshape to (batch, 1, latent_dim)
        MultiheadAttention(embed_dim=latent_dim, num_heads=4)
        LayerNorm + Residual connection
        Output: (batch, latent_dim)
    """

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
        """
        參數：
            x: (batch, latent_dim)
        回傳：
            (batch, latent_dim) 精煉後的表示
        """
        # reshape: (batch, latent_dim) → (batch, 1, latent_dim)
        x_seq = x.unsqueeze(1)

        # Self-Attention + Residual
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        x_seq = self.layer_norm(x_seq + attn_out)

        # Feed-Forward + Residual
        ffn_out = self.ffn(x_seq)
        x_seq = self.layer_norm2(x_seq + ffn_out)

        # reshape back: (batch, 1, latent_dim) → (batch, latent_dim)
        return x_seq.squeeze(1)


# ===========================
# 3. FTDualNet 完整模型
# ===========================

class FTDualNet(nn.Module):
    """
    FTDualNet：基於 PDualNet 改造的雙任務手指敲擊評分模型

    完整架構：
        Input (33 features)
          → Encoder (SiVE) → 16-dim latent
          → Attention (DiSE) → 16-dim refined
          → Score Head → 4 classes (分數 0, 1, 2, 3+)
          → Diagnosis Head → 1 probability (PD yes/no)
    """

    def __init__(self, input_dim=None):
        super().__init__()

        if input_dim is None:
            input_dim = NUM_FEATURES

        # SiVE-inspired: 特徵編碼器（從預訓練的 Autoencoder 載入）
        self.autoencoder = FeatureAutoencoder(input_dim)

        # DiSE-inspired: 特徵注意力層
        self.attention = FeatureAttention()

        # 雙任務 Head
        # Head 1: 評分 (0, 1, 2, 3+)
        self.score_head = nn.Sequential(
            nn.Linear(LATENT_DIM, SCORE_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(SCORE_HEAD_HIDDEN, SCORE_NUM_CLASSES),
        )

        # Head 2: 診斷 (PD yes/no, 二元分類)
        self.diagnosis_head = nn.Sequential(
            nn.Linear(LATENT_DIM, DIAG_HEAD_HIDDEN),
            nn.ReLU(),
            nn.Linear(DIAG_HEAD_HIDDEN, 1),
        )

    def get_encoder(self):
        """取得編碼器（用於凍結/解凍）"""
        return self.autoencoder.encoder

    def freeze_encoder(self):
        """凍結編碼器參數（第二階段訓練用）"""
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """解凍編碼器參數（第三階段訓練用）"""
        for param in self.autoencoder.encoder.parameters():
            param.requires_grad = True

    def get_parameter_groups(self, encoder_lr, head_lr):
        """
        取得不同學習率的參數群組（第三階段訓練用）

        參數：
            encoder_lr: 編碼器的學習率（較低）
            head_lr:    attention + heads 的學習率（較高）

        回傳：
            list of dicts，可直接傳入 optimizer
        """
        return [
            {"params": self.autoencoder.encoder.parameters(), "lr": encoder_lr},
            {"params": self.attention.parameters(), "lr": head_lr},
            {"params": self.score_head.parameters(), "lr": head_lr},
            {"params": self.diagnosis_head.parameters(), "lr": head_lr},
        ]

    def forward(self, x):
        """
        前向傳播

        參數：
            x: (batch, input_dim) 特徵向量

        回傳：
            dict:
                "score_logits":  (batch, SCORE_NUM_CLASSES) 評分 logits
                "diag_prob":     (batch, 1) 診斷機率
                "latent":        (batch, latent_dim) 潛在表示
        """
        # 編碼
        latent = self.autoencoder.encode(x)

        # 注意力精煉
        refined = self.attention(latent)

        # 雙任務輸出
        score_logits = self.score_head(refined)
        diag_prob = torch.sigmoid(self.diagnosis_head(refined))

        return {
            "score_logits": score_logits,
            "diag_prob": diag_prob,
            "latent": latent,
        }

    def predict(self, x):
        """
        推論用：回傳分數預測、信心度、診斷機率

        參數：
            x: (batch, input_dim) 或 (input_dim,) 特徵向量

        回傳：
            dict:
                "predicted_score": int, 預測分數 0-3+
                "score_probs":     list, 各分數的機率
                "confidence":      float, 預測分數的信心度
                "diag_prob":       float, 診斷為 PD 的機率
        """
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
# 4. 損失函數
# ===========================

class UncertaintyWeightedLoss(nn.Module):
    """
    基於不確定性的多任務損失加權 (Kendall et al., CVPR 2018)

    每個任務有一個可學習的 log_var 參數，自動平衡不同任務的損失量級。

    公式：
        Loss_i = exp(-log_var_i) * task_loss_i + log_var_i
        Total  = sum(Loss_i)

    當某個任務的損失很大時，log_var 會自動增大，降低該任務的權重；
    反之亦然，實現自適應的多任務損失平衡。
    """

    def __init__(self, num_tasks=2):
        super().__init__()
        # 可學習的 log(sigma^2) 參數，初始為 0
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        參數：
            losses: list of scalar tensors, 各任務的損失值

        回傳：
            total_loss: scalar tensor, 加權後的總損失
        """
        total_loss = 0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss + self.log_vars[i]
        return total_loss


# ===========================
# 獨立執行：測試模型
# ===========================

if __name__ == "__main__":
    print("=" * 60)
    print("FTDualNet 模型架構測試")
    print("=" * 60)

    # 測試自編碼器
    print("\n1. 測試 FeatureAutoencoder")
    ae = FeatureAutoencoder()
    x = torch.randn(8, NUM_FEATURES)
    reconstructed, latent = ae(x)
    print(f"   輸入: {x.shape}")
    print(f"   潛在表示: {latent.shape}")
    print(f"   重建: {reconstructed.shape}")

    # 測試注意力層
    print("\n2. 測試 FeatureAttention")
    attn = FeatureAttention()
    refined = attn(latent)
    print(f"   輸入: {latent.shape}")
    print(f"   輸出: {refined.shape}")

    # 測試完整模型
    print("\n3. 測試 FTDualNet")
    model = FTDualNet()
    outputs = model(x)
    print(f"   輸入: {x.shape}")
    print(f"   評分 logits: {outputs['score_logits'].shape}")
    print(f"   診斷機率: {outputs['diag_prob'].shape}")
    print(f"   潛在表示: {outputs['latent'].shape}")

    # 測試推論
    print("\n4. 測試推論")
    pred = model.predict(x[0])
    print(f"   預測分數: {pred['predicted_score']}")
    print(f"   各分數機率: {[f'{p:.3f}' for p in pred['score_probs']]}")
    print(f"   信心度: {pred['confidence']:.3f}")
    print(f"   診斷機率: {pred['diag_prob']:.3f}")

    # 測試損失函數
    print("\n5. 測試 UncertaintyWeightedLoss")
    criterion = UncertaintyWeightedLoss(num_tasks=2)
    target_scores = torch.randint(0, SCORE_NUM_CLASSES, (8,))
    loss_score = F.cross_entropy(outputs["score_logits"], target_scores)
    loss_diag = F.binary_cross_entropy(
        outputs["diag_prob"].squeeze(), torch.randint(0, 2, (8,)).float()
    )
    total_loss = criterion([loss_score, loss_diag])
    print(f"   評分損失: {loss_score.item():.4f}")
    print(f"   診斷損失: {loss_diag.item():.4f}")
    print(f"   加權總損失: {total_loss.item():.4f}")
    print(f"   學習到的 log_var: {criterion.log_vars.data.tolist()}")

    # 參數統計
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型總參數: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")

    # 測試凍結/解凍
    print("\n6. 測試凍結/解凍")
    model.freeze_encoder()
    frozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   凍結後可訓練參數: {frozen_params:,}")
    model.unfreeze_encoder()
    unfrozen_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   解凍後可訓練參數: {unfrozen_params:,}")

    print("\n" + "=" * 60)
    print("所有測試通過！")
    print("=" * 60)
