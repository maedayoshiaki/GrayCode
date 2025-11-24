import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import List, Tuple, Optional, Union
from enum import Enum


class AggregationMethod(Enum):
    """複数のピクセルが同じ位置にマップされる場合の集約方法"""

    MEAN = "mean"  # 平均
    MAX = "max"  # 最大値
    MIN = "min"  # 最小値
    LAST = "last"  # 最後の値（上書き）
    # MEDIAN = "median" # PyTorchでの効率的な実装が難しいため、今回は省略またはCPU処理推奨


class InpaintMethod(Enum):
    """穴埋め補完の方法"""

    NONE = "none"
    CONV = "conv"  # 畳み込みによる簡易補完 (GPU対応)
    # TELEA/NS はOpenCV依存のため、GPU完全対応版としては簡易実装を提供


class PixelMapWarperTorch:
    """
    画素対応マップを使用して画像をワーピングするクラス (PyTorch版)

    Data flow:
        Input Image (B, C, H, W) or (C, H, W) -> Tensor
        Output Image (B, C, H_out, W_out)
    """

    def __init__(
        self,
        pixel_map: Union[
            List[Tuple[Tuple[float, float], Tuple[float, float]]],
            np.ndarray,
            torch.Tensor,
        ],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            pixel_map: 画素対応マップ。
                       List形式: [((src_x, src_y), (dst_x, dst_y)), ...]
                       Tensor/Numpy形式: shape (N, 4) -> [src_x, src_y, dst_x, dst_y]
            device: 計算に使用するデバイス ('cpu' or 'cuda')
        """
        self.device = device

        # データのロードとTensor化
        if isinstance(pixel_map, list):
            # List[Tuple] -> Tensor
            # 構造をフラットにして (N, 4) に変換
            flat_data = []
            for (sx, sy), (dx, dy) in pixel_map:
                flat_data.append([sx, sy, dx, dy])
            self.map_tensor = torch.tensor(
                flat_data, dtype=torch.float32, device=self.device
            )
        elif isinstance(pixel_map, np.ndarray):
            self.map_tensor = torch.from_numpy(pixel_map).float().to(self.device)
            if self.map_tensor.ndim == 3 and self.map_tensor.shape[1:] == (2, 2):
                # reshape (N, 2, 2) -> (N, 4) if necessary
                self.map_tensor = self.map_tensor.view(-1, 4)
        elif isinstance(pixel_map, torch.Tensor):
            self.map_tensor = pixel_map.float().to(self.device)

        # NaN除去
        mask = ~torch.isnan(self.map_tensor).any(dim=1)
        self.map_tensor = self.map_tensor[mask]

        self._cache_map_bounds()

    def _cache_map_bounds(self):
        """マップの範囲をキャッシュ"""
        if self.map_tensor.numel() == 0:
            self.src_bounds = (0, 0, 0, 0)
            self.dst_bounds = (0, 0, 0, 0)
            return

        src_x = self.map_tensor[:, 0]
        src_y = self.map_tensor[:, 1]
        dst_x = self.map_tensor[:, 2]
        dst_y = self.map_tensor[:, 3]

        self.src_bounds = (
            src_x.min().item(),
            src_y.min().item(),
            src_x.max().item(),
            src_y.max().item(),
        )
        self.dst_bounds = (
            dst_x.min().item(),
            dst_y.min().item(),
            dst_x.max().item(),
            dst_y.max().item(),
        )

    def forward_warp(
        self,
        src_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        aggregation: AggregationMethod = AggregationMethod.MEAN,
        inpaint: InpaintMethod = InpaintMethod.NONE,
        inpaint_iter: int = 3,
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        順変換 (Splatting)

        Args:
            src_img: 入力画像 Tensor (C, H, W) または (B, C, H, W). 値域は[0, 255]でも[0, 1]でも可。
            dst_size: (width, height)
            src_offset: (offset_x, offset_y)
            aggregation: 集約方法
            inpaint: 補完方法
            inpaint_iter: 補完の反復回数（半径に相当）
            crop_rect: (x, y, w, h)

        Returns:
            変換後画像 Tensor (same shape style as input)
        """
        # 入力形状の正規化 (B, C, H, W)
        is_batch = src_img.ndim == 4
        if not is_batch:
            src_img = src_img.unsqueeze(0)  # (1, C, H, W)

        B, C, H, W = src_img.shape
        src_img = src_img.to(self.device).float()

        # 出力サイズ計算
        if dst_size is None:
            dst_w = int(self.dst_bounds[2]) + 1
            dst_h = int(self.dst_bounds[3]) + 1
        else:
            dst_w, dst_h = dst_size

        # 1. 座標計算
        src_x = torch.floor(self.map_tensor[:, 0]).long() - src_offset[0]
        src_y = torch.floor(self.map_tensor[:, 1]).long() - src_offset[1]
        dst_x = torch.floor(self.map_tensor[:, 2]).long()
        dst_y = torch.floor(self.map_tensor[:, 3]).long()

        # 2. 範囲外のフィルタリング
        valid_mask = (
            (src_x >= 0)
            & (src_x < W)
            & (src_y >= 0)
            & (src_y < H)
            & (dst_x >= 0)
            & (dst_x < dst_w)
            & (dst_y >= 0)
            & (dst_y < dst_h)
        )

        # 有効なインデックスのみ抽出
        s_x = src_x[valid_mask]
        s_y = src_y[valid_mask]
        d_x = dst_x[valid_mask]
        d_y = dst_y[valid_mask]

        # フラットなインデックス
        dst_indices = d_y * dst_w + d_x  # (N_points,)

        # ピクセル値の取得 (B, C, N_points)
        # src_img[:, :, s_y, s_x] は advanced indexing
        pixel_values = src_img[:, :, s_y, s_x]

        # 出力バッファ作成
        out_img = torch.zeros(
            (B, C, dst_h * dst_w), device=self.device, dtype=torch.float32
        )
        count_img = torch.zeros(
            (1, 1, dst_h * dst_w), device=self.device, dtype=torch.float32
        )

        # 3. 集約処理 (Splatting)
        # index_add_ や scatter_reduce_ を使用

        if aggregation == AggregationMethod.MEAN:
            # 加算
            out_img.index_add_(2, dst_indices, pixel_values)

            # カウント（平均計算用）
            ones = (
                torch.ones_like(dst_indices, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            count_img.index_add_(2, dst_indices, ones)

            # 平均化
            mask = count_img > 0
            # ゼロ除算回避
            out_img = torch.where(mask, out_img / (count_img + 1e-8), out_img)

        elif aggregation == AggregationMethod.MAX:
            # 初期値を小さく
            out_img.fill_(-1e9)
            # scatter_reduce_ (PyTorch 1.12+)
            # "amax" は atomic max
            out_img.scatter_reduce_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
                reduce="amax",
                include_self=False,
            )

            # 値が入らなかった場所を0に戻す（または背景色）
            out_img[out_img == -1e9] = 0

            # マスク作成用
            ones = (
                torch.ones_like(dst_indices, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            count_img.index_add_(2, dst_indices, ones)

        elif aggregation == AggregationMethod.MIN:
            out_img.fill_(1e9)
            out_img.scatter_reduce_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
                reduce="amin",
                include_self=False,
            )
            out_img[out_img == 1e9] = 0

            ones = (
                torch.ones_like(dst_indices, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            count_img.index_add_(2, dst_indices, ones)

        elif aggregation == AggregationMethod.LAST:
            # Lastの場合は単純に書き込む（並列処理の場合、順序は保証されないことに注意）
            # 後勝ちにするため、通常の scatter_ （非決定論的）を使用
            out_img.scatter_(
                2, dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1), pixel_values
            )

            ones = (
                torch.ones_like(dst_indices, dtype=torch.float32)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            count_img.index_add_(2, dst_indices, ones)

        # Reshape back to image
        out_img = out_img.view(B, C, dst_h, dst_w)
        count_img = count_img.view(1, 1, dst_h, dst_w)

        # 4. Inpainting (Simple GPU convolution based)
        if inpaint == InpaintMethod.CONV:
            out_img = self._apply_inpaint_conv(
                out_img, count_img, iterations=inpaint_iter
            )

        # 5. Crop
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            out_img = out_img[:, :, cy : cy + ch, cx : cx + cw]

        if not is_batch:
            out_img = out_img.squeeze(0)

        return out_img

    def backward_warp(
        self,
        src_img: torch.Tensor,
        dst_size: Optional[Tuple[int, int]] = None,
        src_offset: Tuple[int, int] = (0, 0),
        mode: str = "bilinear",
        padding_mode: str = "zeros",
        crop_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> torch.Tensor:
        """
        逆変換 (Sampling)
        grid_sampleを使用するため、スパースなマップからデンスなグリッドを作成します。

        注意: pixel_mapがスパース（点群）の場合、事前にDenseなマップへの補間が必要ですが、
        ここでは単純にNearest Neighborで埋める実装になっています。
        """
        is_batch = src_img.ndim == 4
        if not is_batch:
            src_img = src_img.unsqueeze(0)

        B, C, H, W = src_img.shape
        src_img = src_img.to(self.device).float()

        if dst_size is None:
            dst_w = int(self.src_bounds[2]) + 1
            dst_h = int(self.src_bounds[3]) + 1
        else:
            dst_w, dst_h = dst_size

        # 1. Flow Field (Grid) の作成 (dst -> src)
        # gridの初期化 (H, W, 2)
        # 値が入っていない場所を識別するために -2 (範囲外) で初期化
        grid = torch.full(
            (1, dst_h, dst_w, 2), -2.0, device=self.device, dtype=torch.float32
        )

        # 座標取得
        src_x = self.map_tensor[:, 0] - src_offset[0]
        src_y = self.map_tensor[:, 1] - src_offset[1]
        dst_x = torch.floor(self.map_tensor[:, 2]).long()
        dst_y = torch.floor(self.map_tensor[:, 3]).long()

        # 範囲チェック
        valid = (dst_x >= 0) & (dst_x < dst_w) & (dst_y >= 0) & (dst_y < dst_h)
        d_x = dst_x[valid]
        d_y = dst_y[valid]
        s_x = src_x[valid]
        s_y = src_y[valid]

        # 座標の正規化: grid_sampleは [-1, 1] の範囲を期待する
        # -1: 左端/上端, 1: 右端/下端
        norm_s_x = 2.0 * s_x / (W - 1) - 1.0
        norm_s_y = 2.0 * s_y / (H - 1) - 1.0

        # Gridに値を埋める (Last wins)
        # ここはスパースなマップの場合、穴あきになります
        grid[0, d_y, d_x, 0] = norm_s_x
        grid[0, d_y, d_x, 1] = norm_s_y

        # スパースな穴を埋める (簡易的Nearest Neighbor)
        # マップ自体がDenseであることを前提とするか、ここで補間が必要
        # この実装では、マップが存在しないピクセルは padding_mode に従います（初期値が-2なので）

        # 2. Grid Sample
        # grid shape: (N, H_out, W_out, 2)
        grid_batch = grid.expand(B, -1, -1, -1)

        # align_corners=True が OpenCV の挙動に近い
        out_img = F.grid_sample(
            src_img,
            grid_batch,
            mode=mode,
            padding_mode=padding_mode,
            align_corners=True,
        )

        # 3. Crop
        if crop_rect is not None:
            cx, cy, cw, ch = crop_rect
            out_img = out_img[:, :, cy : cy + ch, cx : cx + cw]

        if not is_batch:
            out_img = out_img.squeeze(0)

        return out_img

    def _apply_inpaint_conv(
        self, img: torch.Tensor, count_img: torch.Tensor, iterations: int = 3
    ) -> torch.Tensor:
        """
        GPU上で完結する簡易的なInpainting
        データの無い場所(count_img==0)を、周囲の有効なデータの平均で埋める処理を繰り返す
        """
        # マスク: データがない場所が1
        mask = (count_img == 0).float()

        # カーネル: 3x3 平均フィルタ (中心以外)
        kernel = torch.tensor(
            [[1, 1, 1], [1, 0, 1], [1, 1, 1]], device=self.device, dtype=torch.float32
        )
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, 3, 3).repeat(
            img.shape[1], 1, 1, 1
        )  # (C, 1, 3, 3) for depthwise conv

        current_img = img.clone()

        for _ in range(iterations):
            # データの欠損している部分を抽出
            is_hole = mask > 0.5

            # 周囲の平均を計算（Depthwise Convolution）
            # padding=1でサイズ維持
            neighbor_avg = F.conv2d(current_img, kernel, padding=1, groups=img.shape[1])

            # 穴の部分だけ更新
            # current_img = current_img * (1 - is_hole) + neighbor_avg * is_hole
            # ただし、neighbor_avgも穴を含んでいる可能性があるので、本来はマスク付き畳み込みが必要だが
            # 簡易実装として、値を更新していく
            current_img = torch.where(is_hole, neighbor_avg, current_img)

            # マスクの更新（少し縮小させる＝穴が埋まったとみなす）
            # 厳密なマスク更新ではないが、反復で徐々に埋まる

        return current_img


# --- 使用例 ---
def main():
    # ダミーデータ作成
    # 100x100の画像を 200x200のキャンバスに配置するようなマップ
    src_h, src_w = 100, 100
    dst_h, dst_w = 200, 200

    # 画像
    src_img_np = np.random.randint(0, 255, (src_h, src_w, 3), dtype=np.uint8)
    src_img_tensor = torch.from_numpy(src_img_np).permute(2, 0, 1)  # (3, H, W)

    # マップ: 単純な平行移動とスケーリング
    # src(x, y) -> dst(x*1.5 + 20, y*1.5 + 20)
    map_list = []
    for y in range(src_h):
        for x in range(src_w):
            dx = x * 1.5 + 20
            dy = y * 1.5 + 20
            map_list.append(((x, y), (dx, dy)))

    # Warper初期化
    warper = PixelMapWarperTorch(map_list, device="cpu")  # GPUがあれば 'cuda'

    # Forward Warp
    print("Forward warping...")
    out_forward = warper.forward_warp(
        src_img_tensor,
        dst_size=(dst_w, dst_h),
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.CONV,
    )

    # Backward Warp
    print("Backward warping...")
    out_backward = warper.backward_warp(src_img_tensor, dst_size=(dst_w, dst_h))

    # 保存用に変換
    out_f_np = out_forward.permute(1, 2, 0).byte().cpu().numpy()
    out_b_np = out_backward.permute(1, 2, 0).byte().cpu().numpy()

    cv2.imwrite("torch_forward.jpg", out_f_np)
    cv2.imwrite("torch_backward.jpg", out_b_np)
    print("Done.")


if __name__ == "__main__":
    main()
