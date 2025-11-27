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


class InpaintMethod(Enum):
    """穴埋め補完の方法"""

    NONE = "none"
    CONV = "conv"  # 畳み込みによる簡易補完 (GPU対応)


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
            flat_data = []
            for (sx, sy), (dx, dy) in pixel_map:
                flat_data.append([sx, sy, dx, dy])
            self.map_tensor = torch.tensor(
                flat_data, dtype=torch.float32, device=self.device
            )
        elif isinstance(pixel_map, np.ndarray):
            self.map_tensor = torch.from_numpy(pixel_map).float().to(self.device)
            if self.map_tensor.ndim == 3 and self.map_tensor.shape[1:] == (2, 2):
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
        ソース画像のピクセルをマップに従ってデスティネーションへ飛ばします。
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
        pixel_values = src_img[:, :, s_y, s_x]

        # 出力バッファ作成
        out_img = torch.zeros(
            (B, C, dst_h * dst_w), device=self.device, dtype=torch.float32
        )
        count_img = torch.zeros(
            (1, 1, dst_h * dst_w), device=self.device, dtype=torch.float32
        )

        # 3. 集約処理 (Splatting)
        if aggregation == AggregationMethod.MEAN:
            out_img.index_add_(2, dst_indices, pixel_values)
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)
            mask = count_img > 0
            out_img = torch.where(mask, out_img / (count_img + 1e-8), out_img)

        elif aggregation == AggregationMethod.MAX:
            out_img.fill_(-1e9)
            out_img.scatter_reduce_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
                reduce="amax",
                include_self=False,
            )
            out_img[out_img == -1e9] = 0
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
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
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)

        elif aggregation == AggregationMethod.LAST:
            out_img.scatter_(
                2,
                dst_indices.unsqueeze(0).unsqueeze(0).expand(B, C, -1),
                pixel_values,
            )
            ones = torch.ones_like(dst_indices, dtype=torch.float32).view(1, 1, -1)
            count_img.index_add_(2, dst_indices, ones)

        # Reshape back to image
        out_img = out_img.view(B, C, dst_h, dst_w)
        count_img = count_img.view(1, 1, dst_h, dst_w)

        # 4. Inpainting
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
        fill_grid_iter: int = 5,  # グリッド自体の補間回数
    ) -> torch.Tensor:
        """
        逆変換 (Sampling)
        Forward Map (src -> dst) の情報を使って Backward Map (dst -> src) を構築し、
        グリッドの隙間を畳み込みで埋めた後に grid_sample を実行します。
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

        # ---------------------------------------------------------
        # 1. Flow Field (Grid) の作成 (dst -> src)
        # ---------------------------------------------------------

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

        # --- グリッドを画像として扱い、穴埋めを行う ---

        # グリッド用バッファ: (1, 2, H, W) -> Channel dim is (x, y) coordinates
        # 初期値は0
        grid_map = torch.zeros(
            (1, 2, dst_h, dst_w), device=self.device, dtype=torch.float32
        )

        # データの有無を管理するマスク (1, 1, H, W)
        grid_count = torch.zeros(
            (1, 1, dst_h, dst_w), device=self.device, dtype=torch.float32
        )

        # フラットなインデックス計算
        dst_indices = d_y * dst_w + d_x

        # ソース座標を (2, N) にまとめる
        src_coords = torch.stack([s_x, s_y], dim=0)  # (2, N_points)

        # Scatterを使ってグリッドに座標を書き込む (Last wins)
        # shape合わせ: (1, 2, H*W)
        grid_flat = grid_map.view(1, 2, -1)
        mask_flat = grid_count.view(1, 1, -1)

        # 値の書き込み
        grid_flat.scatter_(
            2, dst_indices.unsqueeze(0).expand(1, 2, -1), src_coords.unsqueeze(0)
        )
        mask_flat.scatter_(
            2,
            dst_indices.unsqueeze(0).expand(1, 1, -1),
            torch.ones_like(dst_indices, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
        )

        # ビューを元に戻す
        grid_map = grid_flat.view(1, 2, dst_h, dst_w)
        grid_count = mask_flat.view(1, 1, dst_h, dst_w)

        # ★ グリッド自体の穴埋め ★
        # 生座標の状態で補間することで、近傍のソース座標で埋める
        if fill_grid_iter > 0:
            grid_map = self._apply_inpaint_conv(
                grid_map, grid_count, iterations=fill_grid_iter
            )

        # ---------------------------------------------------------
        # 2. 座標の正規化と Grid Sample
        # ---------------------------------------------------------

        # grid_sample用に (N, H, W, 2) に変換
        grid = grid_map.permute(0, 2, 3, 1)

        # 座標の正規化: [-1, 1]
        # grid[..., 0] is x, grid[..., 1] is y
        # 分母が0にならないよう max(..., 1)
        grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

        # バッチサイズに合わせて拡張
        grid_batch = grid.expand(B, -1, -1, -1)

        # Grid Sample実行
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
            current_img = torch.where(is_hole, neighbor_avg, current_img)

            # マスクの更新（少し縮小させる＝穴が埋まったとみなす）
            # 注: マスク自体もconvして侵食させるのが正確だが、簡易的に画像の値更新のみで進める
            # 厳密には count_img 自体も伝播させる必要があるが、数回のイテレーションなら
            # 周囲から値が染み出す効果だけで十分な場合が多い。

            # 今回はmaskも更新しないと「まだ埋まってない」と判断されて更新が止まる可能性があるため、
            # マスクも収縮(Erosion)させる
            # Max pooling で近傍に0(データあり)があれば0にする -> Erosion in mask(1=hole) means Dilation of data
            mask = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)

        return current_img


# --- 使用例 ---
def main():
    # GPU設定
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ダミーデータ作成
    # 100x100の画像を 200x200のキャンバスに配置するようなマップ
    src_h, src_w = 100, 100
    dst_h, dst_w = 200, 200

    # 画像: グラデーションで見やすくする
    src_img_np = np.zeros((src_h, src_w, 3), dtype=np.uint8)
    for y in range(src_h):
        for x in range(src_w):
            src_img_np[y, x] = [x * 2, y * 2, (x + y)]

    src_img_tensor = torch.from_numpy(src_img_np).permute(2, 0, 1)  # (3, H, W)

    # マップ: 単純な平行移動とスケーリング
    # src(x, y) -> dst(x*1.8 + 20, y*1.8 + 20)
    # 拡大率が大きいほど穴が開きやすい
    map_list = []
    scale = 1.8
    for y in range(src_h):
        for x in range(src_w):
            dx = x * scale + 20
            dy = y * scale + 20
            map_list.append(((x, y), (dx, dy)))

    # Warper初期化
    warper = PixelMapWarperTorch(map_list, device=device)

    # Forward Warp
    print("Forward warping...")
    out_forward = warper.forward_warp(
        src_img_tensor,
        dst_size=(dst_w, dst_h),
        aggregation=AggregationMethod.MEAN,
        inpaint=InpaintMethod.CONV,
        inpaint_iter=3,
    )

    # Backward Warp
    print("Backward warping...")
    # fill_grid_iterを増やすと、より広い隙間も埋められる
    out_backward = warper.backward_warp(
        src_img_tensor, dst_size=(dst_w, dst_h), fill_grid_iter=10
    )

    # 保存用に変換
    out_f_np = out_forward.permute(1, 2, 0).byte().cpu().numpy()
    out_b_np = out_backward.permute(1, 2, 0).byte().cpu().numpy()

    cv2.imwrite("torch_forward.jpg", out_f_np)
    cv2.imwrite("torch_backward.jpg", out_b_np)
    print("Done. Check torch_forward.jpg and torch_backward.jpg")


if __name__ == "__main__":
    main()
