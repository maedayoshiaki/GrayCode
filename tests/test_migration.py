"""移行スクリプト (migrate.py) の回帰テスト。

旧 UV 規約 (proj 中心 = i+0.5) の保存データを pixel-is-point (proj 中心 = 整数)
へ移行する。プロジェクタ座標から一律 0.5 を引き、カメラ座標は不変であることを固定。
"""

from __future__ import annotations

import numpy as np

from graycode import migrate


def test_shift_proj_dict() -> None:
    d = {(1.5, 1.5): [(10, 10)], (2.5, 5.5): [(20, 20), (21, 21)]}
    out = migrate.shift_proj_dict(d)
    assert set(out.keys()) == {(1.0, 1.0), (2.0, 5.0)}
    assert out[(1.0, 1.0)] == [(10, 10)]
    assert out[(2.0, 5.0)] == [(20, 20), (21, 21)]  # cam 値は不変


def test_shift_p2c_compensated_array() -> None:
    # [proj_x, proj_y, cam_x, cam_y]
    arr = np.array([[3.5, 4.5, 100.0, 200.0], [0.5, 0.5, 1.0, 2.0]], dtype=np.float32)
    out = migrate.shift_p2c_compensated_array(arr)
    assert np.allclose(out[:, 0:2], [[3.0, 4.0], [0.0, 0.0]])  # proj -0.5
    assert np.allclose(out[:, 2:4], arr[:, 2:4])  # cam 不変


def test_shift_c2p_object_array() -> None:
    # (N,2,2) object: [[cam],[proj]]
    arr = np.empty((2, 2, 2), dtype=object)
    arr[0] = [[10, 20], [3.5, 4.5]]
    arr[1] = [[11, 21], [5.5, 6.5]]
    out = migrate.shift_c2p_object_array(arr)
    assert float(out[0, 1, 0]) == 3.0 and float(out[0, 1, 1]) == 4.0  # proj -0.5
    assert float(out[1, 1, 0]) == 5.0 and float(out[1, 1, 1]) == 6.0
    assert float(out[0, 0, 0]) == 10 and float(out[0, 0, 1]) == 20  # cam 不変


def test_migrate_npy_p2c_compensated(tmp_path) -> None:
    p = tmp_path / "result_p2c_compensated_delaunay.npy"
    np.save(p, np.array([[3.5, 4.5, 9.0, 8.0]], dtype=np.float32))
    out = tmp_path / "out.npy"
    migrate.migrate_npy(str(p), str(out))
    res = np.load(out)
    assert np.allclose(res, [[3.0, 4.0, 9.0, 8.0]])


def test_migrate_npy_p2c_dict_roundtrip(tmp_path) -> None:
    p = tmp_path / "result_p2c.npy"
    np.save(p, np.array({(2.5, 3.5): [(7, 8)]}, dtype=object))
    out = tmp_path / "out.npy"
    migrate.migrate_npy(str(p), str(out))
    res = np.load(out, allow_pickle=True).item()
    assert res == {(2.0, 3.0): [(7, 8)]}


def test_migrate_npy_c2p_object_roundtrip(tmp_path) -> None:
    arr = np.empty((1, 2, 2), dtype=object)
    arr[0] = [[10, 20], [3.5, 4.5]]
    p = tmp_path / "result_c2p.npy"
    np.save(p, arr)
    out = tmp_path / "out.npy"
    migrate.migrate_npy(str(p), str(out))
    res = np.load(out, allow_pickle=True)
    assert float(res[0, 1, 0]) == 3.0 and float(res[0, 1, 1]) == 4.0
    assert float(res[0, 0, 0]) == 10.0


def test_migrate_csv_p2c(tmp_path) -> None:
    p = tmp_path / "result_p2c.csv"
    p.write_text("proj_x, proj_y, cam_x, cam_y\n3.5, 4.5, 100, 200\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    migrate.migrate_csv(str(p), str(out))
    text = out.read_text(encoding="utf-8").splitlines()
    assert text[0] == "proj_x, proj_y, cam_x, cam_y"
    cells = [c.strip() for c in text[1].split(",")]
    assert cells[0] == "3" and cells[1] == "4"  # proj -0.5
    assert cells[2] == "100" and cells[3] == "200"  # cam 不変


def test_migrate_csv_c2p_proj_columns_detected(tmp_path) -> None:
    # c2p CSV は proj 列が後ろ (cam_x, cam_y, proj_x, proj_y)
    p = tmp_path / "result_c2p.csv"
    p.write_text("cam_x, cam_y, proj_x, proj_y\n5, 6, 3.5, 4.5\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    migrate.migrate_csv(str(p), str(out))
    cells = [c.strip() for c in out.read_text(encoding="utf-8").splitlines()[1].split(",")]
    assert cells[0] == "5" and cells[1] == "6"  # cam 不変
    assert cells[2] == "3" and cells[3] == "4"  # proj -0.5
