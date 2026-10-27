#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
asdt1_convert.py — AS-DT1 Mobile の記録ファイル (.asdt1) を PLY / CSV / GPX / npz に展開する

使い方:
    python asdt1_convert.py REC001_20261105_093000.asdt1              # 同名フォルダに展開
    python asdt1_convert.py REC001.asdt1 -o out_dir                    # 出力先を指定
    python asdt1_convert.py REC001.asdt1 --no-ply                      # PLYを書かない（高速）
    python asdt1_convert.py REC001.asdt1 --flip-table                  # 並び替え表を反転（画像が不自然なときの検証用）

出力:
    frames.npz   … xyz_pic (N,576,3) float32 [m] センサ座標系(X下,Y左,Z前), xyz_cam (右,下,前)=(−Y,X,Z),
                   xyz2_pic 第2反射, flag (0無効/1第1/2第2), host_epoch_ms, tm_ms, ts, temp_C, n_points,
                   gps_* (各コマに付与されたスマホGPS), frame_no
    times.csv    … 1コマ1行の時刻台帳（旧ワークフローの times.csv に相当。スマホ時計で記録）
    ply/frame_NNNN.ply … (NNNN は times.csv の seq) lidar-viewer互換(ASCII, float x y z [mm], 576行, 無効点は 0 0 0)。
                   座標系は viewer と同じ(x右, y下, z前)。点の並びは画像順（行優先24×24）。
    track.gpx    … スマホGPSの軌跡（記録中の全測位）＋マーカー
    imu.csv      … 内蔵IMU(加速度[g]・角速度[deg/s]・温度・時刻カウンタ)
    markers.csv  … マーカー（区間開始/終了、距離標など）と時刻・GPS
    summary.json … 機器情報・設定・集計

必要なもの: Python 3.9+, numpy
出典: Sony AS-DT1 API Manual Ver.1.01 / SDK_for_AS-DT1_Ver1.01 (as_dt1_api_pcd.py) の展開ロジックを忠実に移植。
"""
import argparse, csv, json, os, struct, sys, datetime
import numpy as np

# 計測出力順 j → 画像順 MPDATATBL[j]  (Sony SDK _MPDATATBL; API manual p.11 の表と同値)
MPDATATBL = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
    336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
    528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
    456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
    504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575
]
MPDATATBL_MPXY = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
    48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71,
    24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
    72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95,
    96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
    120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
    216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287,
    288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
    336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359,
    312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
    360, 361, 362, 363, 364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 380, 381, 382, 383,
    384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407,
    432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 453, 454, 455,
    408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431,
    456, 457, 458, 459, 460, 461, 462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479,
    480, 481, 482, 483, 484, 485, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
    528, 529, 530, 531, 532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551,
    504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527,
    552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575
]

MAGIC = b'ASDT1LOG'
REC_MAGIC = 0x31544441
REC_FRAME, REC_GPS, REC_MARK, REC_INFO = 1, 2, 3, 4
FRAME_FIXED = 84
MODE_BY_CODE = {1: '20m', 2: '30mstd', 3: '30m15f', 4: '30m30f', 5: '40m'}

INV_MPDATATBL = [0] * 576
for _k in range(576):
    INV_MPDATATBL[MPDATATBL[_k]] = _k      # = Sony の _xyDataTbl


def unpack_xyz(mp: bytes, n: int) -> np.ndarray:
    """1点15バイト(20bit×6, big-endian)を展開。戻り値 (n,6) int32、単位 0.25mm。Sony SDK と同一の式。"""
    dd = np.frombuffer(mp, dtype=np.uint8).astype(np.int64).reshape(n, 15)
    x1 = dd[:, 0] << 12 | dd[:, 1] << 4 | dd[:, 2] >> 4
    y1 = (dd[:, 2] & 0x0f) << 16 | dd[:, 3] << 8 | dd[:, 4]
    z1 = dd[:, 5] << 12 | dd[:, 6] << 4 | dd[:, 7] >> 4
    x2 = (dd[:, 7] & 0x0f) << 16 | dd[:, 8] << 8 | dd[:, 9]
    y2 = dd[:, 10] << 12 | dd[:, 11] << 4 | dd[:, 12] >> 4
    z2 = (dd[:, 12] & 0x0f) << 16 | dd[:, 13] << 8 | dd[:, 14]
    out = np.stack([x1, y1, z1, x2, y2, z2], axis=1)
    for c in (0, 1, 3, 4):                 # X,Y は符号付き20bit（Zは符号なし）
        v = out[:, c]
        out[:, c] = np.where(v & 0x80000, v - 0x100000, v)
    return out.astype(np.int32)


def to_picture(raw6: np.ndarray, n: int, use_second=True, flip_table=False):
    """画像順(24×24, 行優先)の点群 [mm]。Sony make_pcd_from_raw と同じ第2反射フォールバック。"""
    pts = np.zeros((576, 3), np.float32)
    pts2 = np.zeros((576, 3), np.float32)
    flag = np.zeros(576, np.uint8)
    tbl = MPDATATBL if flip_table else INV_MPDATATBL
    for k in range(576):
        j = tbl[k]
        if j >= n:
            continue
        x, y, z = raw6[j, 0] / 4, raw6[j, 1] / 4, raw6[j, 2] / 4
        x2, y2, z2 = raw6[j, 3] / 4, raw6[j, 4] / 4, raw6[j, 5] / 4
        pts2[k] = (x2, y2, z2)
        if z >= 50000:                      # 第1反射がレンジ外 → 第2反射が有効なら採用
            if use_second and z2 < 50000:
                x, y, z = x2, y2, z2
                flag[k] = 2 if z != 0 else 0
            else:
                x = y = z = 0
        else:
            flag[k] = 1 if z != 0 else 0
        pts[k] = (x, y, z)
    return pts, pts2, flag


def decode_imu(rec: bytes):
    h = rec[0]
    ax, ay, az, gx, gy, gz = struct.unpack('>hhhhhh', rec[1:13])
    t = struct.unpack('>b', rec[13:14])[0]
    ts = struct.unpack('>H', rec[14:16])[0]
    return dict(head=h, ax=ax / 8192, ay=ay / 8192, az=az / 8192, gx=gx / 65.536, gy=gy / 65.536, gz=gz / 65.536, temp=t / 2.07 + 25, ts=ts)


def parse_log(path):
    with open(path, 'rb') as f:
        buf = f.read()
    if buf[:8] != MAGIC:
        raise SystemExit('これは .asdt1 ファイルではありません: ' + path)
    jl = struct.unpack_from('<I', buf, 12)[0]
    meta = json.loads(buf[16:16 + jl].decode('utf-8'))
    o = 16 + jl
    frames, gps, marks, infos = [], [], [], []
    while o + 12 <= len(buf):
        if struct.unpack_from('<I', buf, o)[0] != REC_MAGIC:
            o += 1
            continue
        typ = buf[o + 4]
        ln = struct.unpack_from('<I', buf, o + 8)[0]
        p = o + 12
        if p + ln > len(buf):
            print('警告: 末尾の不完全なレコードを無視します', file=sys.stderr)
            break
        if typ == REC_FRAME:
            (frame_no, host_end, mono, tm, ts, temp, mode_code, flags, n_points, n_imu, tm_end, ts_end,
             gps_t, lat, lon, alt, acc, speed, heading) = struct.unpack_from('<IddIIhBBHHIIdddffff', buf, p)
            mp = buf[p + FRAME_FIXED: p + FRAME_FIXED + n_points * 15]
            imu = [buf[p + FRAME_FIXED + n_points * 15 + i * 16: p + FRAME_FIXED + n_points * 15 + (i + 1) * 16] for i in range(n_imu)]
            frames.append(dict(frame_no=frame_no, host_end=host_end, mono=mono, tm=tm, ts=ts, temp=temp, mode_code=mode_code,
                               flags=flags, n_points=n_points, tm_end=tm_end, ts_end=ts_end, gps_t=gps_t, lat=lat, lon=lon,
                               alt=alt, acc=acc, speed=speed, heading=heading, mp=mp, imu=imu))
        elif typ == REC_GPS:
            t, lat, lon, alt, acc, speed, heading, alt_acc = struct.unpack_from('<dddfffff', buf, p)
            gps.append(dict(t=t, lat=lat, lon=lon, alt=alt, acc=acc, speed=speed, heading=heading, alt_acc=alt_acc))
        elif typ in (REC_MARK, REC_INFO):
            t = struct.unpack_from('<d', buf, p)[0]
            text = buf[p + 8: p + ln].decode('utf-8', errors='replace')
            (marks if typ == REC_MARK else infos).append(dict(t=t, text=text))
        o = p + ln
    return meta, frames, gps, marks, infos


def iso_local(epoch_ms, tz_min):
    tz = datetime.timezone(datetime.timedelta(minutes=tz_min))
    return datetime.datetime.fromtimestamp(epoch_ms / 1000, tz).isoformat(timespec='milliseconds')


def write_ply(path, cam_mm):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('ply\nformat ascii 1.0\ncomment AS-DT1 Mobile; axes x=right y=down z=forward [mm]; picture order 24x24 row-major\n')
        f.write('element vertex %d\nproperty float x\nproperty float y\nproperty float z\nend_header\n' % len(cam_mm))
        for x, y, z in cam_mm:
            f.write('%.2f %.2f %.2f\n' % (x, y, z))


def main():
    ap = argparse.ArgumentParser(description='AS-DT1 Mobile .asdt1 converter')
    ap.add_argument('input')
    ap.add_argument('-o', '--out', default=None, help='出力フォルダ（既定: 入力ファイル名のフォルダ）')
    ap.add_argument('--no-ply', action='store_true', help='PLYを書かない')
    ap.add_argument('--no-second', action='store_true', help='第2反射へのフォールバックをしない')
    ap.add_argument('--flip-table', action='store_true', help='並び替え表を反転して展開（検証用）')
    args = ap.parse_args()
    out = args.out or os.path.splitext(os.path.basename(args.input))[0]
    os.makedirs(out, exist_ok=True)
    meta, frames, gps, marks, infos = parse_log(args.input)
    tz_min = meta.get('timezoneOffsetMin', 540)
    dev = meta.get('device', {})
    print('機器: %s S/N %s FW %s レンジ %s / コマ数 %d / GPS測位 %d / マーカー %d' % (
        dev.get('model'), dev.get('serial'), dev.get('firmware'), dev.get('mode'), len(frames), len(gps), len(marks)))
    if dev.get('simulation'):
        print('注意: これは模擬センサーの記録です（実機データではありません）')
    N = len(frames)
    xyz = np.zeros((N, 576, 3), np.float32)
    xyz2 = np.zeros((N, 576, 3), np.float32)
    flag = np.zeros((N, 576), np.uint8)
    host = np.zeros(N); tm = np.zeros(N, np.int64); ts = np.zeros(N, np.int64); temp = np.zeros(N, np.int16)
    npts = np.zeros(N, np.int16); fno = np.zeros(N, np.int64)
    glat = np.full(N, np.nan); glon = np.full(N, np.nan); galt = np.full(N, np.nan); gacc = np.full(N, np.nan)
    gspd = np.full(N, np.nan); ghdg = np.full(N, np.nan); gt = np.full(N, np.nan)
    if not args.no_ply:
        os.makedirs(os.path.join(out, 'ply'), exist_ok=True)
    imu_rows = []
    with open(os.path.join(out, 'times.csv'), 'w', newline='', encoding='utf-8') as ft:
        w = csv.writer(ft)
        w.writerow(['seq', 'frame', 'host_time_local', 'host_epoch_ms', 'tm_ms', 'ts', 'tm_imu_end_ms', 'temp_C', 'n_points', 'n_valid', 'n_second_peak',
                    'z_median_m', 'z_min_m', 'z_max_m', 'gps_epoch_ms', 'lat', 'lon', 'alt_m', 'gps_acc_m', 'speed_mps', 'heading_deg', 'n_imu'])
        for i, fr in enumerate(frames):
            n = fr['n_points']
            raw = unpack_xyz(fr['mp'], n)
            p, p2, fl = to_picture(raw, n, use_second=not args.no_second, flip_table=args.flip_table)
            xyz[i] = p / 1000; xyz2[i] = p2 / 1000; flag[i] = fl
            host[i] = fr['host_end']; tm[i] = fr['tm']; ts[i] = fr['ts']; temp[i] = fr['temp']; npts[i] = n; fno[i] = fr['frame_no']
            glat[i] = fr['lat']; glon[i] = fr['lon']; galt[i] = fr['alt']; gacc[i] = fr['acc']; gspd[i] = fr['speed']; ghdg[i] = fr['heading']; gt[i] = fr['gps_t']
            zv = p[fl > 0, 2] / 1000
            w.writerow([i + 1, fr['frame_no'], iso_local(fr['host_end'], tz_min), '%.0f' % fr['host_end'], fr['tm'], fr['ts'],
                        '' if fr['tm_end'] == 0xFFFFFFFF else fr['tm_end'], fr['temp'], n, int((fl > 0).sum()), int((fl == 2).sum()),
                        '%.3f' % np.median(zv) if zv.size else '', '%.3f' % zv.min() if zv.size else '', '%.3f' % zv.max() if zv.size else '',
                        '' if np.isnan(fr['gps_t']) else '%.0f' % fr['gps_t'], '' if np.isnan(fr['lat']) else '%.7f' % fr['lat'],
                        '' if np.isnan(fr['lon']) else '%.7f' % fr['lon'], '' if np.isnan(fr['alt']) else '%.1f' % fr['alt'],
                        '' if np.isnan(fr['acc']) else '%.1f' % fr['acc'], '' if np.isnan(fr['speed']) else '%.2f' % fr['speed'],
                        '' if np.isnan(fr['heading']) else '%.0f' % fr['heading'], len(fr['imu'])])
            for j, rec in enumerate(fr['imu']):
                d = decode_imu(rec)
                imu_rows.append([fr['frame_no'], j, '0x%02X' % d['head'], '%.4f' % d['ax'], '%.4f' % d['ay'], '%.4f' % d['az'],
                                 '%.2f' % d['gx'], '%.2f' % d['gy'], '%.2f' % d['gz'], '%.1f' % d['temp'], d['ts']])
            if not args.no_ply:
                cam = np.stack([-p[:, 1] + 0.0, p[:, 0], p[:, 2]], axis=1)   # x=右(−Y), y=下(X), z=前(Z)  (+0.0 で -0.00 表記を避ける)
                write_ply(os.path.join(out, 'ply', 'frame_%04d.ply' % (i + 1)), cam)   # 連番は記録内の通し番号(seq)
    xyz_cam = np.stack([-xyz[:, :, 1], xyz[:, :, 0], xyz[:, :, 2]], axis=2)
    np.savez_compressed(os.path.join(out, 'frames.npz'), xyz_pic=xyz, xyz_cam=xyz_cam, xyz2_pic=xyz2, flag=flag, host_epoch_ms=host,
                        tm_ms=tm, ts=ts, temp_C=temp, n_points=npts, frame_no=fno, gps_epoch_ms=gt, gps_lat=glat, gps_lon=glon,
                        gps_alt=galt, gps_acc=gacc, gps_speed=gspd, gps_heading=ghdg)
    with open(os.path.join(out, 'imu.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['frame', 'idx', 'head', 'ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps', 'temp_C', 'ts_us']); w.writerows(imu_rows)
    with open(os.path.join(out, 'markers.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['time_local', 'epoch_ms', 'text', 'lat', 'lon'])
        for m in marks:
            g = nearest_gps(gps, m['t'])
            w.writerow([iso_local(m['t'], tz_min), '%.0f' % m['t'], m['text'], '' if g is None else '%.7f' % g['lat'], '' if g is None else '%.7f' % g['lon']])
    with open(os.path.join(out, 'track.gpx'), 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n<gpx version="1.1" creator="asdt1_convert" xmlns="http://www.topografix.com/GPX/1/1">\n')
        f.write('<trk><name>%s</name><trkseg>\n' % meta.get('name', 'AS-DT1'))
        for g in gps:
            f.write('<trkpt lat="%.7f" lon="%.7f">%s<time>%s</time></trkpt>\n' % (
                g['lat'], g['lon'], '' if np.isnan(g['alt']) else '<ele>%.1f</ele>' % g['alt'],
                datetime.datetime.fromtimestamp(g['t'] / 1000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'))
        f.write('</trkseg></trk>\n')
        for m in marks:
            g = nearest_gps(gps, m['t'])
            if g is not None:
                f.write('<wpt lat="%.7f" lon="%.7f"><name>%s</name><time>%s</time></wpt>\n' % (
                    g['lat'], g['lon'], m['text'].replace('&', '&amp;').replace('<', '&lt;'),
                    datetime.datetime.fromtimestamp(m['t'] / 1000, datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')))
        f.write('</gpx>\n')
    valid = flag > 0
    summary = dict(input=os.path.basename(args.input), device=dev, mode=dev.get('mode'), n_frames=N, n_gps=len(gps), n_markers=len(marks),
                   start_local=iso_local(host[0], tz_min) if N else None, end_local=iso_local(host[-1], tz_min) if N else None,
                   duration_s=round(float(host[-1] - host[0]) / 1000, 1) if N > 1 else 0,
                   fps_mean=round(float((N - 1) / ((host[-1] - host[0]) / 1000)), 2) if N > 1 else None,
                   valid_ratio_mean=round(float(valid.mean()), 4) if N else None,
                   z_median_m=round(float(np.median(xyz[:, :, 2][valid])), 3) if valid.any() else None,
                   temp_C_range=[int(temp.min()), int(temp.max())] if N else None, flip_table=args.flip_table, infos=infos,
                   axes='xyz_pic: X=down Y=left Z=forward (sensor); xyz_cam / ply: x=right y=down z=forward (same as lidar-viewer)')
    with open(os.path.join(out, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(dict(summary, meta=meta), f, ensure_ascii=False, indent=1)
    print('出力: %s/ (frames.npz, times.csv, imu.csv, markers.csv, track.gpx, summary.json%s)' % (out, '' if args.no_ply else ', ply/'))
    print('コマ数 %d, 平均 %.2f コマ/秒, 有効点率 %.1f%%, 距離中央値 %s m' % (
        N, summary['fps_mean'] or 0, 100 * (summary['valid_ratio_mean'] or 0), summary['z_median_m']))


def nearest_gps(gps, t):
    if not gps:
        return None
    return min(gps, key=lambda g: abs(g['t'] - t))


if __name__ == '__main__':
    main()
