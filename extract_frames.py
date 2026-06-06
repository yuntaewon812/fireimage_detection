"""
extract_frames.py — YouTube 영상에서 학습용 프레임 자동 추출
============================================================
링크만 넣고 실행하면: 다운로드 → 듬성 샘플링 → 중복 제거 → 폴더 저장.

핵심 설계:
  · 영상마다 고유 video_id를 파일명 접두사로 사용
      예: dQw4w9WgXcQ_f0001.jpg
    → load_data_with_groups의 그룹 정규식(^(.+?)_f\\d+)과 일치
    → 같은 영상 프레임이 train/test에 섞이지 않음 (누수 방지)
  · 연속 프레임 중복 제거: 직전 저장 프레임과 너무 비슷하면 건너뜀
  · interval초마다 1장만 후보로 → 다양성 확보

사전 준비:
  pip install yt-dlp opencv-python

사용법 1) 링크 목록 파일:
  links.txt 에 "label,URL" 형식으로 한 줄씩 작성 후:
    python extract_frames.py --file links.txt

  links.txt 예시:
    abnormal,https://www.youtube.com/watch?v=AAAAAAA
    abnormal,https://www.youtube.com/watch?v=BBBBBBB
    normal,https://www.youtube.com/watch?v=CCCCCCC

사용법 2) 단일 링크:
  python extract_frames.py --url https://youtu.be/XXXX --label normal

옵션:
  --interval 3   몇 초마다 1장 후보 (기본 3초)
  --max 40       영상당 최대 저장 장수 (기본 40)
  --dedup 0.92   중복 판정 임계 (1=완전동일, 낮출수록 더 많이 버림)
"""
import os
import sys
import argparse
import subprocess
import tempfile
import glob

import cv2
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

BASE = 'data/fireimage'   # 저장 루트
IMG_SIZE = None            # None이면 원본 해상도 유지 (학습 시 resize함)


def check_ytdlp():
    try:
        subprocess.run(['yt-dlp', '--version'], capture_output=True, check=True)
        return True
    except Exception:
        print('[오류] yt-dlp 가 없습니다.  설치:  pip install yt-dlp')
        return False


def get_video_id(url):
    """URL에서 video_id 추출 (그룹 키로 사용)."""
    import re
    m = re.search(r'(?:v=|youtu\.be/|/shorts/|/embed/)([A-Za-z0-9_-]{11})', url)
    if m:
        return m.group(1)
    # 못 찾으면 URL 해시
    return 'vid' + str(abs(hash(url)) % 10**8)


def download_video(url, tmpdir):
    """yt-dlp로 영상 다운로드 (480p 정도면 충분)."""
    out = os.path.join(tmpdir, 'video.%(ext)s')
    cmd = ['yt-dlp', '-f', 'best[height<=480]/best',
           '-o', out, '--no-playlist', url]
    print(f'  다운로드 중...')
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  [실패] {r.stderr.strip()[-300:]}')
        return None
    files = glob.glob(os.path.join(tmpdir, 'video.*'))
    return files[0] if files else None


def phash(img):
    """간단한 perceptual hash (8x8 평균 해시) — 중복 판정용."""
    g = cv2.cvtColor(cv2.resize(img, (8, 8)), cv2.COLOR_BGR2GRAY)
    return (g > g.mean()).flatten()


def hash_similarity(h1, h2):
    return (h1 == h2).mean()


def extract(video_path, vid, label, interval, max_frames, dedup_th):
    """영상에서 프레임 추출 → 저장."""
    save_dir = os.path.join(BASE, label, 'youtube_new')
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = int(fps * interval)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    saved, prev_hash, idx, fi = 0, None, 0, 0
    while saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h = phash(frame)
        # 직전 저장 프레임과 너무 비슷하면 skip
        if prev_hash is None or hash_similarity(h, prev_hash) < dedup_th:
            fi += 1
            fname = f'{vid}_f{fi:04d}.jpg'
            cv2.imwrite(os.path.join(save_dir, fname), frame)
            prev_hash = h
            saved += 1
        idx += step
        if idx >= total:
            break
    cap.release()
    print(f'  → {saved}장 저장: {save_dir}/{vid}_f####.jpg')
    return saved


def process(url, label, interval, max_frames, dedup_th):
    vid = get_video_id(url)
    print(f'[{label}] {vid}  ({url})')
    with tempfile.TemporaryDirectory() as tmp:
        vpath = download_video(url, tmp)
        if not vpath:
            return 0
        return extract(vpath, vid, label, interval, max_frames, dedup_th)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--file', help='label,URL 형식 목록 파일')
    p.add_argument('--url', help='단일 영상 URL')
    p.add_argument('--label', choices=['normal', 'abnormal'],
                   help='단일 URL일 때 레이블')
    p.add_argument('--interval', type=float, default=3.0, help='몇 초당 1장 (기본 3)')
    p.add_argument('--max', type=int, default=40, help='영상당 최대 장수 (기본 40)')
    p.add_argument('--dedup', type=float, default=0.92,
                   help='중복 임계 (기본 0.92, 낮출수록 더 버림)')
    return p.parse_args()


def main():
    args = parse_args()
    if not check_ytdlp():
        return

    jobs = []
    if args.file:
        with open(args.file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if ',' not in line:
                    print(f'[건너뜀] 형식 오류: {line}')
                    continue
                label, url = line.split(',', 1)
                jobs.append((url.strip(), label.strip()))
    elif args.url and args.label:
        jobs.append((args.url, args.label))
    else:
        print('[사용법] --file links.txt  또는  --url <URL> --label <normal|abnormal>')
        return

    print(f'\n총 {len(jobs)}개 영상 처리\n' + '='*50)
    total = {'normal': 0, 'abnormal': 0}
    nvid = {'normal': 0, 'abnormal': 0}
    for url, label in jobs:
        n = process(url, label, args.interval, args.max, args.dedup)
        if n > 0:
            total[label] += n
            nvid[label] += 1
        print()

    print('='*50)
    print('[완료]')
    print(f'  정상 : 영상 {nvid["normal"]}편, 프레임 {total["normal"]}장')
    print(f'  화재 : 영상 {nvid["abnormal"]}편, 프레임 {total["abnormal"]}장')
    print(f'  저장 위치: {BASE}/<normal|abnormal>/youtube_new/')
    print('  → 그룹 정규식 호환 (영상별 분리). 재학습 시 자동 인식됨.')


if __name__ == '__main__':
    main()
