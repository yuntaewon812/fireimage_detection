"""
map_utils.py — EXIF GPS 추출 + folium 지도 HTML 생성.
발화점 마커 + 확산방향 화살표를 Leaflet 지도에 표시.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple


def extract_gps(image_path: str) -> Optional[Tuple[float, float]]:
    """이미지 EXIF에서 (위도, 경도) 추출. 없으면 None."""
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS, GPSTAGS
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None
        gps = {}
        for tag, val in exif.items():
            if TAGS.get(tag) == "GPSInfo":
                for t, v in val.items():
                    gps[GPSTAGS.get(t, t)] = v
        if not gps:
            return None

        def _conv(dms, ref):
            d = dms[0] + dms[1] / 60.0 + dms[2] / 3600.0
            return -d if ref in ("S", "W") else d

        lat = _conv(gps["GPSLatitude"], gps["GPSLatitudeRef"])
        lon = _conv(gps["GPSLongitude"], gps["GPSLongitudeRef"])
        return float(lat), float(lon)
    except Exception:
        return None


def build_map_html(lat: float, lon: float,
                   spread_deg: Optional[float] = None,
                   grade: str = "중형") -> str:
    """folium 지도 HTML 문자열 반환 (Gradio gr.HTML 임베드용)."""
    try:
        import folium
    except ImportError:
        return f"<p>folium 미설치. 위치: {lat:.5f}, {lon:.5f}</p>"

    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    color = {"소형": "orange", "중형": "red", "대형": "darkred"}.get(grade, "red")
    folium.Marker([lat, lon], tooltip=f"발화 추정점 ({grade})",
                  icon=folium.Icon(color=color, icon="fire", prefix="fa")).add_to(m)
    folium.Circle([lat, lon], radius={"소형": 300, "중형": 800, "대형": 2000}.get(grade, 800),
                  color=color, fill=True, fill_opacity=0.15).add_to(m)

    # 확산 방향 화살표 (연기 이동방향으로 끝점 계산)
    if spread_deg is not None:
        dist = 0.012  # 약 1.3km
        rad = math.radians(spread_deg)
        end = [lat + dist * math.cos(rad), lon + dist * math.sin(rad)]
        folium.PolyLine([[lat, lon], end], color="blue", weight=3,
                        tooltip=f"확산방향 {spread_deg:.0f}°").add_to(m)

    html = m._repr_html_()
    # Gradio iframe 안에서 높이 확보
    return f'<div style="height:420px">{html}</div>'
