import json
import base64
import folium
with open('data.json', 'r', encoding='utf-8') as f:  # 위도 경도를 넣어주자
    datas = json.load(f)
    for data in datas:
        encoded = base64.b64encode(open(data["mImagePath"], 'rb').read()).decode()
        html = '<img src="data:image/jpeg;base64,{}">'.format
        #iframe = IFrame(html(encoded), width=800 + 20, height=450 + 20)
        popup = folium.Popup(iframe, max_width=2650)

        icon = folium.Icon(color="red", icon="ok")
        marker = folium.Marker(location=[data["mLatitude"], data["mLongitude"]], popup=popup, icon=icon)
        mapa.add_child(marker)