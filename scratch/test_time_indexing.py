import requests

res = requests.get('https://api.open-meteo.com/v1/forecast', params={
    'latitude': 23.25,
    'longitude': 77.40,
    'timezone': 'auto',
    'current': 'temperature_2m,wind_speed_10m',
    'hourly': 'temperature_2m,precipitation_probability,wind_speed_10m',
    'forecast_days': 3
}).json()

times = res['hourly']['time']
cur_time = res['current']['time']

def find_target_index(times, cur_time, req):
    if not req or req == 'current':
        return None
    cur_date = cur_time.split('T')[0]
    all_dates = sorted(list(set(t.split('T')[0] for t in times)))
    tom_date = all_dates[1] if len(all_dates) > 1 else cur_date
    req = req.lower()
    target_date = tom_date if 'tomorrow' in req else cur_date
    if 'morning' in req: target_hour = '08:00'
    elif 'afternoon' in req: target_hour = '14:00'
    elif 'evening' in req: target_hour = '18:00'
    elif 'night' in req or 'tonight' in req: target_hour = '21:00'
    elif 'tomorrow' in req: target_hour = '12:00'
    else: return None
    target_str = f'{target_date}T{target_hour}'
    for i, t in enumerate(times):
        if t.startswith(f'{target_date}T{target_hour[:2]}'):
            return i
    return None

for r in ['current', 'this evening', 'tonight', 'tomorrow morning', 'tomorrow evening']:
    idx = find_target_index(times, cur_time, r)
    if idx is not None:
        t_val = times[idx]
        temp = res['hourly']['temperature_2m'][idx]
        rain = res['hourly']['precipitation_probability'][idx]
        wind = res['hourly']['wind_speed_10m'][idx]
        print(f"{r:20} -> {t_val} | temp: {temp}°C | rain: {rain}% | wind: {wind} km/h")
    else:
        print(f"{r:20} -> CURRENT ({cur_time}) | temp: {res['current']['temperature_2m']}°C")
