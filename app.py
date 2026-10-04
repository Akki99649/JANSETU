from flask import Flask, render_template, jsonify, request, redirect, url_for, session
import sqlite3
from pathlib import Path
import json

from ml_model import train_model, predict_risk

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'jansetu.db'

app = Flask(__name__)
app.secret_key = 'jansetu-prototype-secret-key-change-me'


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS villages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state TEXT NOT NULL,
            district TEXT NOT NULL,
            village TEXT NOT NULL,
            scheme TEXT NOT NULL,
            eligible INTEGER NOT NULL,
            applications INTEGER NOT NULL,
            approved INTEGER NOT NULL,
            received INTEGER NOT NULL,
            pending INTEGER NOT NULL,
            dropout_rate REAL NOT NULL,
            service_distance_km REAL NOT NULL,
            processing_days REAL NOT NULL,
            anomaly_score REAL NOT NULL
        )
    ''')
    count = conn.execute('SELECT COUNT(*) AS c FROM villages').fetchone()['c']
    if count == 0:
        seed_data(conn)
    conn.commit()
    conn.close()


def seed_data(conn):
    rows = [
        ('Karnataka','Bengaluru Rural','Devanahalli','Skill Development',5200,4100,3700,3010,1090,18,12.5,16,0.21),
        ('Karnataka','Bengaluru Rural','Doddaballapur','Skill Development',4800,3900,3500,3200,700,12,8.2,11,0.10),
        ('Karnataka','Mysuru','Hunsur','Social Welfare',6200,5000,4200,3000,2000,24,18.0,21,0.34),
        ('Karnataka','Mysuru','Nanjangud','Social Welfare',5600,4700,4300,3900,800,11,6.5,10,0.08),
        ('Maharashtra','Pune','Baramati','Education',7000,6100,5500,5100,1000,9,5.1,9,0.06),
        ('Maharashtra','Pune','Junnar','Education',5900,4300,3700,2500,1800,29,21.0,24,0.42),
        ('Tamil Nadu','Madurai','Melur','Healthcare',6800,5100,4400,3100,2000,27,19.0,22,0.37),
        ('Tamil Nadu','Madurai','Thirumangalam','Healthcare',6100,5200,4800,4500,700,8,7.0,8,0.05),
        ('Odisha','Ganjam','Chhatrapur','Housing',8200,6300,5200,3500,2800,31,25.0,27,0.48),
        ('Odisha','Ganjam','Berhampur Rural','Housing',7600,6700,6100,5700,1000,10,9.0,12,0.07),
        ('Rajasthan','Jaipur','Chomu','Agriculture',7300,6000,5200,4200,1800,20,15.0,18,0.28),
        ('Rajasthan','Jaipur','Sambhar','Agriculture',6500,4800,4000,2700,2100,26,23.0,25,0.40),
    ]
    conn.executemany('''
        INSERT INTO villages
        (state,district,village,scheme,eligible,applications,approved,received,pending,dropout_rate,service_distance_km,processing_days,anomaly_score)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    ''', rows)


MODEL = None
FEATURES = []


def rows_for_query(state=None, district=None, scheme=None):
    conn = get_db()
    sql = 'SELECT * FROM villages WHERE 1=1'
    params = []
    if state and state != 'All':
        sql += ' AND state=?'; params.append(state)
    if district and district != 'All':
        sql += ' AND district=?'; params.append(district)
    if scheme and scheme != 'All':
        sql += ' AND scheme=?'; params.append(scheme)
    sql += ' ORDER BY state, district, village'
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def enrich(rows):
    result = []
    for r in rows:
        risk, label = predict_risk(MODEL, r, FEATURES)
        gap = max(0, 100 * (1 - (r['received'] / max(r['eligible'], 1))))
        approval_rate = 100 * r['approved'] / max(r['applications'], 1)
        receipt_rate = 100 * r['received'] / max(r['approved'], 1)
        result.append({**r, 'gap': round(gap,1), 'approval_rate': round(approval_rate,1),
                       'receipt_rate': round(receipt_rate,1), 'risk': round(risk,1), 'risk_label': label})
    return result


@app.route('/')
def index():
    return redirect(url_for('dashboard'))


@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username','')
        password = request.form.get('password','')
        if username == 'admin' and password == 'admin123':
            session['user'] = username
            return redirect(url_for('dashboard'))
        return render_template('login.html', error='Invalid demo credentials')
    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session['user'])


@app.get('/api/filters')
def filters():
    conn = get_db()
    states = [r[0] for r in conn.execute('SELECT DISTINCT state FROM villages ORDER BY state').fetchall()]
    districts = [r[0] for r in conn.execute('SELECT DISTINCT district FROM villages ORDER BY district').fetchall()]
    schemes = [r[0] for r in conn.execute('SELECT DISTINCT scheme FROM villages ORDER BY scheme').fetchall()]
    conn.close()
    return jsonify({'states': states, 'districts': districts, 'schemes': schemes})


@app.get('/api/overview')
def overview():
    rows = enrich(rows_for_query(request.args.get('state'), request.args.get('district'), request.args.get('scheme')))
    eligible = sum(r['eligible'] for r in rows)
    applications = sum(r['applications'] for r in rows)
    approved = sum(r['approved'] for r in rows)
    received = sum(r['received'] for r in rows)
    pending = sum(r['pending'] for r in rows)
    gap = 100 * (1 - received / max(eligible,1))
    high_risk = sum(1 for r in rows if r['risk'] >= 70)
    return jsonify({
        'eligible': eligible, 'applications': applications, 'approved': approved,
        'received': received, 'pending': pending, 'gap': round(gap,1),
        'high_risk_villages': high_risk, 'villages': len(rows), 'rows': rows
    })


@app.get('/api/villages')
def villages():
    rows = enrich(rows_for_query(request.args.get('state'), request.args.get('district'), request.args.get('scheme')))
    return jsonify(rows)


@app.get('/api/alerts')
def alerts():
    rows = enrich(rows_for_query())
    alerts = []
    for r in rows:
        if r['risk'] >= 70:
            alerts.append({'type':'critical','title':f"High delivery-gap risk: {r['village']}",
                           'text':f"{r['gap']}% estimated gap; {r['pending']} applications pending."})
        elif r['anomaly_score'] >= 0.30:
            alerts.append({'type':'warning','title':f"Anomaly detected: {r['village']}",
                           'text':f"Unusual service pattern detected in {r['district']}."})
    return jsonify(alerts[:8])


@app.post('/api/simulate')
def simulate():
    data = request.get_json(silent=True) or {}
    extra_centres = max(0, int(data.get('extra_centres', 0)))
    mobile_camps = max(0, int(data.get('mobile_camps', 0)))
    rows = enrich(rows_for_query(data.get('state'), data.get('district'), data.get('scheme')))
    current_gap = sum(r['gap'] for r in rows) / max(len(rows),1)
    current_days = sum(r['processing_days'] for r in rows) / max(len(rows),1)
    improvement = min(0.65, extra_centres * 0.08 + mobile_camps * 0.05)
    predicted_gap = max(0, current_gap * (1 - improvement))
    predicted_days = max(1, current_days * (1 - improvement * 0.75))
    extra_apps = round(sum(r['eligible'] for r in rows) * improvement * 0.08)
    return jsonify({
        'current_gap': round(current_gap,1),
        'predicted_gap': round(predicted_gap,1),
        'current_days': round(current_days,1),
        'predicted_days': round(predicted_days,1),
        'gap_reduction': round(current_gap - predicted_gap,1),
        'estimated_extra_applications': extra_apps,
        'recommendation': 'Deploy mobile camps first, then add service-centre capacity in the highest-risk villages.' if mobile_camps else 'Add mobile service camps to reduce access barriers.'
    })


@app.get('/api/report')
def report():
    rows = enrich(rows_for_query())
    return jsonify({'generated_for':'JanSetu AI prototype','rows':rows})


init_db()
MODEL, FEATURES = train_model(DB_PATH)


if __name__ == '__main__':
    print('\nJanSetu AI running at http://127.0.0.1:5000')
    print('Demo login: admin / admin123\n')
    app.run(debug=True)
