from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import json
import plotly

app = Flask(__name__)

# Load dataset once
df_raw = pd.read_csv('marketing_campaign.csv', sep='\t')
df = df_raw.copy()

# Preprocessing
df = df.dropna()
df['Age'] = 2025 - df['Year_Birth']
df['Income'] = df['Income'].fillna(df['Income'].median())

# UI filter references
education_levels = sorted(df['Education'].unique())
kid_options = sorted(df['Kidhome'].unique())
teen_options = sorted(df['Teenhome'].unique())
anak_options = sorted(set([k + t for k in kid_options for t in teen_options]))

@app.route('/')
def index():
    return render_template('index.html', education_levels=education_levels, kid_options=kid_options, anak_options=anak_options)

@app.route('/result', methods=['POST'])
def result():
    try:
        # Ambil input user
        usia_min = int(request.form['usia_min'])
        usia_max = int(request.form['usia_max'])
        income_min = int(request.form['income_min'])
        income_max = int(request.form['income_max'])
        pendidikan = request.form['pendidikan']
        anak = int(request.form.get('anak', 0))
        komplain = 'complain' in request.form

        # Filter dataset berdasarkan input
        df_filtered = df[
            (df['Age'] >= usia_min) &
            (df['Age'] <= usia_max) &
            (df['Income'] >= income_min) &
            (df['Income'] <= income_max) &
            (df['Education'] == pendidikan) &
            ((df['Kidhome'] + df['Teenhome']) == anak) &
            (df['Complain'] == int(komplain))
        ]

        print('DEBUG: Filtered Data:')
        print(df_filtered[['Age','Income','Education','Kidhome','Teenhome','Complain']])

        if df_filtered.empty:
            return render_template('error.html', message=f"Tidak ditemukan data untuk filter: usia {usia_min}-{usia_max}, income {income_min}-{income_max}, pendidikan {pendidikan}, anak {anak}, komplain {komplain}. Silakan coba kombinasi input lain.")

        # Fitur yang digunakan untuk clustering
        features = ['Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts']
        X = df_filtered[features]

        # Normalisasi
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Penentuan K optimal
        wcss = []
        silhouettes = []
        range_k = range(2, 7)
        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_scaled)
            wcss.append(kmeans.inertia_)
            silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

        # Gunakan k terbaik
        k_optimal = range_k[silhouettes.index(max(silhouettes))]
        kmeans = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        df_filtered['Cluster'] = labels

        # Visualisasi Cluster dengan Plotly
        scatter = go.Scatter(
            x=df_filtered['Income'],
            y=df_filtered['MntWines'],
            mode='markers',
            marker=dict(color=labels, colorscale='Viridis'),
            text=[f"Cluster: {c}" for c in labels]
        )

        layout = go.Layout(
            title="Visualisasi Clustering (Income vs MntWines)",
            xaxis=dict(title='Pendapatan'),
            yaxis=dict(title='Pengeluaran Wine'),
            hovermode='closest'
        )

        plot_json = json.dumps([scatter], cls=plotly.utils.PlotlyJSONEncoder)
        layout_json = json.dumps(layout, cls=plotly.utils.PlotlyJSONEncoder)

        # Elbow dan Silhouette Plot
        elbow_plot = go.Figure()
        elbow_plot.add_trace(go.Scatter(x=list(range_k), y=wcss, mode='lines+markers', name='WCSS'))
        elbow_plot.update_layout(title='Elbow Method - WCSS', xaxis_title='Jumlah Cluster', yaxis_title='WCSS')
        elbow_html = json.dumps(elbow_plot, cls=plotly.utils.PlotlyJSONEncoder)

        silhouette_plot = go.Figure()
        silhouette_plot.add_trace(go.Scatter(x=list(range_k), y=silhouettes, mode='lines+markers', name='Silhouette Score'))
        silhouette_plot.update_layout(title='Silhouette Score vs Cluster', xaxis_title='Jumlah Cluster', yaxis_title='Silhouette Score')
        silhouette_html = json.dumps(silhouette_plot, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template(
            'result.html',
            plot_data=plot_json,
            plot_layout=layout_json,
            elbow_data=elbow_html,
            silhouette_data=silhouette_html,
            tables=df_filtered[['Age', 'Income', 'Education', 'Kidhome', 'Teenhome', 'Complain', 'Cluster']].to_html(classes='table table-bordered', index=False)
        )
    except Exception as e:
        return render_template('error.html', message=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
