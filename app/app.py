from flask import Flask, request, jsonify
import pandas as pd
import os

STATS_CSV = 'data/stats/stats_condition_drug.csv'
MODELS_DIR = 'data/models'

app = Flask(__name__)

if os.path.exists(STATS_CSV):
    stats_df = pd.read_csv(STATS_CSV)
else:
    stats_df = pd.DataFrame(columns=['condition','drugName','mean_rating','count_reviews'])

@app.route('/suggest', methods=['GET'])
def suggest():
    condition = request.args.get('condition','').strip()
    top_n = int(request.args.get('top_n', 10))
    min_reviews = int(request.args.get('min_reviews', 1))
    if condition == '':
        return jsonify({'error':'Please provide condition parameter'}), 400
    cond = stats_df[stats_df['condition'].str.lower() == condition.lower()]
    if cond.empty:
        return jsonify({'condition':condition, 'suggestions':[]})
    cond = cond[cond['count_reviews'] >= min_reviews]
    cond = cond.sort_values(['mean_rating','count_reviews'], ascending=[False, False]).head(top_n)
    suggestions = cond.to_dict(orient='records')
    return jsonify({'condition':condition, 'suggestions':suggestions})

@app.route('/')
def index():
    return """<html><body>
    <h2>Drug Recommender - Demo</h2>
    <form action='/suggest' method='get'>
    Condition: <input type='text' name='condition'/>
    Top N: <input type='number' name='top_n' value='5'/>
    <input type='submit' value='Suggest'/>
    </form>
    </body></html>"""

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
