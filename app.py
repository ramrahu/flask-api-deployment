from flask import Flask, request, render_template
import model

app = Flask(__name__)
# client_model = pickle.load(open('model.pkl','rb'))


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')


@app.route('/get_results', methods=['POST'])
def predict():
    user_id = [id for id in request.form.values()]
    recommendations = model.get_recommend(user_id[0])
    filtered_recommendations = model.sentiment_based_filtering(recommendations)
    return str(filtered_recommendations)


if __name__=='__main__':
    app.run(debug=True)