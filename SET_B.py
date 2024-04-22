from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to apply rules and generate recommendations
def generate_recommendations(answers):
    fruits = ['oranges', 'apples', 'pears', 'grapes', 'watermelon', 'lemon', 'lime']

    # Extract answers to questions
    party_on_weekends = answers.get('party_on_weekends', 'no')
    flavor_preference = answers.get('flavor_preference', '')
    texture_dislike = answers.get('texture_dislike', '')
    price_range = answers.get('price_range', 0)

    # Apply rules based on user answers
    if party_on_weekends.lower() == 'yes':
        fruits_allowed = ['apples', 'pears', 'grapes', 'watermelon']
    else:
        fruits_allowed = fruits

    if flavor_preference.lower() == 'cider':
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit in ['apples', 'oranges', 'lemon', 'lime']]
    elif flavor_preference.lower() == 'sweet':
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit in ['watermelon', 'oranges']]
    elif flavor_preference.lower() == 'waterlike':
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit == 'watermelon']

    if 'grapes' in fruits_allowed:
        fruits_allowed.remove('watermelon')

    if 'smooth' in texture_dislike.lower():
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit != 'pears']

    if 'slimy' in texture_dislike.lower():
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit not in ['watermelon', 'lime', 'grapes']]

    if 'waterlike' in texture_dislike.lower():
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit != 'watermelon']

    if int(price_range) < 3:
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit not in ['watermelon', 'lime']]

    if 4 < int(price_range) < 7:
        fruits_allowed = [fruit for fruit in fruits_allowed if fruit not in ['pears', 'apples']]

    return fruits_allowed

# Flask route to handle POST requests
@app.route('/recommend-fruits', methods=['POST'])
def recommend_fruits():
    data = request.json
    recommendations = generate_recommendations(data)
    return jsonify({'recommendations': recommendations})

if __name__ == '__main__':
    app.run(debug=True)
