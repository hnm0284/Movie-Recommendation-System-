from MRS import get_recommendations, get_recommendations_genre, get_recommendations_director,get_recommendations_year
from flask import Flask, render_template, request, redirect
from datetime import datetime
import csv
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

# @app.route('/login', methods=['POST'])
# def login():
#     username = request.form['username']
#     password = request.form['password']
#     current=datetime.now()
#     with open('C:/Users/Dell/PycharmProjects/pythonProject/users.txt', 'a') as f:
#         f.write(f"{username},{password},{current}\n")
#
#     return redirect('/index')




@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    current = datetime.now()
    if not username or not password:
        message = 'Please enter a username and password'
        return render_template('login.html', message=message)
    # open the CSV file with user credentials
    with open('users.txt', newline='') as f:
        reader = csv.reader(f)
        # loop through each row in the CSV file
        for row in reader:
            # check if the provided username and password match
            if row[0] == username and row[1] == password:
                # if there is a match, record the login time
                with open('logins.csv', 'a', newline='') as login_file:
                    writer = csv.writer(login_file)
                    writer.writerow([username, current])
                # redirect the user to the home page
                return redirect('/index')

    # if the username and password don't match any records, return an error message
    message = 'Incorrect username or password'
    return render_template('login.html', message=message)


@app.route('/index')
def main():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('query')
    search_by = request.args.get('search_by')

    if search_by == 'title':
        recommendations,tag = get_recommendations(query)
        message = "Title not found, but you may like: "
        print(recommendations)
    elif search_by == 'genre':
        recommendations ,tag= get_recommendations_genre(query)
        message = "Genre not found, but you may like: "
        print(recommendations)
    elif search_by == 'director':
        recommendations,tag = get_recommendations_director(query)
        message = "Director not found, but you may like: "
        print(recommendations)
    elif search_by == 'year':
        recommendations ,tag= get_recommendations_year(query)
        message = "Year not found, but you may like: "
        print(recommendations)
    else:
        return "no data found"
        # recommendations = get_recommendations(query)
    R = list()
    R = recommendations
    R = [r.replace(' ', '+') for r in R]
    moreInfo = list()
    moreInfo = recommendations
    moreInfo = [m.replace(' ', '%20') for m in recommendations]
    current = datetime.now()
    with open('C:/Users/Dell/PycharmProjects/pythonProject/history.txt', 'a') as f:
        f.write(f"{current},{search_by},{query},{recommendations[0]},{recommendations[1]},{recommendations[2]},{recommendations[3]},{recommendations[4]}\n")
    if tag:
        return render_template('results.html', recommendations=recommendations, R=R, moreInfo=moreInfo,message=message)
    else:
        return render_template('results.html', recommendations=recommendations, R=R, moreInfo=moreInfo)

@app.route('/history')
def history():
    with open('C:/Users/Dell/PycharmProjects/pythonProject/history.txt', 'r') as f:
        history = [line.strip().split(',') for line in f.readlines()]
    return render_template('history.html', history=history)

@app.route('/logout')
def logout():

    return render_template('logout.html')




if __name__ == '__main__':
    app.run(debug=True)
