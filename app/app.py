from flask import Flask, render_template, flash, request, redirect
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from flask_pymongo import PyMongo
import functions
import numpy
import scipy


# App config.
DEBUG = True
app = Flask(__name__, static_url_path='/static')
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '1111111'



class ReusableForm(Form):
    book_input_1 = TextField('book input 1:', validators=[validators.required()])

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)

    print
    form.errors

    import re

    def clean(a):
        return a.lower().title()

    if request.method == 'POST':
        book_input_1 = clean(request.form['book_input_1']).lower()
        book_input_2 = clean(request.form['book_input_2']).lower()
        book_input_3 = clean(request.form['book_input_3']).lower()
        book_input_4 = clean(request.form['book_input_4']).lower()
        book_input_5 = clean(request.form['book_input_5']).lower()

        rating_input_1 = request.form['rating_input_1']
        rating_input_2 = request.form['rating_input_2']
        rating_input_3 = request.form['rating_input_3']
        rating_input_4 = request.form['rating_input_4']
        rating_input_5 = request.form['rating_input_5']

        print(book_input_1)
        book_input_list = [book_input_1, book_input_2, book_input_3, book_input_4, book_input_5]
        rating_input_list = [rating_input_1, rating_input_2, rating_input_3, rating_input_4, rating_input_5]

        #--------------------------------------------------------------------------------------------------------------


        # Add user--------------------------------------
        # coding: utf-8

        # In[1]:

        import pandas as pd

        # In[3]:


        book_df = pd.read_csv("static/data/raw/books_small.csv")

        # only get books with book_ids whose most counted book_tags is in the non_fiction tag list:
        # non_fiction_tags_df = pd.read_csv("static/data/raw/non_fiction_tags.csv")
        # book_tags_df = pd.read_csv("static/data/raw/book_tags.csv")
        # book_tags_df = book_tags_df.groupby("count").first()
        # book_tags_df = book_tags_df[book_tags_df.tag_id.isin(non_fiction_tags_df.tag_id.unique())]
        # book_df = book_df[book_df.book_id.isin(book_tags_df.goodreads_book_id.unique())]

        # drop rows that have null in "title"
        book_df = book_df.dropna(axis=0, subset=["title"])

        # lowercase all titles in book_df, getting ready for searching for user input
        book_df.title = book_df.title.str.lower()

        print(len(book_df))

        # In[4]:

        rating_df = pd.read_csv(
            "static/data/raw/book_ratings_data_set_small.csv")
        # only keep rows in rating_df if the book_id is inside book_df:
        rating_df = rating_df[rating_df.book_id.isin(book_df.book_id.unique())]


        # create new user
        # add 1 to max user_id to create new user_id
        user_id = rating_df["user_id"].max() + 1
        user_list = []
        book_list = []
        rating_list = []

        # In[6]:

        # book_input = "Biker gangs@"
        # rating_input = "3"

        # In[7]:

        def rate_a_book():

            # book_input and rating_input are already taken from html form
            # use book_input to get book_id

            for i in range(len(book_input_list)):
                book_input = book_input_list[i]
                rating_input = rating_input_list[i]
                book_id = print(book_df[book_df["title"] == book_input]["book_id"].tolist()[0])


                user_list.append(user_id)
                book_list.append(book_id)
                rating_list.append(rating_input)

        # In[8]:

        # In[11]:

        new_rating_df = pd.DataFrame(
            {
                "user_id": user_list,
                "book_id": book_list,
                "value": rating_list
            })

        all_rating_df = rating_df.append(new_rating_df).reset_index(drop=True)

        # In[12]:

        print("Great! We have all your ratings.")
        print("Here are our recommendations for you:")

        # modeling------------------------------------------

        # coding: utf-8

        # In[1]:

        import numpy as np
        import pandas as pd
        import pickle
        import functions

        # Load user ratings
        raw_dataset_df = all_rating_df
        # raw_dataset_df = raw_dataset_df.append({"user_id":101, "book_id":2, "value":4}, ignore_index=True)
        # print(raw_dataset_df.tail())

        # Convert the running list of user ratings into a matrix
        ratings_df = pd.pivot_table(raw_dataset_df, index='user_id', columns='book_id', aggfunc=np.max)

        # Normalize the ratings (center them around their mean)
        normalized_ratings, means = functions.normalize_ratings(ratings_df.values)

        # In[2]:

        # Apply matrix factorization to find the latent features
        U, M = functions.low_rank_matrix_factorization(normalized_ratings,
                                                       num_features=11,
                                                       regularization_amount=1.1)

        # In[3]:

        # Find all predicted ratings by multiplying U and M
        predicted_ratings = np.matmul(U, M)

        # Add back in the mean ratings for each product to de-normalize the predicted results
        predicted_ratings = predicted_ratings + means

        # Save features and predicted ratings to files for later use
        pickle.dump(U, open("user_features.dat", "wb"))
        pickle.dump(M, open("product_features.dat", "wb"))
        pickle.dump(predicted_ratings, open("predicted_ratings.dat", "wb"))
        pickle.dump(means, open("means.dat", "wb"))

        # recommend------------------------------------------------

        # coding: utf-8

        # In[1]:


        import pickle
        import pandas as pd

        # Load prediction rules from data files
        U = pickle.load(open("user_features.dat", "rb"))
        M = pickle.load(open("product_features.dat", "rb"))
        predicted_ratings = pickle.load(open("predicted_ratings.dat", "rb"))

        print("books we will recommend:")
        print(predicted_ratings[-1])
        user_ratings = predicted_ratings[-1]

        print(len(book_df))
        print(len(user_ratings))
        book_df['rating'] = user_ratings
        book_df = book_df.sort_values(by=['rating'], ascending=False)

        output_df = book_df[['title', 'rating']].head(10)

        #--------------------------------------------------------------------------------------------------------------

        print
        "Great."

        if form.validate():
            # print out user inputs
            flash("--------------------------------------------------------------------------------------------------------------")
            flash("Here are the books you just rated:")
            for i in range(len(book_input_list)):
                if book_input_list[i]!="" and rating_input_list[i]!="":
                    flash("book: " + book_input_list[i].title())
                    flash("Your rating: " + rating_input_list[i])
            flash("--------------------------------------------------------------------------------------------------------------")

            # print out recommendations
            flash("And here are our recommendations for you based on your ratings:")
            for index, row in output_df.iterrows():
                if str(row.title)!="nan":
                    flash("book: " + str(row.title).title())
                    flash("----")
            flash("--------------------------------------------------------------------------------------------------------------")

        else:
            flash('Error: All the form fields are required. ')

        return render_template('index.html', form=form, book_input_1=book_input_1, book_input_2=book_input_2, book_input_3=book_input_3, book_input_4=book_input_4, book_input_5=book_input_5)

    else:
        return render_template("index.html", form=form)



if __name__ == "__main__":
    app.run()