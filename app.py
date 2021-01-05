from flask import Flask,render_template,request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder
app=Flask(__name__)

ohe=pickle.load(open('model/ohe.pkl','rb'))
locality_df=pickle.load(open('model/locality_df.pkl','rb'))
reg=pickle.load(open('model/model.pkl','rb'))
scaler=pickle.load(open('model/scaler.pkl','rb'))

status_encoder = pickle.load(open('model/status_encoder.pkl','rb'))
transaction_encoder=pickle.load(open('model/transaction_encoder.pkl','rb'))
type_encoder=pickle.load(open('model/type_encoder.pkl','rb'))


@app.route('/')
def index():
    return render_template("index.html")

# RECIEVING DATA FROM FORM WHICH GOES TO PREDICT FUNCTION. WE ARE RECIENVING THE DATA VIA POST METHOD
@app.route('/predict',methods=['post'])
def predict():
    # recieve form data here
    area = float(request.form.get('area'))
    bhk = int(request.form.get('bhk'))
    bathrooms = float(request.form.get('bathrooms'))
    status = request.form.get('status')
    transaction = request.form.get('transaction')
    property = request.form.get('property')
    locality = request.form.get('locality')

    per_sqft=locality_df[locality_df['Locality'] == locality]['Per_Sqft'].mean()

    # Onehot encode bhk and bathroom

    X_trans=ohe.transform(np.array([[bhk,bathrooms]])).toarray()

    # Label encode status,transaction and property
    status = str(status_encoder.transform([status])[0])
    transaction=str(transaction_encoder.transform([transaction])[0])
    property = str(type_encoder.transform([property])[0])

    # derive per_sqft value from locality

    per_sqft = locality_df[locality_df['Locality'] == locality]['Per_Sqft'].mean()

    X=np.array([[area,status,transaction,property,per_sqft]])

    #X = np.asarray(X, dtype='float64')

    X=np.hstack((X,X_trans))

   #X = np.asarray(X, dtype='float64')

    #print(X)

    #print(X.shape)


    X = scaler.transform(X)

    # print(X)


    y_pred=reg.predict(X)

    #print(y_pred)

    # FOR DISPLAYING THE RESULT
    return render_template('index.html',price=y_pred[0])

if __name__=="__main__":
    # IF WE KEEP DEBUG=TRUE THEN THE CHANGES ARE AUTOMATICALLY REFLECTED IN THE WEBPAGE
    app.run(debug=True)
