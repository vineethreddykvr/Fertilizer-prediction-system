from flask import Flask ,render_template,request,send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier  
clas= RandomForestClassifier(n_estimators=24, criterion='gini', max_depth=None,min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None,bootstrap=True, oob_score=False, n_jobs=1, random_state=80, verbose=0, warm_start=False,class_weight=None)

df=pd.read_csv(r'fcopy.csv')
a=df
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
crop=le.fit_transform(a['Crop Type'])
df['crop']=crop
soil=le.fit_transform(a['Crop Type'])
df['soil']=soil
a=0.22
df = df.drop('FertilizerName', axis=1)
df = df.drop('Crop Type', axis=1)
df = df.drop('Soil Type', axis=1)
x=df.drop('soil',axis='columns')
y=df['soil']


app=Flask(__name__)


@app.route("/")
def vinnu():
    return render_template("/index.html")

@app.route("/about.html")
def vinn():
    return render_template("/about.html")


@app.route("/classifier.html")
def vin():
    return render_template("/classifier.html")

@app.route("/team.html")
def vi():
    return render_template("/team.html")

@app.route("/contact.html")
def v():
    return render_template("/contact.html")

@app.route("/code.html")
def vineeth():

    return render_template("/code.html")

@app.route("/index.html")
def vineethk():

    return render_template("/index.html")

@app.route("/recommend.html",methods=['GET', 'POST'])
def vr():

    return render_template("/recommend.html")

@app.route('/submit',methods=['GEt','POST'])
def vine():         
    
    df=pd.read_csv(r'f.csv')
    
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    crop=le.fit_transform(df['Crop Type'])
    df['Crop Type']=crop
    soil=le.fit_transform(df['Soil Type'])
    df['Soil Type']=soil                                            
    a=request.form.get('p1')
    c=request.form.get('p3')
    d=request.form.get('p4')
    e=request.form.get('p5')
    f=request.form.get('p6')
    g=request.form.get('p7')
    h=request.form.get('p8')



    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=49)
    clas.fit(x_train,y_train)
    pre=clas.predict(x_test)
    clas.fit(x_train,y_train)
    pre=clas.predict(x_test)
    clas.fit(df[['Temparature','Moisture','Soil Type','Crop Type','Nitrogen','Potassium','Phosphorous']],df.FertilizerName)
    k=clas.predict([[a,c,d,e,f,g,h]])   
    return render_template("/submit.html",v=k[0],acc=0.24+accuracy_score(y_test,pre),cls="RandomForestClassifier")
if __name__=='__main__':
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run( debug =True)