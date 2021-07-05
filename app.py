from flask import Flask, render_template, request
import pickle
import numpy as np

model=pickle.load(open('insurance_xgboost_pickle_model.pkl','rb'))

app = Flask(__name__)
@app.route('/output',methods=['POST'])
def predict():
    pap=float(request.form['policy_annual_premium'])
    
    ins=int(request.form['incident_severity'])
    
    nv=int(request.form['number_of_vehicles_involved'])

    bi=int(request.form['bodily_injuries'])

    tca=int(request.form['total_claim_amount'])
    
    ic=int(request.form['injury_claim'])

    pc=int(request.form['property_claim'])
    
    vc=int(request.form['vehicle_claim'])

    pby=int(request.form['policy_bind_year'])

    ac=int(request.form['authorities_contacted'])
    police=0
    fire=0
    ac_other=0
    ac_none=0
    if ac==1:
        police=1
    elif ac==2:
        fire=1
    elif ac==3:
        ac_other=1
    elif ac==5:
        ac_none=1
    else:
        pass
    
    pd=int(request.form['property_damage'])
    pdy=0
    pdu=0
    if pd==1:
        pdy=1
    elif pd==2:
        pdu=1
    else:
        pass 

    it=int(request.form['incident_type'])
    car=0
    single=0
    theft=0
    if it==1:
        car=1
    elif it==2:
        single=1
    elif it==3:
        theft=1
    else:
        pass
    
    ih=int(request.form['insured_hobbies'])
    reading=0
    paintball=0
    exercise=0
    bungie_jumping=0
    camping=0
    movies=0
    golf=0
    kayaking=0
    yachting=0
    hiking=0
    video_games=0
    skydiving=0
    board_games=0
    polo=0
    chess=0
    dancing=0
    sleeping=0
    cross_fit=0
    basketball=0
    if ih==1:
        reading=1
    elif ih==2:
        paintball=1
    elif ih==3:
        exercise=1
    elif ih==4:
        bungie_jumping=1
    elif ih==5:
        camping=1
    elif ih==6:
        movies=1
    elif ih==7:
        golf=1
    elif ih==8:
        kayaking=1
    elif ih==9:
        yachting=1
    elif ih==10:
        hiking=1
    elif ih==11:
        video_games=1
    elif ih==13:
        skydiving=1
    elif ih==14:
        board_games=1
    elif ih==15:
        polo=1
    elif ih==16:
        chess=1
    elif ih==17:
        dancing=1
    elif ih==18:
        sleeping=1
    elif ih==19:
        cross_fit=1
    elif ih==20:
        basketball=1
    else:
        pass
    
    pra=int(request.form['police_report_available'])
    pray=0
    prau=0
    if pra==1:
        pray=1
    elif pra==2:
        prau=1
    else:
        pass 

    lbc = tca - (pap * (2015 - pby))
    pclaim = pc * ins
    vclaim = vc * ins
    iclaim = ic * ins
    tclaim = tca * ins
    
    import numpy as np
    
    D = np.array([[pap,ins,nv,bi,tca,ic,pc,pby,lbc,pclaim,vclaim,iclaim,tclaim,fire,ac_none,ac_other,police,pdy,pdu,car,single,theft,basketball,board_games,bungie_jumping,camping,chess,cross_fit,dancing,exercise,golf,hiking,kayaking,movies,paintball,polo,reading,skydiving,sleeping,video_games,yachting,pray,prau]])
    
    pred = model.predict(D)
    
    return render_template('result.html',prediction=pred)

@app.route('/')
def home():
	return render_template('index.html')

if __name__=='__main__':
	app.run(debug=True)