from flask import Flask,render_template,jsonify,request
import pandas as pd
import os
app=Flask(__name__)
app.config['DATA_FOLDER']=os.path.abspath('data_files')

def save_file(file):
    filename=file.filename
    file.save(os.path.join(app.config['DATA_FOLDER'],filename))

def delete_file(file):
    upload_dir=app.config['DATA_FOLDER']
    target_path=upload_dir+'\\'+file
    if os.path.isfile(target_path):
        os.remove(target_path)

@app.route('/')
def mainframe():
    return render_template('mainframe.html')

@app.route('/api/save_file',methods=['POST'])
def api_save_file():
    file=request.files['file']
    result=save_file(file)
    return jsonify({
        'status':'success',
        'result':result
    })

@app.route('/api/delete_file',methods=['POST'])
def api_delete_file():
    file=request.form.get('filename')
    result=delete_file(file)
    return jsonify({
        'status':'success',
        'result':result
    })

@app.route('/preview/<filename>')
def preview(filename):
    upload_dir=app.config['DATA_FOLDER']
    target_path=upload_dir+'\\'+filename
    if filename.endswith('.csv'):
        with open(target_path,'r',encoding='UTF-8') as fr:
            df=pd.read_csv(fr)
    else :
        with open(target_path,'r',encoding='UTF-8') as fr:
            df=pd.read_excel(fr)
    table_html=df.head(50).to_html(index=False)
    return render_template('preview.html',table=table_html)


if __name__=='__main__':
    if not os.path.exists(app.config['DATA_FOLDER']):
        os.makedirs(app.config['DATA_FOLDER'])
    app.run(debug=True)