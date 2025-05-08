document.getElementById('add_data').addEventListener('change',async function(e){
    const file=e.target.files[0];
    if (!file)return;
    let table=document.getElementById('table');
    let length=table.rows.length;
    let new_row=table.insertRow(length);
    let count=new_row.insertCell(0);
    let filename=new_row.insertCell(1);
    let instruction=new_row.insertCell(2);
    count.innerHTML=length;
    filename.innerHTML=file.name;
    instruction.innerHTML=`<button onclick='preview_row(this)' class='button-like'>预览</button>
                             <a href='/data_files/${file.name}' download='${file.name}' class='button-like'>导出</a> 
                             <button onclick='delete_row(this)' class='button-like'>删除</button>
                             <button onclick='visualization(this)' class="button-like">可视化</button>
                             <button onclick='prediction(this)' class="button-like">预测并保存至新文件</button>`;
    const formData=new FormData();
    formData.append('file',file);
    const result = await fetch('/api/save_file',{
        method:'POST',
        body:formData
    });
    const jsonResult = await result.json();
    console.log(jsonResult);
    e.target.value=null;
});

function delete_row(button){
    const formData=new FormData();
    filename=button.parentNode.parentNode.children[1].innerHTML;
    formData.append('filename',filename);
    fetch('/api/delete_file',{
        method:'POST',
        body:formData
    })
    let row=button.parentNode.parentNode;
    row.parentNode.removeChild(row);
}

function preview_row(button){
    let next='/preview/'+button.parentNode.parentNode.children[1].innerHTML;
    window.open(next);
}

function visualization(button) {
    filename=button.parentNode.parentNode.children[1].innerHTML;
    let next = '/visualization/' + filename;
    window.open(next);
}

async function prediction(button){
    const formData=new FormData();
    let filename=button.parentNode.parentNode.children[1].innerHTML;
    formData.append('filename',filename);
    const data=await fetch('/prediction',{
        method:'POST',
        body:formData
    })
    let table=document.getElementById('table');
    let length=table.rows.length;
    let new_row=table.insertRow(length);
    let count=new_row.insertCell(0);
    let file_name=new_row.insertCell(1);
    let instruction=new_row.insertCell(2);
    count.innerHTML=length;
    let json_data=await data.json();
    let change_name=json_data.change_name;
    file_name.innerHTML=change_name;
    instruction.innerHTML=`<button onclick='preview_row(this)' class='button-like'>预览</button>
                             <a href='/data_files/${change_name}' download='${change_name}' class='button-like'>导出</a> 
                             <button onclick='delete_row(this)' class='button-like'>删除</button>
                             <button onclick='visualization(this)' class="button-like">可视化</button>
                             <button onclick='prediction(this)' class="button-like">预测并保存至新文件</button>`;
}