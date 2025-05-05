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
    instruction.innerHTML="<button onclick='preview_row(this)'>预览</button> <button onclick='delete_row(this)'>删除</button>"
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
    window.open(next,'_blank');
}