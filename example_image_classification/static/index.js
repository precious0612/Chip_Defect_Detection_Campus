
/* Event */
function distinguish() {
    var file = document.getElementById("btn_file").files[0];
    var formData = new FormData();
    if (file) {
        formData.append("file", file);
    }

    let i = document.getElementById("uploadimg");
    let img = util.convertImageToBase64(i)

    util.request.post('start', {
        img: img
    }, function (res) {
        //定义物体数组
        var dict = {'plane':"飞机", 'car':"小汽车", 'bird':"鸟", 'cat':"猫", 'deer':"鹿", 'dog':"狗", 'frog':"青蛙", 'horse':"马", 'ship':"船", 'truck':"卡车"};
        var rNumDiv = document.getElementById("result-num-div");
        var height = rNumDiv.offsetHeight;
        rNumDiv.style.cssText = "line-height:" + height + "px";
        var data = res.max_class;
        for(var k in dict){
            if(data == k)
                data = dict[k];
        }
        rNumDiv.textContent = data;
        var map = res.probability;
        var tableDiv = document.getElementById("result-table-div");
        var tab = '<table cellspacing="0" cellpadding="0">'
        tab += "<tr style='background:#EEEDED;border:1px solid #EBEEF5'>" + '<th>识别图片</th><th>置信度</th></tr>'

        for (var key in map) {
            var value = map[key];
            for(var k in dict){
                if(key == k)
                    key = dict[k];
            }
            tab += '<tr>' + "<td style='border:1px solid #EBEEF5'>" + key + "</td>" + "<td style='border:1px solid #EBEEF5'>" + value + "</td>" + '</tr>';
        }
        tab += '</table>';
        tableDiv.innerHTML = tab;
     })
}

document.addEventListener('DOMContentLoaded', function () {
    var img, container;
    img = document.getElementById("uploadimg");

    container = document.getElementById("uploadimg-div");

    var length = container.offsetHeight > container.offsetWidth ? container.offsetWidth : container.offsetHeight;

    img.width = length;

    img.height = img.width;
});

function upLoad() {
    document.getElementById("btn_file").click();
}

function changepic() {
    var reads = new FileReader();
    f = document.getElementById('btn_file').files[0];
    reads.readAsDataURL(f);
    reads.onload = function (e) {
        document.getElementById('uploadimg').src = this.result;
        document.getElementById('top-img-div').style.display = "none";
        document.getElementById('uploadimg').style.display = "block";
        document.getElementsByClassName('title-text')[0].style.display = "block";
    };
}

function clearAll() {
    var rNumDiv = document.getElementById("result-num-div");
    rNumDiv.textContent = "";

    var tableDiv = document.getElementById("result-table-div");
    tableDiv.innerHTML = "";

    var uploadImg = document.getElementById('uploadimg');
    uploadImg.src = "#";
    uploadImg.style.display = "none";
    document.getElementById("btn_file").value = "";
}



